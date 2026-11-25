"""FAW leaf-damage analyzer - local web app.

Run (project root):  python faw_seg\\webapp\\app.py      then open http://127.0.0.1:5000
Loads the 5-fold ensembles of the main RGB model (e00) and the RGB-thermal FiLM model (e10).
"""
import json
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

HERE = Path(__file__).resolve().parent
FAW = HERE.parent
sys.path.insert(0, str(FAW / "src"))
sys.path.insert(0, str(HERE))
from faw.config import load_config  # noqa: E402
import inference  # noqa: E402
import advice  # noqa: E402

MODELS = {"rgb": ("config_e00_baseline.yaml", "Main RGB model (e00), 5-fold ensemble"),
          "rgb_thermal": ("config_e10_fusion_film.yaml", "RGB + thermal FiLM fusion model (e10), 5-fold ensemble")}
PERF = {"rgb": {"test_mean_dice": 0.446, "test_pooled_dice": 0.497, "region_recall": 0.777},
        "rgb_thermal": {"test_mean_dice": 0.454, "test_pooled_dice": 0.503, "region_recall": 0.774}}


def load_predictors():
    import tensorflow as tf
    from faw import fusion
    predictors, settings, names = {}, {}, {}
    for mode, (cfg_name, label) in MODELS.items():
        cfg = load_config(cfg_name)
        sys.path.insert(0, str(cfg["root"] / cfg["paths"]["phase1_model_code"]))
        import model_experiment_06_mobilenetv2_unet as M
        exp = cfg["p"]["outputs"] / "experiments" / cfg["train"]["exp_name"]
        models = []
        for k in range(cfg["splits"]["n_folds"]):
            w = exp / f"fold{k}" / "best.weights.h5"
            if not w.exists():
                print(f"[skip] {label}: missing {w}")
                models = []
                break
            m = fusion.build_model(cfg, M)
            m.load_weights(str(w))
            models.append(m)
        if not models:
            continue
        is_fusion = bool((cfg.get("model") or {}).get("fusion"))

        def fn(rb, tb, models=models, is_fusion=is_fusion):
            outs = []
            for m in models:
                x = [tf.constant(rb), tf.constant(tb)] if is_fusion else tf.constant(rb)
                outs.append(m(x, training=False).numpy()[..., 0])
            return np.mean(outs, 0)

        predictors[mode] = fn
        settings[mode] = (float(cfg["final"]["threshold"]), int(cfg["final"]["min_region_px"]))
        names[mode] = label
        print(f"[ok] {label}: threshold {settings[mode][0]}, min region {settings[mode][1]} px")
    return predictors, settings, names


app = Flask(__name__, template_folder=str(HERE / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
PREDICTORS, SETTINGS, NAMES = load_predictors()


@app.get("/")
def index():
    return render_template("index.html", models=NAMES)


@app.post("/api/analyze")
def analyze():
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify({"error": "Please choose an image file."}), 400
    try:
        res = inference.analyze(f.read(), PREDICTORS, SETTINGS)
    except inference.InputError as e:
        return jsonify({"error": str(e)}), 400
    mode = res["info"]["mode"]
    res["model"] = NAMES.get(mode, mode)
    res["performance"] = PERF.get(mode)
    res["advice"] = advice.recommendations(res["damage_pct_of_leaf"])
    res["filename"] = f.filename
    return app.response_class(json.dumps(res), mimetype="application/json")


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=False)
