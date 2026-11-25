"""Step 13 - Qualitative figures from the saved test ensemble predictions (no re-evaluation, no tuning).
Panels per row: RGB | ground truth (yellow outline) | prediction (cyan outline) | error map (TP green, FP red, FN blue).
Rows: 2 lowest, 2 median, 2 highest per-image Dice (damaged) + 2 healthy images (most / least predicted area).
Output: outputs/experiments/<exp>/final_eval/figures/
Usage (project root): python faw_seg\\scripts\\13_make_figures.py
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

FAW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAW / "src"))
from faw.config import load_config  # noqa: E402
from faw import data, seg  # noqa: E402

W, H = 480, 360


def outline(img, m, col):
    o = img.copy()
    c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(o, c, -1, col, 2)
    return o


def error_map(img, pred, gt):
    e = (img * 0.45).astype(np.uint8)
    e[pred & gt] = (0, 200, 0)
    e[pred & ~gt] = (0, 0, 255)
    e[~pred & gt] = (255, 128, 0)
    return e


def label(t, text):
    t = cv2.resize(t, (W, H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (W - 1, 24), (255, 255, 255), -1)
    cv2.putText(t, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return t


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    cfg = load_config(ap.parse_args().config)
    F, T = cfg["final"], cfg["train"]
    ev = cfg["p"]["outputs"] / "experiments" / T["exp_name"] / "final_eval"
    out = ev / "figures"
    out.mkdir(exist_ok=True)
    z = np.load(ev / "test_ensemble_probs.npz")
    probs = {str(i): p for i, p in zip(z["ids"], z["probs"])}
    recs = {r["sample_id"]: r for r in data.load_records(cfg, ["test"])}
    rows = list(csv.DictReader(open(ev / "test_per_image.csv")))
    rows.sort(key=lambda r: float(r["dice"]))
    m = len(rows) // 2
    pick = [("lowest", rows[0]), ("lowest", rows[1]), ("median", rows[m - 1]), ("median", rows[m]),
            ("highest", rows[-2]), ("highest", rows[-1])]
    tb = int(round(F["threshold"] * 255))
    healthy = [s for s in probs if recs[s]["class"] == "healthy"]
    harea = {s: seg.remove_small(probs[s] >= tb, F["min_region_px"]).mean() for s in healthy}
    hs = sorted(healthy, key=lambda s: harea[s])
    lines, meta = [], []
    for tag, r in pick + [("healthy, most FP", {"sample_id": hs[-1]}), ("healthy, least FP", {"sample_id": hs[0]})]:
        sid = r["sample_id"]
        rgb = data.read_rgb(recs[sid]["img"])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gt = data.read_mask(recs[sid]["mask"]) > 0 if recs[sid]["mask"] else np.zeros((480, 640), bool)
        pred = seg.remove_small(probs[sid] >= tb, F["min_region_px"])
        d = seg.image_metrics(pred, gt)
        txt = f"{sid} ({tag})  Dice {d['dice']:.2f}" if gt.any() else f"{sid} ({tag})  pred {100 * pred.mean():.2f}% of image"
        row = np.hstack([label(bgr, txt), label(outline(bgr, gt, (0, 255, 255)), "ground truth"),
                         label(outline(bgr, pred, (255, 255, 0)), "prediction"),
                         label(error_map(bgr, pred, gt), "TP green | FP red | FN blue")])
        cv2.imwrite(str(out / f"{sid}_{tag.split(',')[0].replace(' ', '_')}.png"), row)
        lines.append(row)
        meta.append({"sample_id": sid, "group": tag, **{k: round(float(v), 4) for k, v in d.items() if k in ("dice", "precision", "recall", "gt_pct", "pred_pct")}})
    cv2.imwrite(str(out / "figure_test_qualitative.png"), np.vstack(lines))
    cv2.imwrite(str(out / "figure_test_qualitative_damaged_only.png"), np.vstack(lines[:6]))
    json.dump({"threshold": F["threshold"], "min_region_px": F["min_region_px"], "rows": meta},
              open(out / "figure_rows.json", "w"), indent=2)
    print(json.dumps(meta, indent=1))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
