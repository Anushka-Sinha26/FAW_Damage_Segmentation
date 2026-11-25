"""Step 15 - Damage as % of visible leaf area (Phase 2 damage analysis).

GT %  = GT damage / (leaf OR GT damage);  predicted % = predicted damage / (leaf OR predicted damage)
(deployment-style: each uses only its own damage mask). Leaf mask: faw.leaf (Lab a* Otsu).
Evaluated on pooled OOF predictions and on the saved one-time test predictions (no re-tuning).
Outputs: outputs/leaf_area/ (leaf_qc_sheet.png, per-image CSVs, leaf_area_summary.json)
Usage (project root): python faw_seg\\scripts\\15_leaf_area_damage.py
"""
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

FAW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAW / "src"))
from faw.config import load_config  # noqa: E402
from faw import data, seg, leaf  # noqa: E402

EXPS = {"e00_baseline": ("e00_baseline_exp06recipe_cv", 0.35, 100),
        "e01_aug_healthy": ("e01_mnv2unet_fullresval_aug_healthy", 0.45, 100),
        "e10_rgbt_film": ("e10_rgbt_film_cv", 0.30, 50)}


def agreement(g, p):
    g, p = np.asarray(g), np.asarray(p)
    d = p - g
    return {"n": int(len(g)), "mean_gt_pct": float(g.mean()), "mean_pred_pct": float(p.mean()),
            "bias_pp": float(d.mean()), "mae_pp": float(np.abs(d).mean()), "rmse_pp": float(np.sqrt((d ** 2).mean())),
            "loa95_pp": [float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))],
            "pearson_r": float(np.corrcoef(g, p)[0, 1]),
            "r2": float(1 - (d ** 2).sum() / ((g - g.mean()) ** 2).sum())}


def main():
    cfg = load_config()
    K = cfg["splits"]["n_folds"]
    out = cfg["p"]["outputs"] / "leaf_area"
    out.mkdir(exist_ok=True)
    root = cfg["p"]["outputs"] / "experiments"
    recs = {r["sample_id"]: r for r in data.load_records(cfg, [f"fold{k}" for k in range(K)] + ["test"])}
    dmg_ids = sorted(s for s, r in recs.items() if r["class"] == "damaged")

    leaves, gtpct, rows = {}, {}, []
    for s in dmg_ids:
        rgb = data.read_rgb(recs[s]["img"])
        lm, t = leaf.leaf_mask(rgb)
        gt = data.read_mask(recs[s]["mask"]) > 0
        leaves[s] = lm
        gtpct[s] = leaf.damage_pct_of_leaf(gt, lm)
        rows.append({"sample_id": s, "split": recs[s]["split"], "a_star_otsu": t, "leaf_pct_of_image": 100 * lm.mean(),
                     "gt_damage_pct_of_image": 100 * gt.mean(), "gt_damage_pct_of_leaf": gtpct[s],
                     "gt_damage_inside_leaf_mask": float((gt & lm).sum() / max(gt.sum(), 1))})
    with open(out / "leaf_masks_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    random.seed(42)
    qc_ids = random.sample([s for s in dmg_ids if recs[s]["split"] != "test"], 12)
    tiles = []
    for s in qc_ids:
        bgr = cv2.cvtColor(data.read_rgb(recs[s]["img"]), cv2.COLOR_RGB2BGR)
        o = bgr.copy(); o[~leaves[s]] = (o[~leaves[s]] * 0.2).astype(np.uint8)
        t = np.hstack([cv2.resize(bgr, (320, 240)), cv2.resize(o, (320, 240))])
        cv2.putText(t, s, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        tiles.append(t)
    cv2.imwrite(str(out / "leaf_qc_sheet.png"), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 12, 2)]))

    lp = np.array([r["leaf_pct_of_image"] for r in rows]); gl = np.array([r["gt_damage_pct_of_leaf"] for r in rows])
    summary = {"method": "Lab a* per-image Otsu + 5x5 open/close; denominator = leaf OR damage",
               "damaged_images": len(rows), "leaf_pct_of_image_median_iqr": [float(np.median(lp)), float(np.percentile(lp, 25)), float(np.percentile(lp, 75))],
               "gt_damage_pct_of_leaf_mean_median_max": [float(gl.mean()), float(np.median(gl)), float(gl.max())],
               "gt_damage_pixels_inside_leaf_mask_mean": float(np.mean([r["gt_damage_inside_leaf_mask"] for r in rows])),
               "qc_images": qc_ids, "results": {}}

    for name, (folder, thr, ma) in EXPS.items():
        tb = int(round(thr * 255))
        res = {}
        for part in ["oof", "test"]:
            probs = {}
            if part == "oof":
                for k in range(K):
                    z = np.load(root / folder / f"fold{k}" / "val_probs.npz")
                    probs.update({str(i): p for i, p in zip(z["ids"], z["probs"])})
            else:
                z = np.load(root / folder / "final_eval" / "test_ensemble_probs.npz")
                probs.update({str(i): p for i, p in zip(z["ids"], z["probs"])})
            ids = [s for s in dmg_ids if s in probs]
            pp = [leaf.damage_pct_of_leaf(seg.remove_small(probs[s] >= tb, ma), leaves[s]) for s in ids]
            res[part] = agreement([gtpct[s] for s in ids], pp)
            with open(out / f"{name}_{part}_damage_pct_of_leaf.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["sample_id", "gt_pct_of_leaf", "pred_pct_of_leaf"])
                w.writerows([[s, gtpct[s], v] for s, v in zip(ids, pp)])
        summary["results"][name] = {"threshold": thr, "min_region_px": ma, **res}
    json.dump(summary, open(out / "leaf_area_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
