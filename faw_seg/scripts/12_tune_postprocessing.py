"""Step 12 - Choose threshold + minimum predicted-region size on POOLED OUT-OF-FOLD predictions only.
No test data is read. Output: outputs/experiments/<exp>/final_eval/postproc_grid.csv and printout.
Usage (project root): python faw_seg\\scripts\\12_tune_postprocessing.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

FAW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAW / "src"))
from faw.config import load_config  # noqa: E402
from faw import data, seg  # noqa: E402

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
MIN_AREAS = [0, 10, 25, 50, 100, 200]


def main():
    cfg = load_config()
    T, K = cfg["train"], cfg["splits"]["n_folds"]
    exp = cfg["p"]["outputs"] / "experiments" / T["exp_name"]
    recs = {r["sample_id"]: r for r in data.load_records(cfg, [f"fold{k}" for k in range(K)])}
    dp, dm, hp = [], [], []
    for k in range(K):
        z = np.load(exp / f"fold{k}" / "val_probs.npz")
        for sid, p in zip(z["ids"], z["probs"]):
            r = recs[str(sid)]
            assert r["split"] == f"fold{k}"
            if r["class"] == "damaged":
                dp.append(p)
                dm.append(data.read_mask(r["mask"]) > 0)
            else:
                hp.append(p)
    print(f"OOF images: {len(dp)} damaged, {len(hp)} healthy (test not used)")
    rows = []
    for t in THRESHOLDS:
        tb = int(round(t * 255))
        pb_d = [p >= tb for p in dp]
        pb_h = [p >= tb for p in hp]
        for a in MIN_AREAS:
            dices, gtr, det, prr, cor, tp, fp, fn = [], 0, 0, 0, 0, 0, 0, 0
            for pb, gt in zip(pb_d, dm):
                pr = seg.remove_small(pb, a)
                dices.append(seg.dice_iou(pr, gt)[0])
                rm = seg.region_metrics(pr, gt)
                gtr += rm["gt_regions"]; det += rm["gt_regions_detected"]
                prr += rm["pred_regions"]; cor += rm["pred_regions_correct"]
                c = seg.confusion(pr, gt); tp += c[0]; fp += c[1]; fn += c[2]
            ha = np.array([100 * seg.remove_small(pb, a).mean() for pb in pb_h])
            rows.append({"threshold": t, "min_area_px": a, "mean_dice": round(float(np.mean(dices)), 4),
                         "global_dice": round(2 * tp / (2 * tp + fp + fn), 4),
                         "region_recall": round(det / gtr, 3), "region_precision": round(cor / max(prr, 1), 3),
                         "pred_regions": prr, "healthy_imgs_any_pred": int((ha > 0).sum()),
                         "healthy_imgs_gt_0.1pct": int((ha > 0.1).sum()), "healthy_mean_pct": round(float(ha.mean()), 4)})
            print(rows[-1])
    out = exp / "final_eval"
    out.mkdir(exist_ok=True)
    with open(out / "postproc_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    best = max(rows, key=lambda r: (r["mean_dice"], r["region_precision"]))
    base = max([r for r in rows if r["min_area_px"] == 0], key=lambda r: r["mean_dice"])
    print("\nBest without clean-up :", json.dumps(base))
    print("Best overall          :", json.dumps(best))


if __name__ == "__main__":
    main()
