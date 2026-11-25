"""Step 14 - Paired comparison of two experiments on the SAME pooled out-of-fold images (no test data).
Each experiment uses its own settings chosen by the identical procedure (scripts/12, max OOF mean Dice).
Paired tests: per-image Dice difference (bootstrap 95% CI + sign-flip permutation test, 20k draws);
healthy images: per-image false-alarm area (permutation test) and counts.
Usage (project root):
  python faw_seg\\scripts\\14_compare_experiments.py
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

EXPS = {  # name: (folder, threshold, min_region_px) - settings from each experiment's postproc_grid.csv best row
    "e00_baseline": ("e00_baseline_exp06recipe_cv", 0.35, 100),
    "e01_aug_healthy": ("e01_mnv2unet_fullresval_aug_healthy", 0.45, 100),
}


def load_oof(exp_dir, K, recs):
    d, h = {}, {}
    for k in range(K):
        z = np.load(exp_dir / f"fold{k}" / "val_probs.npz")
        for sid, p in zip(z["ids"], z["probs"]):
            sid = str(sid)
            assert recs[sid]["split"] == f"fold{k}"
            (d if recs[sid]["class"] == "damaged" else h)[sid] = p
    return d, h


def paired(a, b, n=20000, seed=0):
    diff = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(5000)]
    obs = abs(diff.mean())
    flips = rng.choice([-1, 1], size=(n, len(diff)))
    perm = np.abs((flips * diff).mean(1))
    return {"mean_diff_b_minus_a": float(diff.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_value_two_sided": float((np.sum(perm >= obs) + 1) / (n + 1)),
            "images_b_better": int((diff > 0).sum()), "images_a_better": int((diff < 0).sum()), "ties": int((diff == 0).sum())}


def parse_exp(s):
    name, rest = s.split("=")
    folder, thr, ma = rest.split(",")
    return name, (folder, float(thr), int(ma))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=None, help="name=folder,threshold,min_region_px (reference)")
    ap.add_argument("--b", default=None, help="name=folder,threshold,min_region_px (comparison)")
    args = ap.parse_args()
    global EXPS
    if args.a and args.b:
        EXPS = dict([parse_exp(args.a), parse_exp(args.b)])
    cfg = load_config()
    K = cfg["splits"]["n_folds"]
    root = cfg["p"]["outputs"] / "experiments"
    recs = {r["sample_id"]: r for r in data.load_records(cfg, [f"fold{k}" for k in range(K)])}
    masks = {s: data.read_mask(r["mask"]) > 0 for s, r in recs.items() if r["class"] == "damaged"}
    per, summ = {}, {}
    for name, (folder, t, a) in EXPS.items():
        d, h = load_oof(root / folder, K, recs)
        tb = int(round(t * 255))
        dice, bt, reg = {}, {}, np.zeros(4, int)
        for s, p in d.items():
            pr = seg.remove_small(p >= tb, a)
            dice[s] = seg.dice_iou(pr, masks[s])[0]
            bt[s] = seg.boundary_tolerant_dice(pr, masks[s], 3)
            r = seg.region_metrics(pr, masks[s])
            reg += [r["gt_regions"], r["gt_regions_detected"], r["pred_regions"], r["pred_regions_correct"]]
        ha = {s: 100 * seg.remove_small(p >= tb, a).mean() for s, p in h.items()}
        per[name] = {"dice": dice, "bt": bt, "healthy": ha}
        hv = np.array(list(ha.values()))
        summ[name] = {"threshold": t, "min_region_px": a, "damaged_images": len(dice), "healthy_images": len(ha),
                      "mean_dice": float(np.mean(list(dice.values()))), "mean_bt_dice": float(np.mean(list(bt.values()))),
                      "region_recall": reg[1] / reg[0], "region_precision": reg[3] / max(reg[2], 1),
                      "healthy_any_pred": int((hv > 0).sum()), "healthy_gt_0.1pct": int((hv > 0.1).sum()),
                      "healthy_gt_1pct": int((hv > 1).sum()), "healthy_mean_pct": float(hv.mean())}
    A, B = list(EXPS)
    ids = sorted(per[A]["dice"])
    hids = sorted(per[A]["healthy"])
    assert ids == sorted(per[B]["dice"]) and hids == sorted(per[B]["healthy"])
    comp = {
        "dice": paired([per[A]["dice"][s] for s in ids], [per[B]["dice"][s] for s in ids]),
        "boundary_tolerant_dice": paired([per[A]["bt"][s] for s in ids], [per[B]["bt"][s] for s in ids]),
        "healthy_false_alarm_pct": paired([per[A]["healthy"][s] for s in hids], [per[B]["healthy"][s] for s in hids]),
    }
    out = {"a": A, "b": B, "summary": summ, "paired_b_minus_a": comp,
           "note": "OOF only; each experiment at its own OOF-selected threshold/min-region (same procedure). Test data not used."}
    od = root / f"comparison_{A}_vs_{B}"
    od.mkdir(exist_ok=True)
    json.dump(out, open(od / "oof_comparison.json", "w"), indent=2)
    with open(od / "oof_per_image_dice.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", f"dice_{A}", f"dice_{B}", f"bt_{A}", f"bt_{B}"])
        for s in ids:
            w.writerow([s, per[A]["dice"][s], per[B]["dice"][s], per[A]["bt"][s], per[B]["bt"][s]])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
