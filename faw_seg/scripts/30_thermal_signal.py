"""Step 30 (Phase 3) - Is feeding damage visible in thermal? (CV images only; test untouched)

Per damaged image, thermal is converted to apparent temperature and warped onto the RGB grid
(faw.thermal, camera PiP calibration). Inside the thermal footprint we compare:
  D_leaf : GT damage pixels that look like leaf (scraped / window-pane tissue)
  D_hole : GT damage pixels that do not look like leaf (holes / missing tissue -> background seen)
  N      : healthy leaf in a ring 3-15 px around GT damage (local reference)
  S      : soil / non-leaf background away from damage
Statistics: per-image mean temperature differences (paired sign-flip permutation test),
per-image AUC of temperature for damage vs nearby healthy leaf.
Outputs: outputs/thermal_signal/thermal_signal_per_image.csv, thermal_signal_summary.json
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
from faw import data, leaf, thermal  # noqa: E402


def auc(pos, neg):
    if len(pos) < 20 or len(neg) < 20:
        return None
    x = np.concatenate([pos, neg]); r = x.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def perm_test(d, n=20000, seed=0):
    d = np.asarray([x for x in d if x is not None and np.isfinite(x)])
    rng = np.random.default_rng(seed)
    p = (np.sum(np.abs((rng.choice([-1, 1], (n, len(d))) * d).mean(1)) >= abs(d.mean())) + 1) / (n + 1)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    return {"n": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "share_positive": float((d > 0).mean()), "p_value": float(p)}


def main():
    cfg = load_config()
    K = cfg["splits"]["n_folds"]
    out = cfg["p"]["outputs"] / "thermal_signal"
    out.mkdir(exist_ok=True)
    recs = [r for r in data.load_records(cfg, [f"fold{k}" for k in range(K)]) if r["class"] == "damaged"]
    ring_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    ring_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    rows = []
    for r in recs:
        rgb = data.read_rgb(r["img"])
        fp = Path(str(r["img"]).replace("\\rgb\\", "\\flir\\").replace("/rgb/", "/flir/"))
        t, p = thermal.apparent_temperature(fp)
        if t is None:
            continue
        tr, valid = thermal.to_rgb_grid(t, p.get("real2ir", 1.3123), p.get("offx", -9), p.get("offy", 5))
        lm, _ = leaf.leaf_mask(rgb)
        gt = data.read_mask(r["mask"]) > 0
        g8 = gt.astype(np.uint8)
        ring = (cv2.dilate(g8, ring_out) > 0) & ~(cv2.dilate(g8, ring_in) > 0)
        D_leaf, D_hole = gt & lm & valid, gt & ~lm & valid
        N = ring & lm & valid
        S = ~lm & valid & ~(cv2.dilate(g8, ring_out) > 0)
        L = lm & valid & ~gt
        m = lambda a: float(tr[a].mean()) if a.sum() >= 20 else None  # noqa: E731
        row = {"sample_id": r["sample_id"], "split": r["split"],
               "gt_px": int(gt.sum()), "gt_in_thermal_footprint": float((gt & valid).sum() / max(gt.sum(), 1)),
               "T_leaf_all": m(L), "T_soil": m(S), "T_damage_leaf": m(D_leaf), "T_damage_hole": m(D_hole), "T_near_healthy_leaf": m(N),
               "T_damage_all": m(gt & valid),
               "auc_damage_vs_near_leaf": auc(tr[gt & valid], tr[N]),
               "auc_damageleaf_vs_near_leaf": auc(tr[D_leaf], tr[N])}
        for a, b, k in [("T_damage_all", "T_near_healthy_leaf", "dT_damage_minus_near"),
                        ("T_damage_leaf", "T_near_healthy_leaf", "dT_damageleaf_minus_near"),
                        ("T_damage_hole", "T_near_healthy_leaf", "dT_hole_minus_near"),
                        ("T_soil", "T_leaf_all", "dT_soil_minus_leaf")]:
            row[k] = row[a] - row[b] if row[a] is not None and row[b] is not None else None
        rows.append(row)
    with open(out / "thermal_signal_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    col = lambda k: [x[k] for x in rows if x[k] is not None]  # noqa: E731
    cov = np.array([x["gt_in_thermal_footprint"] for x in rows])
    summ = {"images": len(rows), "test_used": False,
            "share_of_gt_damage_inside_thermal_footprint": {"mean": float(cov.mean()), "median": float(np.median(cov)),
                                                            "images_with_no_damage_in_footprint": int((cov == 0).sum())},
            "dT_soil_minus_leaf": perm_test(col("dT_soil_minus_leaf")),
            "dT_damage_all_minus_near_healthy_leaf": perm_test(col("dT_damage_minus_near")),
            "dT_damage_holes_minus_near_healthy_leaf": perm_test(col("dT_hole_minus_near")),
            "dT_damaged_leaf_tissue_minus_near_healthy_leaf": perm_test(col("dT_damageleaf_minus_near")),
            "auc_damage_vs_near_leaf": {"n": len(col("auc_damage_vs_near_leaf")), "mean": float(np.mean(col("auc_damage_vs_near_leaf"))),
                                        "median": float(np.median(col("auc_damage_vs_near_leaf")))},
            "auc_damaged_leaf_tissue_vs_near_leaf": {"n": len(col("auc_damageleaf_vs_near_leaf")), "mean": float(np.mean(col("auc_damageleaf_vs_near_leaf"))),
                                                     "median": float(np.median(col("auc_damageleaf_vs_near_leaf")))}}
    json.dump(summ, open(out / "thermal_signal_summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
