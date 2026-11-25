"""Step 11 - Final evaluation of the 5-fold experiment.

Part A (no test data): pooled out-of-fold (OOF) results. Each non-test damaged image was
  predicted once, by the model that did NOT train on it. The single decision threshold is
  chosen here (max mean per-image Dice over all OOF images) - test data plays no part.
Part B (ONE time): test set (53 damaged + 46 healthy). The 5 fold models are averaged
  (ensemble) and evaluated at the OOF threshold. A lock file prevents accidental re-runs.

Usage (project root):
  python faw_seg\\scripts\\11_evaluate_cv.py --oof-only     # Part A only (safe, any time)
  python faw_seg\\scripts\\11_evaluate_cv.py                # Part A + the one-time test
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

FAW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAW / "src"))
from faw.config import load_config  # noqa: E402
from faw import data, seg  # noqa: E402

BT_TOL = 3


def damage_stats(g, p):
    g, p = np.asarray(g, float), np.asarray(p, float)
    ok = len(g) > 2 and g.std() > 0 and p.std() > 0
    return {"mean_gt_pct": float(g.mean()), "mean_pred_pct": float(p.mean()), "bias_pp": float((p - g).mean()),
            "mae_pp": float(np.abs(p - g).mean()), "rmse_pp": float(np.sqrt(((p - g) ** 2).mean())),
            "pearson_r": float(np.corrcoef(g, p)[0, 1]) if ok else None,
            "r2": float(1 - ((p - g) ** 2).sum() / ((g - g.mean()) ** 2).sum()) if g.std() > 0 else None}


def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def evaluate(probs, masks, ids, thr, min_area=0):
    rows, tp, fp, fn = [], 0, 0, 0
    for sid, pr, m in zip(ids, probs, masks):
        pred, gt = seg.remove_small(pr >= thr, min_area), m > 0
        r = {"sample_id": sid, **seg.image_metrics(pred, gt), "bt_dice": seg.boundary_tolerant_dice(pred, gt, BT_TOL),
             **seg.region_metrics(pred, gt)}
        rows.append(r)
        tp, fp, fn = tp + r["tp"], fp + r["fp"], fn + r["fn"]
    s = lambda k: int(sum(r[k] for r in rows))  # noqa: E731
    dice = [r["dice"] for r in rows]
    summ = {
        "images": len(rows), "threshold": thr, "min_region_px": min_area,
        "mean_per_image": {k: float(np.mean([r[k] for r in rows])) for k in ["dice", "iou", "precision", "recall", "bt_dice"]},
        "mean_dice_95ci": bootstrap_ci(dice),
        "median_dice": float(np.median(dice)), "dice_iqr": [float(np.percentile(dice, 25)), float(np.percentile(dice, 75))],
        "global_pixel": {"dice": 2 * tp / (2 * tp + fp + fn), "iou": tp / (tp + fp + fn),
                         "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn)},
        "regions": {"gt_regions": s("gt_regions"), "gt_detected": s("gt_regions_detected"),
                    "region_recall": s("gt_regions_detected") / max(1, s("gt_regions")),
                    "pred_regions": s("pred_regions"), "pred_correct": s("pred_regions_correct"),
                    "region_precision": s("pred_regions_correct") / max(1, s("pred_regions"))},
        "damage_percent_of_image": damage_stats([r["gt_pct"] for r in rows], [r["pred_pct"] for r in rows]),
    }
    return rows, summ


def healthy_stats(probs, thr, area, min_area=0):
    a = np.array([100 * seg.remove_small(p >= thr, min_area).mean() for p in probs]) if len(probs) else np.array([])
    if not len(a):
        return {"images": 0}
    return {"images": int(len(a)), "mean_pred_pct": float(a.mean()), "median_pred_pct": float(np.median(a)),
            "p90_pred_pct": float(np.percentile(a, 90)), "max_pred_pct": float(a.max()),
            f"images_gt_{area}pct": int((a > area).sum()), "images_gt_1pct": int((a > 1).sum()),
            "images_with_any_prediction": int((a > 0).sum())}


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-only", action="store_true")
    ap.add_argument("--allow-missing-folds", action="store_true", help="debug only")
    ap.add_argument("--force", action="store_true", help="re-run test despite lock (must be justified)")
    ap.add_argument("--config", default=None, help="config file in faw_seg/configs (default config.yaml)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Config: {args.config or 'config.yaml'} | experiment: {cfg['train']['exp_name']}")
    T, K = cfg["train"], cfg["splits"]["n_folds"]
    exp_dir = cfg["p"]["outputs"] / "experiments" / T["exp_name"]
    out = exp_dir / "final_eval"
    out.mkdir(exist_ok=True)
    area = T["healthy_fp_area_pct"]

    recs = {r["sample_id"]: r for r in data.load_records(cfg, [f"fold{k}" for k in range(K)] + ["test"])}
    folds = [k for k in range(K) if (exp_dir / f"fold{k}" / "fold_summary.json").exists()]
    if len(folds) != K and not args.allow_missing_folds:
        raise SystemExit(f"Only folds {folds} are finished. Train all {K} folds first.")
    print(f"Folds used: {folds}")

    # ---------------- Part A: pooled out-of-fold ----------------
    oof_d_ids, oof_d_p, oof_d_m, oof_h_p, per_fold = [], [], [], [], []
    for k in folds:
        z = np.load(exp_dir / f"fold{k}" / "val_probs.npz")
        fs = json.load(open(exp_dir / f"fold{k}" / "fold_summary.json"))
        for sid, p in zip(z["ids"], z["probs"]):
            sid = str(sid)
            assert recs[sid]["split"] == f"fold{k}", f"{sid} not in fold{k}"
            if recs[sid]["class"] == "damaged":
                oof_d_ids.append(sid)
                oof_d_p.append(p.astype(np.float32) / 255.0)
                oof_d_m.append(data.read_mask(recs[sid]["mask"]))
            else:
                oof_h_p.append(p.astype(np.float32) / 255.0)
        per_fold.append({"fold": k, "best_epoch": fs["best_epoch"], "val_dice_at_0.5": fs["best_val_fullres_dice_at_0.5"],
                         "fold_threshold": fs["fold_threshold"]["threshold"], "val_dice_at_fold_thr": fs["fold_threshold"]["mean_dice"]})
    lo, hi, st = T["threshold_grid"]
    sweep = []
    for t in np.round(np.arange(lo, hi + 1e-9, st), 3):
        d = [seg.dice_iou(p >= t, m > 0) for p, m in zip(oof_d_p, oof_d_m)]
        sweep.append({"threshold": float(t), "mean_dice": float(np.mean([x[0] for x in d])), "mean_iou": float(np.mean([x[1] for x in d]))})
    F = cfg.get("final") or {}
    thr = F.get("threshold", max(sweep, key=lambda r: (r["mean_dice"], r["mean_iou"]))["threshold"])
    min_area = F.get("min_region_px", 0)
    print(f"Decision settings (from OOF only): threshold {thr}, min region {min_area} px")
    oof_rows, oof_summ = evaluate(oof_d_p, oof_d_m, oof_d_ids, thr, min_area)
    _, oof_raw = evaluate(oof_d_p, oof_d_m, oof_d_ids, thr, 0)
    fd = [f["val_dice_at_fold_thr"] for f in per_fold]
    part_a = {"folds": folds, "per_fold": per_fold,
              "fold_dice_mean_sd": [float(np.mean(fd)), float(np.std(fd, ddof=1)) if len(fd) > 1 else 0.0],
              "oof_threshold": thr, "min_region_px": min_area, "threshold_sweep (no clean-up)": sweep, "oof_damaged": oof_summ,
              "oof_damaged_without_cleanup": oof_raw,
              "oof_healthy": healthy_stats(oof_h_p, thr, area, min_area),
              "oof_healthy_without_cleanup": healthy_stats(oof_h_p, thr, area, 0)}
    write_csv(oof_rows, out / "oof_per_image.csv")
    with open(out / "oof_summary.json", "w") as f:
        json.dump(part_a, f, indent=2)
    print("\n=== PART A: pooled out-of-fold (no test data) ===")
    print(json.dumps({k: part_a[k] for k in ["fold_dice_mean_sd", "oof_threshold", "min_region_px", "oof_damaged", "oof_healthy"]}, indent=2))
    if args.oof_only:
        print(f"\nSaved to {out}  (test NOT evaluated)")
        return

    # ---------------- Part B: one-time test ----------------
    lock = out / "TEST_EVALUATED.lock"
    if lock.exists() and not args.force:
        raise SystemExit(f"Test already evaluated ({lock.read_text().strip()}). Results are in {out}. Not re-running.")
    if len(folds) != K:
        raise SystemExit("All folds required for the test evaluation.")
    import tensorflow as tf
    sys.path.insert(0, str(cfg["root"] / cfg["paths"]["phase1_model_code"]))
    import model_experiment_06_mobilenetv2_unet as M
    from faw import fusion  # needs TensorFlow; imported only for the test step

    te = [r for r in recs.values() if r["split"] == "test"]
    te_d = sorted([r for r in te if r["class"] == "damaged"], key=lambda r: r["sample_id"])
    te_h = sorted([r for r in te if r["class"] == "healthy"], key=lambda r: r["sample_id"])
    d_imgs, d_masks = data.load_arrays(te_d)
    h_imgs, _ = data.load_arrays(te_h)
    is_fusion = bool((cfg.get("model") or {}).get("fusion"))
    d_th = data.load_thermal(cfg, te_d) if is_fusion else None
    h_th = data.load_thermal(cfg, te_h) if is_fusion else None
    print(f"\n=== PART B: test set {len(te_d)} damaged + {len(te_h)} healthy, threshold {thr} (from OOF) ===")
    t0 = time.time()
    sum_d = [np.zeros((480, 640), np.float32) for _ in te_d]
    sum_h = [np.zeros((480, 640), np.float32) for _ in te_h]
    single = []
    for k in folds:
        tf.keras.backend.clear_session()
        model = fusion.build_model(cfg, M, encoder_trainable=False)
        model.load_weights(str(exp_dir / f"fold{k}" / "best.weights.h5"))
        pd_ = seg.predict_full(model, d_imgs, T["patch"], T["stride"], thermals=d_th)
        ph_ = seg.predict_full(model, h_imgs, T["patch"], T["stride"], thermals=h_th)
        for i, p in enumerate(pd_):
            sum_d[i] += p
        for i, p in enumerate(ph_):
            sum_h[i] += p
        _, s1 = evaluate(pd_, d_masks, [r["sample_id"] for r in te_d], thr, min_area)
        single.append({"fold_model": k, "mean_dice": s1["mean_per_image"]["dice"], "global_dice": s1["global_pixel"]["dice"]})
        print(f"  fold {k} model alone: mean Dice {single[-1]['mean_dice']:.4f}")
    ens_d = [s / len(folds) for s in sum_d]
    ens_h = [s / len(folds) for s in sum_h]
    ids_d = [r["sample_id"] for r in te_d]
    rows, summ = evaluate(ens_d, d_masks, ids_d, thr, min_area)
    _, summ_raw = evaluate(ens_d, d_masks, ids_d, thr, 0)
    sens = {str(t): evaluate(ens_d, d_masks, ids_d, t, min_area)[1]["mean_per_image"]["dice"]
            for t in [0.3, 0.4, 0.5, 0.6]}
    result = {"primary": "5-fold ensemble (mean probability) at OOF threshold + OOF min region size", "threshold": thr,
              "min_region_px": min_area, "test_damaged": summ, "test_healthy": healthy_stats(ens_h, thr, area, min_area),
              "test_damaged_without_cleanup": summ_raw, "test_healthy_without_cleanup": healthy_stats(ens_h, thr, area, 0),
              "single_fold_models": single,
              "single_fold_mean_dice_mean_sd": [float(np.mean([s["mean_dice"] for s in single])),
                                                float(np.std([s["mean_dice"] for s in single], ddof=1))],
              "sensitivity_mean_dice_by_threshold (not used for selection)": sens,
              "minutes": round((time.time() - t0) / 60, 1)}
    write_csv(rows, out / "test_per_image.csv")
    with open(out / "test_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(out / "test_ensemble_probs.npz",
                        ids=np.array([r["sample_id"] for r in te_d + te_h]),
                        probs=np.stack([(p * 255).round().astype(np.uint8) for p in ens_d + ens_h]))
    lock.write_text(time.strftime("evaluated %Y-%m-%d %H:%M"))
    print(json.dumps(result, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
