"""Step 10 - Train one cross-validation fold (RGB, MobileNetV2-U-Net from Exp 06).

Improvements over Exp 06, all decided from validation evidence (never test):
  * checkpoint / early stopping on FULL-RESOLUTION validation Dice (same 6-patch protocol as evaluation)
  * random-scale crops from full images + flips + colour/brightness jitter (shadows, illumination)
  * healthy images as damage-free examples (isolated false alarms were 56% of FP pixels)
  * 5-fold capture-block-grouped CV; the 'test' split is never touched here

Usage (from faw_seg/):
  python scripts/10_train_cv.py --fold 0 --smoke     # 2-minute pipeline check
  python scripts/10_train_cv.py --fold 0             # real training, then folds 1..4
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
from faw.config import load_config, set_seed  # noqa: E402
from faw import data, seg, fusion  # noqa: E402

import tensorflow as tf  # noqa: E402


def import_phase1_model(cfg):
    sys.path.insert(0, str(cfg["root"] / cfg["paths"]["phase1_model_code"]))
    import model_experiment_06_mobilenetv2_unet as m  # noqa: E402
    return m


class FullResMonitor(tf.keras.callbacks.Callback):
    """Full-resolution validation Dice each epoch; saves best weights, reduces LR, early-stops."""

    def __init__(self, imgs, masks, weights_path, T, patience, state, stage, thermals=None):
        super().__init__()
        self.imgs, self.masks, self.path, self.T = imgs, masks, weights_path, T
        self.thermals = thermals
        self.patience, self.state, self.stage = patience, state, stage
        self.wait = self.lr_wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        probs = seg.predict_full(self.model, self.imgs, self.T["patch"], self.T["stride"], thermals=self.thermals)
        d = float(np.mean([seg.dice_iou(p >= self.T["val_threshold"], m > 0)[0] for p, m in zip(probs, self.masks)]))
        lr = float(tf.convert_to_tensor(self.model.optimizer.learning_rate))
        improved = d > self.state["best"] + 1e-4
        if improved:
            self.state.update(best=d, stage=self.stage, epoch=epoch + 1)
            self.model.save_weights(str(self.path))
            self.wait = self.lr_wait = 0
        else:
            self.wait += 1
            self.lr_wait += 1
            if self.lr_wait >= self.T["lr_patience"]:
                new = max(lr * self.T["lr_factor"], self.T["min_lr"])
                self.model.optimizer.learning_rate.assign(new)
                self.lr_wait = 0
            if self.wait >= self.patience:
                self.model.stop_training = True
        row = {"stage": self.stage, "epoch": epoch + 1, "lr": lr,
               "train_loss": float(logs.get("loss", np.nan)),
               "train_dice": float(logs.get("dice_metric", np.nan)),
               "val_fullres_dice": d, "best_so_far": self.state["best"], "saved": improved}
        self.state["history"].append(row)
        print(f"  [{self.stage} ep {epoch + 1}] train_dice={row['train_dice']:.4f} "
              f"val_fullres_dice={d:.4f} best={self.state['best']:.4f}{' *saved*' if improved else ''} lr={lr:.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    ap.add_argument("--config", default=None, help="config file name in faw_seg/configs (default config.yaml)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Config: {args.config or 'config.yaml'} | experiment: {cfg['train']['exp_name']}")
    T = cfg["train"]
    K = cfg["splits"]["n_folds"]
    assert 0 <= args.fold < K
    seed = cfg["seed"] + args.fold
    set_seed(seed)
    exp = T["exp_name"] + ("_SMOKE" if args.smoke else "")
    out = cfg["p"]["outputs"] / "experiments" / exp / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)

    val_split = f"fold{args.fold}"
    train_splits = [f"fold{k}" for k in range(K) if k != args.fold]
    tr = data.load_records(cfg, train_splits)
    va = data.load_records(cfg, [val_split])
    assert not any(r["split"] == "test" for r in tr + va), "test split must never be used in training"

    tr_d = [r for r in tr if r["class"] == "damaged"]
    tr_h = [r for r in tr if r["class"] == "healthy"]
    va_d = [r for r in va if r["class"] == "damaged"]
    va_h = [r for r in va if r["class"] == "healthy"]
    if args.smoke:
        tr_d, tr_h, va_d, va_h = tr_d[:12], tr_h[:6], va_d[:4], va_h[:2]
    print(f"Fold {args.fold}: train {len(tr_d)} damaged + {len(tr_h)} healthy | "
          f"val {len(va_d)} damaged + {len(va_h)} healthy | masks: {cfg['p']['masks']}")

    t0 = time.time()
    d_imgs, d_masks = data.load_arrays(tr_d)
    h_imgs, _ = data.load_arrays(tr_h) if T["healthy_fraction"] > 0 else ([], None)
    v_imgs, v_masks = data.load_arrays(va_d)
    vh_imgs, _ = data.load_arrays(va_h)
    is_fusion = bool((cfg.get("model") or {}).get("fusion"))
    d_th = h_th = v_th = vh_th = None
    if is_fusion:
        d_th = data.load_thermal(cfg, tr_d)
        h_th = data.load_thermal(cfg, tr_h) if h_imgs else None
        v_th = data.load_thermal(cfg, va_d)
        vh_th = data.load_thermal(cfg, va_h)
        print(f"RGB-thermal fusion: {cfg['model']['fusion']} (thermal maps: outputs/thermal_aligned)")
    print(f"Loaded images in {time.time() - t0:.0f}s")

    ds, steps = data.make_train_dataset(tf, d_imgs, d_masks, h_imgs, cfg, seed, d_th, h_th)
    if args.smoke:
        steps = 5

    M = import_phase1_model(cfg)
    model = fusion.build_model(cfg, M, encoder_trainable=False)
    print(f"Model: {model.name} | parameters: {model.count_params():,}")
    M.compile_model(model, learning_rate=T["stage1"]["lr"])
    wpath = out / "best.weights.h5"
    state = {"best": -1.0, "stage": None, "epoch": None, "history": []}

    e1 = 1 if args.smoke else T["stage1"]["epochs"]
    e2 = min(1, T["stage2"]["epochs"]) if args.smoke else T["stage2"]["epochs"]
    print(f"\nStage 1 (encoder frozen): up to {e1} epochs, {steps} steps/epoch")
    model.fit(ds, steps_per_epoch=steps, epochs=e1, verbose=1,
              callbacks=[FullResMonitor(v_imgs, v_masks, wpath, T, T["stage1"]["patience"], state, "stage1", v_th)])

    if e2 > 0:
        model.load_weights(str(wpath))
        M.set_encoder_fine_tuning(model, trainable_from_layer=T["stage2"]["fine_tune_from"])
        M.compile_model(model, learning_rate=T["stage2"]["lr"])
        print(f"\nStage 2 (fine-tune from layer {T['stage2']['fine_tune_from']}): up to {e2} epochs")
        model.fit(ds, steps_per_epoch=steps, epochs=e2, verbose=1,
                  callbacks=[FullResMonitor(v_imgs, v_masks, wpath, T, T["stage2"]["patience"], state, "stage2", v_th)])
    else:
        print("\nStage 2 skipped (stage2.epochs = 0 in config)")

    model.load_weights(str(wpath))
    print("\nBest checkpoint reloaded. Validation-fold analysis ...")
    pv = seg.predict_full(model, v_imgs, T["patch"], T["stride"], thermals=v_th)
    ph = seg.predict_full(model, vh_imgs, T["patch"], T["stride"], thermals=vh_th) if vh_imgs else []
    lo, hi, st = T["threshold_grid"]
    grid = np.round(np.arange(lo, hi + 1e-9, st), 3)
    sweep = []
    for t in grid:
        d = [seg.dice_iou(p >= t, m > 0) for p, m in zip(pv, v_masks)]
        sweep.append({"threshold": float(t), "mean_dice": float(np.mean([x[0] for x in d])),
                      "mean_iou": float(np.mean([x[1] for x in d]))})
    best_t = max(sweep, key=lambda r: (r["mean_dice"], r["mean_iou"]))
    t = best_t["threshold"]
    hfp = [100 * (p >= t).mean() for p in ph]

    np.savez_compressed(out / "val_probs.npz",
                        ids=np.array([r["sample_id"] for r in va_d + va_h]),
                        probs=np.stack([(p * 255).round().astype(np.uint8) for p in pv + ph]))
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(state["history"][0]))
        w.writeheader()
        w.writerows(state["history"])
    summary = {
        "fold": args.fold, "experiment": exp, "seed": seed, "smoke": args.smoke,
        "masks": str(cfg["p"]["masks"]), "train_damaged": len(tr_d), "train_healthy": len(tr_h),
        "val_damaged": len(va_d), "val_healthy": len(va_h), "test_used": False,
        "best_stage": state["stage"], "best_epoch": state["epoch"],
        "best_val_fullres_dice_at_0.5": state["best"],
        "fold_threshold": best_t, "threshold_sweep": sweep,
        "healthy_val_at_fold_threshold": {
            "images": len(hfp), "mean_pred_pct": float(np.mean(hfp)) if hfp else None,
            f"images_with_pred_area_gt_{T['healthy_fp_area_pct']}pct":
                int(sum(x > T["healthy_fp_area_pct"] for x in hfp))},
        "minutes": round((time.time() - t0) / 60, 1),
        "config_train": T,
    }
    with open(out / "fold_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({k: summary[k] for k in ["fold", "best_stage", "best_epoch", "best_val_fullres_dice_at_0.5",
                                              "fold_threshold", "healthy_val_at_fold_threshold", "minutes"]}, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
