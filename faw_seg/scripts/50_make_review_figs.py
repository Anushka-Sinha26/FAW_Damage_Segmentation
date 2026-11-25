"""Review figures from saved outputs (no re-training, no new test runs).
Usage: python make_review_figs.py <faw_seg/outputs dir> <out_dir>
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

O, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
B = os.path.join(O, "paper_figures", "results_bundle")
E00 = os.path.join(O, "experiments", "e00_baseline_exp06recipe_cv")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
                     "axes.linewidth": 0.7, "axes.edgecolor": "#52606d", "xtick.color": "#3e4c59",
                     "ytick.color": "#3e4c59", "axes.labelcolor": "#1f2933", "axes.titleweight": "bold",
                     "axes.titlesize": 10, "axes.titlelocation": "left"})
INK, MUTED, GRID = "#1f2933", "#52606d", "#e4e7eb"
C_RGB, C_E01, C_TH, C_OOF = "#1f6fb2", "#7b8794", "#c05621", "#b8c2cc"
MODELS = [("e00", "e00 RGB (main)", C_RGB), ("e01", "e01 RGB, extended aug.", C_E01), ("e10", "e10 RGB + thermal", C_TH)]


def save(fig, name):
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def clean(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6); ax.set_axisbelow(True)


def stats(gt, pr):
    d = pr - gt
    return dict(bias=d.mean(), mae=np.abs(d).mean(), r=np.corrcoef(gt, pr)[0, 1], n=len(gt))


# ---------------------------------------------------------------- 1. actual vs predicted
ti = pd.read_csv(os.path.join(B, "e00_test_per_image.csv"))
oi = pd.read_csv(os.path.join(B, "e00_oof_per_image.csv"))
tl = pd.read_csv(os.path.join(B, "e00_baseline_test_damage_pct_of_leaf.csv"))
ol = pd.read_csv(os.path.join(B, "e00_baseline_oof_damage_pct_of_leaf.csv"))

fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
panels = [(axs[0], oi.gt_pct, oi.pred_pct, ti.gt_pct, ti.pred_pct, ti.sample_id, "Damage as % of image", 10.5),
          (axs[1], ol.gt_pct_of_leaf, ol.pred_pct_of_leaf, tl.gt_pct_of_leaf, tl.pred_pct_of_leaf, tl.sample_id, "Damage as % of visible leaf", 30)]
for k, (ax, og, op, tg, tp_, tid, title, lim) in enumerate(panels):
    ax.plot([0, lim], [0, lim], ls="--", lw=1, color=MUTED, zorder=1)
    ax.scatter(og, op, s=14, color=C_OOF, alpha=0.8, lw=0, label=f"Cross-validation (OOF, n={len(og)})", zorder=2)
    ax.scatter(tg, tp_, s=26, color=C_RGB, edgecolor="white", lw=0.6, label=f"Held-out test (n={len(tg)})", zorder=3)
    so, st = stats(np.asarray(og), np.asarray(op)), stats(np.asarray(tg), np.asarray(tp_))
    ax.text(0.03, 0.97, f"Test: r = {st['r']:.2f}, MAE = {st['mae']:.2f} pp, bias = {st['bias']:+.2f} pp\n"
                        f"OOF:  r = {so['r']:.2f}, MAE = {so['mae']:.2f} pp, bias = {so['bias']:+.2f} pp",
            transform=ax.transAxes, va="top", fontsize=7.8, color=INK, family="DejaVu Sans Mono")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("Actual (reference annotation), %"); ax.set_ylabel("Predicted (model), %")
    ax.set_title(f"({'ab'[k]}) {title}"); clean(ax, "both")
    if k == 1:
        i = int(np.argmax(np.asarray(tg) - np.asarray(tp_)))
        ax.annotate(f"{tid.iloc[i]}: finely holed area\nenclosed by one polygon", (tg.iloc[i], tp_.iloc[i]),
                    xytext=(tg.iloc[i] - 12, tp_.iloc[i] + 7), fontsize=8, color=MUTED,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
axs[0].legend(loc="lower right", frameon=False, fontsize=8.5)
fig.suptitle("Actual vs predicted damage percentage, main RGB model (e00); dashed line = perfect agreement",
             x=0.01, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(); save(fig, "fig_rev_actual_vs_predicted")

# ---------------------------------------------------------------- 2. training curves
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for f in range(5):
    h = pd.read_csv(os.path.join(E00, f"fold{f}", "history.csv"))
    lab = (f == 0)
    axs[0].plot(h.epoch, h.train_loss, color=C_RGB, lw=1.1, alpha=0.85, label="Training loss" if lab else None)
    axs[1].plot(h.epoch, h.train_dice, color=C_RGB, lw=1.1, alpha=0.85, label="Training Dice (patches)" if lab else None)
    axs[1].plot(h.epoch, h.val_fullres_dice, color=C_TH, lw=1.1, alpha=0.85, label="Validation Dice (full image)" if lab else None)
    b = h.val_fullres_dice.idxmax()
    axs[1].scatter(h.epoch[b], h.val_fullres_dice[b], s=30, color=C_TH, edgecolor="white", lw=0.8, zorder=4,
                   label="Saved checkpoint (best validation)" if lab else None)
axs[0].set_title("(a) Training loss, 5 folds"); axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss (0.5 BCE + 0.5 Dice)")
axs[1].set_title("(b) Training vs validation Dice, 5 folds"); axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Dice")
axs[1].set_ylim(0.15, 0.7)
for a in axs: clean(a); a.legend(frameon=False, fontsize=8.5, loc="upper right" if a is axs[0] else "lower right")
fig.tight_layout(); save(fig, "fig_rev_training_curves")

# ---------------------------------------------------------------- 3. Dice distribution
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(7, 4.2))
data = [pd.read_csv(os.path.join(B, f"{m}_test_per_image.csv")).dice.values for m, _, _ in MODELS]
bp = ax.boxplot(data, widths=0.45, showfliers=False, patch_artist=True, medianprops=dict(color=INK, lw=1.4),
                whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED), boxprops=dict(lw=0.8, color=MUTED))
for i, ((m, lab, c), d) in enumerate(zip(MODELS, data)):
    bp["boxes"][i].set_facecolor(c); bp["boxes"][i].set_alpha(0.25)
    ax.scatter(i + 1 + rng.uniform(-0.17, 0.17, len(d)), d, s=14, color=c, lw=0, alpha=0.9, zorder=3)
    ax.text(i + 1.3, np.mean(d), f"mean {np.mean(d):.3f}", va="center", fontsize=8.5, color=INK)
    ax.plot([i + 0.78, i + 1.22], [np.mean(d)] * 2, color=INK, lw=1, ls=":")
ax.set_xticks([1, 2, 3], [l for _, l, _ in MODELS]); ax.set_ylabel("Dice per image"); ax.set_ylim(0, 1)
ax.set_title("Per-image Dice on 53 damaged test images (each dot = one image)")
clean(ax); fig.tight_layout(); save(fig, "fig_rev_dice_distribution")

# ---------------------------------------------------------------- 4. threshold sensitivity (OOF, chosen min area)
g = pd.read_csv(os.path.join(E00, "final_eval", "postproc_grid.csv")); g = g[g.min_area_px == 100]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
axs[0].plot(g.threshold, g.mean_dice, color=C_RGB, lw=2, marker="o", ms=5)
axs[0].set_ylim(0.40, 0.46); axs[0].set_ylabel("Mean per-image Dice (OOF)")
axs[0].set_title("(a) Damage segmentation vs threshold")
axs[1].plot(g.threshold, g.healthy_mean_pct, color=C_TH, lw=2, marker="o", ms=5)
axs[1].set_ylim(0, 0.2); axs[1].set_ylabel("Mean false-alarm area on healthy\nimages (% of image, OOF)")
axs[1].set_title("(b) False alarms vs threshold")
for a in axs:
    a.axvline(0.35, color=MUTED, ls="--", lw=1)
    a.text(0.352, a.get_ylim()[0] + 0.02 * np.diff(a.get_ylim())[0], "chosen 0.35", fontsize=8.5, color=MUTED)
    a.set_xlabel("Probability threshold (minimum region 100 px)"); clean(a)
fig.tight_layout(); save(fig, "fig_rev_threshold")

# ---------------------------------------------------------------- 5. pixel confusion matrix (pooled test)
tp, fp, fn = ti.tp.sum(), ti.fp.sum(), ti.fn.sum()
tn = len(ti) * 640 * 480 - tp - fp - fn
M = np.array([[tp, fn], [fp, tn]], float); R = M / M.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(5.4, 4.2))
cm = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f5f7fa", C_RGB])
ax.imshow(R, cmap=cm, vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{100 * R[i, j]:.1f}%\n({int(M[i, j]):,} px)", ha="center", va="center", fontsize=10,
                color="white" if R[i, j] > 0.6 else INK)
ax.set_xticks([0, 1], ["Predicted damage", "Predicted not damage"]); ax.set_yticks([0, 1], ["Actual damage", "Actual not damage"])
ax.set_title("Pixel confusion matrix, test set (53 images, e00)\nrow-normalised")
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0); fig.tight_layout(); save(fig, "fig_rev_confusion_matrix")

# ---------------------------------------------------------------- 6. model comparison (test)
S = {m: json.load(open(os.path.join(B, f"{m}_test_summary.json")))["test_damaged"] for m, _, _ in MODELS}
mets = [("Dice", lambda s: s["mean_per_image"]["dice"]), ("IoU", lambda s: s["mean_per_image"]["iou"]),
        ("Precision", lambda s: s["mean_per_image"]["precision"]), ("Recall", lambda s: s["mean_per_image"]["recall"]),
        ("Boundary-tolerant\nDice", lambda s: s["mean_per_image"]["bt_dice"]), ("Region\nrecall", lambda s: s["regions"]["region_recall"])]
fig, ax = plt.subplots(figsize=(10, 4.2))
x = np.arange(len(mets)); w = 0.26
for k, (m, lab, c) in enumerate(MODELS):
    v = [f(S[m]) for _, f in mets]
    bars = ax.bar(x + (k - 1) * w, v, w - 0.03, color=c, label=lab)
    for b_, val in zip(bars, v):
        ax.text(b_.get_x() + b_.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(x, [n for n, _ in mets]); ax.set_ylim(0, 0.9); ax.set_ylabel("Score (test set, 53 damaged images)")
ax.set_title("Test-set metrics of the three models (Dice differences vs e00 not significant: e01 p = 0.60, e10 p = 0.23)")
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5); clean(ax)
fig.tight_layout(); save(fig, "fig_rev_model_comparison")

# ---------------------------------------------------------------- 7. Dice vs amount of damage
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
ax = axs[0]
ax.scatter(ti.gt_pct, ti.dice, s=24, color=C_RGB, edgecolor="white", lw=0.5)
ax.set_xlabel("Reference damage (% of image)"); ax.set_ylabel("Dice per image"); ax.set_ylim(0, 1)
ax.set_title("(a) Dice vs amount of damage (test, e00)"); clean(ax, "both")
ax = axs[1]
ax.scatter(ti.recall, ti.precision, s=24, color=C_RGB, edgecolor="white", lw=0.5)
ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel("Recall (share of damage found)"); ax.set_ylabel("Precision (share of prediction correct)")
ax.set_title("(b) Precision vs recall per image (test, e00)"); clean(ax, "both")
fig.tight_layout(); save(fig, "fig_rev_dice_vs_damage")

print("pixel confusion:", dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn)))
print("r(gt_pct, dice) test:", round(np.corrcoef(ti.gt_pct, ti.dice)[0, 1], 3))
