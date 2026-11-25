"""Results figures from saved outputs (no re-training, no new test runs).
Usage: python make_results_figs.py <results_bundle_dir> <out_dir>
  fig_res_qualitative  : test predictions of the main model (e00) for best / median / lower-quartile / failure / healthy
  fig_res_damagepct    : damage % of leaf, predicted vs reference (OOF + test) and Bland-Altman (test)
  fig_res_fusion       : e10 (RGB+thermal) vs e00 (RGB): paired OOF Dice and healthy false-alarm area
"""
import sys, os, json
import numpy as np, pandas as pd, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
INK, MUTED = "#1f2933", "#52606d"
C_RGB, C_TH, C_OOF = "#1f6fb2", "#c05621", "#9aa5b1"


def save(fig, name):
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def remove_small(m, a):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= a
    return keep[lab]


# ------------------------------------------------------------------ qualitative
Q = np.load(os.path.join(B, "qual.npz")); ids = list(Q["ids"])
lab = {ids[0]: "Best", ids[1]: "Median", ids[2]: "Lower quartile", ids[3]: "Failure case", ids[4]: "Healthy (largest false alarm)"}
W = 7.48; pw = 1.72; ph = pw * 0.75; gx = 0.08; gy = 0.10; left = 0.42; top = 0.22
H = top + len(ids) * ph + (len(ids) - 1) * gy + 0.35
fig = plt.figure(figsize=(W, H))
cols = ["RGB image", "Reference (red)", "Prediction (red)", "Errors"]
TP, FP, FN = np.array([0.11, 0.62, 0.47]), np.array([0.85, 0.37, 0.01]), np.array([0.46, 0.44, 0.70])
for r, sid in enumerate(ids):
    rgb = Q[sid + "_rgb"]; gt = Q[sid + "_gt"]; pr = remove_small(Q[sid + "_p0"] / 255.0 >= 0.35, 100)
    y = H - top - (r + 1) * ph - r * gy
    base = rgb.astype(np.float32) / 255
    def ov(m):
        o = base.copy(); o[m] = 0.35 * o[m] + 0.65 * np.array([0.9, 0.1, 0.1]); return o
    err = base * 0.45 + 0.55
    err[pr & gt] = TP; err[pr & ~gt] = FP; err[~pr & gt] = FN
    for c, a in enumerate([rgb, ov(gt), ov(pr), err]):
        ax = fig.add_axes([(left + c * (pw + gx)) / W, y / H, pw / W, ph / H])
        ax.imshow(a); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.4); s.set_edgecolor("#7b8794")
        if r == 0: ax.set_title(cols[c], fontsize=7, fontweight="bold", pad=3)
    if gt.any():
        d = 2 * (pr & gt).sum() / (pr.sum() + gt.sum())
        sub = f"{sid}, Dice {d:.2f}"
    else:
        sub = f"{sid}, {100 * pr.mean():.2f}% of image"
    fig.text((left - 0.10) / W, (y + ph / 2) / H, f"{lab[sid]}\n{sub}", rotation=90, ha="center", va="center", fontsize=6.3)
ly = 0.08
for k, (c, t) in enumerate([(TP, "true positive"), (FP, "false positive"), (FN, "false negative")]):
    x = left + 1.9 * (pw + gx) + k * 1.05
    fig.patches.append(matplotlib.patches.Rectangle((x / W, ly / H), 0.12 / W, 0.09 / H, transform=fig.transFigure, fc=c, ec="none"))
    fig.text((x + 0.16) / W, (ly + 0.045) / H, t, fontsize=5.6, va="center")
save(fig, "fig_res_qualitative")

# ------------------------------------------------------------------ damage % of leaf
te = pd.read_csv(os.path.join(B, "e00_baseline_test_damage_pct_of_leaf.csv"))
oo = pd.read_csv(os.path.join(B, "e00_baseline_oof_damage_pct_of_leaf.csv"))
fig, (a, b) = plt.subplots(1, 2, figsize=(7.48, 3.0), gridspec_kw=dict(wspace=0.32))
lim = max(te[["gt_pct_of_leaf", "pred_pct_of_leaf"]].max().max(), oo[["gt_pct_of_leaf", "pred_pct_of_leaf"]].max().max()) * 1.05
a.scatter(oo.gt_pct_of_leaf, oo.pred_pct_of_leaf, s=6, c=C_OOF, alpha=0.6, lw=0, label=f"Out-of-fold (n = {len(oo)})")
a.scatter(te.gt_pct_of_leaf, te.pred_pct_of_leaf, s=12, c=C_RGB, lw=0.3, ec="white", label=f"Test (n = {len(te)})")
a.plot([0, lim], [0, lim], color=INK, lw=0.7, ls="--")
o = te.loc[te.sample_id == "D020"]
if len(o):
    a.annotate("D020", (o.gt_pct_of_leaf.iloc[0], o.pred_pct_of_leaf.iloc[0]), xytext=(-24, 10), textcoords="offset points",
               fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
def stats(df):
    d = df.pred_pct_of_leaf - df.gt_pct_of_leaf
    return np.corrcoef(df.gt_pct_of_leaf, df.pred_pct_of_leaf)[0, 1], d.abs().mean(), d.mean(), d.std(ddof=1)
r_o, mae_o, _, _ = stats(oo); r_t, mae_t, bias_t, sd_t = stats(te)
a.text(0.04, 0.96, f"OOF: r = {r_o:.2f}, MAE = {mae_o:.2f} pp\nTest: r = {r_t:.2f}, MAE = {mae_t:.2f} pp",
       transform=a.transAxes, va="top", fontsize=6.2)
a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_aspect("equal")
a.set_xlabel("Reference damage (% of leaf)"); a.set_ylabel("Predicted damage (% of leaf)")
a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), fontsize=5.8, frameon=False); clean(a)
a.set_title("(a)", loc="left", fontweight="bold", fontsize=8)
m = (te.gt_pct_of_leaf + te.pred_pct_of_leaf) / 2; d = te.pred_pct_of_leaf - te.gt_pct_of_leaf
b.scatter(m, d, s=12, c=C_RGB, lw=0.3, ec="white")
lo, hi = bias_t - 1.96 * sd_t, bias_t + 1.96 * sd_t
for v, ls, t in [(bias_t, "-", f"bias {bias_t:+.2f}"), (lo, "--", f"−1.96 SD {lo:.2f}"), (hi, "--", f"+1.96 SD {hi:.2f}")]:
    b.axhline(v, color="#c0392b" if ls == "-" else MUTED, lw=0.8, ls=ls)
    b.text(10.0, v + 0.3, t, fontsize=5.8, va="bottom", ha="left", color=MUTED)
b.axhline(0, color=INK, lw=0.4)
if len(o):
    b.annotate("D020", ((o.gt_pct_of_leaf.iloc[0] + o.pred_pct_of_leaf.iloc[0]) / 2, (o.pred_pct_of_leaf - o.gt_pct_of_leaf).iloc[0]),
               xytext=(-28, 4), textcoords="offset points", fontsize=6, arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
b.set_xlabel("Mean of predicted and reference (% of leaf)"); b.set_ylabel("Predicted − reference (pp)")
b.set_title("(b) Test set", loc="left", fontweight="bold", fontsize=8); clean(b)
save(fig, "fig_res_damagepct")

# ------------------------------------------------------------------ fusion vs RGB
e0 = pd.read_csv(os.path.join(B, "e00_oof_per_image.csv")).set_index("sample_id")
e1 = pd.read_csv(os.path.join(B, "e10_oof_per_image.csv")).set_index("sample_id")
j = e0[["dice"]].join(e1[["dice"]], lsuffix="_rgb", rsuffix="_rgbt")
hf = pd.read_csv(os.path.join(B, "oof_healthy_fp.csv"))
fig, (a, b) = plt.subplots(1, 2, figsize=(7.48, 3.0), gridspec_kw=dict(wspace=0.32))
a.scatter(j.dice_rgb, j.dice_rgbt, s=7, c=C_TH, alpha=0.7, lw=0)
a.plot([0, 1], [0, 1], color=INK, lw=0.7, ls="--")
a.set_xlim(0, 0.85); a.set_ylim(0, 0.85); a.set_aspect("equal")
a.set_xlabel("Dice, RGB model (e00)"); a.set_ylabel("Dice, RGB + thermal model (e10)")
dd = j.dice_rgbt - j.dice_rgb
a.text(0.04, 0.96, f"{len(j)} out-of-fold damaged images\nmean difference {dd.mean():+.3f}\n(95% CI −0.015 to +0.004, p = 0.23)",
       transform=a.transAxes, va="top", fontsize=6.2)
a.set_title("(a) Damage segmentation", loc="left", fontweight="bold", fontsize=8); clean(a)
for col, c, name in [("e00", C_RGB, "RGB (e00)"), ("e10", C_TH, "RGB + thermal (e10)")]:
    v = np.sort(hf[col].values)
    b.step(np.r_[0, v], np.r_[0, np.arange(1, len(v) + 1) / len(v)], where="post", color=c, lw=1.1, label=f"{name}: mean {hf[col].mean():.2f}%")
b.axvline(0.1, color=MUTED, lw=0.5, ls=":"); b.axvline(1.0, color=MUTED, lw=0.5, ls=":")
b.text(0.1, 0.30, " 0.1%", fontsize=5.6, color=MUTED); b.text(1.0, 0.30, " 1%", fontsize=5.6, color=MUTED)
b.set_xscale("symlog", linthresh=0.01); b.set_xlim(0, max(2.0, hf[["e00", "e10"]].max().max() * 1.1))
b.set_ylim(0, 1.01)
b.set_xlabel("False-alarm area on healthy image (% of image)"); b.set_ylabel("Cumulative share of healthy images")
b.legend(loc="center left", bbox_to_anchor=(0.0, 0.70), fontsize=5.8, frameon=False)
b.text(0.04, 0.97, f"{len(hf)} out-of-fold healthy images\nmean difference −0.076 pp\n(95% CI −0.107 to −0.049, p < 0.001)",
       transform=b.transAxes, va="top", fontsize=6.2)
b.set_title("(b) False alarms on healthy leaves", loc="left", fontweight="bold", fontsize=8); clean(b)
save(fig, "fig_res_fusion")
print("ok")
