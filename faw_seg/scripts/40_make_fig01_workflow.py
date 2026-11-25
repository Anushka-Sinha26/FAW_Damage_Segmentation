"""Figure 1 -- overall workflow, built from real project data (test image D152, e00 ensemble).
Usage: python make_fig01.py <fig01_D152_data.npz> <out_dir>
"""
import sys, json, os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
from matplotlib.lines import Line2D

src, out = sys.argv[1], sys.argv[2]
D = np.load(src, allow_pickle=True)
rgb, gt, T, Ta = D["rgb"], D["gt"], D["T"], D["Ta"]
prob, pre, pred, leaf = D["prob"], D["pre"], D["pred"], D["leaf"]
thumbs, polys = D["thumbs"], D["polys"]
st = json.loads(str(D["stats"]))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})

W, H = 7.48, 8.4                        # inches (190 mm full page width)
fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")

INK = "#1f2933"; MUTED = "#52606d"
BLUE, BLUE_BG = "#1f6fb2", "#e8f1fa"
GREEN, GREEN_BG = "#2f855a", "#e9f5ee"
ORANGE, ORANGE_BG = "#c05621", "#fdf0e6"
SLATE, SLATE_BG = "#334e68", "#eef2f6"


def box(x, y, w, h, fc="white", ec="#9aa5b1", lw=0.6, ls="-", r=0.06, z=1):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def text(x, y, s, size=7, weight="normal", color=INK, ha="center", va="center", z=5, **kw):
    bg.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=z, **kw)


def arrow(x0, y0, x1, y1, color=INK):
    bg.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=color, mutation_scale=8), zorder=6)


def img(x, y, w, h, a, cmap=None, vmin=None, vmax=None, border="#9aa5b1"):
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest" if a.dtype == bool else "antialiased")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(border); s.set_linewidth(0.5)
    return ax, im


def band(y, h, num, title, color, fill):
    box(0.08, y, W - 0.16, h, fc=fill, ec=color, lw=0.8, r=0.08, z=0)
    bg.add_patch(FancyBboxPatch((0.16, y + h - 0.27), 0.42 + 0.072 * len(title), 0.21, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=color, ec="none", zorder=2))
    text(0.26, y + h - 0.165, f"{num}", size=8, weight="bold", color="white", ha="left")
    text(0.42, y + h - 0.165, title, size=7.5, weight="bold", color="white", ha="left")


def overlay(base, mask, color, alpha=0.55):
    o = base.astype(np.float32) / 255
    c = np.array(color, np.float32)
    o[mask] = (1 - alpha) * o[mask] + alpha * c
    return o


IW, IH = 1.12, 0.84                      # standard 4:3 image panel

# ---------------------------------------------------------------- band 1: data
y1, h1 = 6.62, 1.62
band(y1, h1, 1, "DATA ACQUISITION", SLATE, SLATE_BG)
cy = y1 + 0.66
box(0.25, cy - 0.42, 1.30, 0.84)
text(0.90, cy + 0.17, "FLIR E8 handheld", weight="bold")
text(0.90, cy - 0.02, "one shutter press:", color=MUTED)
text(0.90, cy - 0.20, "visual + radiometric", color=MUTED)
arrow(1.58, cy, 1.88, cy)
img(1.93, cy - IH / 2, IW, IH, rgb)
text(1.93 + IW / 2, cy + IH / 2 + 0.09, "RGB frame 640 × 480", size=6.5)
ax, im = img(3.20, cy - IH / 2, IW, IH, T, cmap="inferno")
text(3.20 + IW / 2, cy + IH / 2 + 0.09, "Thermal frame 320 × 240", size=6.5)
cax = fig.add_axes([(3.20 + IW + 0.04) / W, (cy - IH / 2) / H, 0.05 / W, IH / H])
cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
cb.set_label("apparent °C", fontsize=5, labelpad=1)
text(3.08, cy, "+", size=10, weight="bold")
arrow(4.70, cy, 4.95, cy)
box(5.00, cy - 0.42, 2.28, 0.84)
text(6.14, cy + 0.20, "Pairing check", weight="bold")
text(6.14, cy - 0.02, "identical EXIF capture time +", color=MUTED)
text(6.14, cy - 0.19, "embedded-image match: 633/633", color=MUTED)

# ---------------------------------------------------------------- band 2: annotation
y2, h2 = 4.90, 1.58
band(y2, h2, 2, "ANNOTATION", GREEN, GREEN_BG)
cy = y2 + 0.66
ax, _ = img(0.30, cy - IH / 2, IW, IH, rgb)
for p in polys:
    ax.add_patch(Polygon(p, closed=True, fill=False, ec="#e53e3e", lw=0.6))
text(0.30 + IW / 2, cy - IH / 2 - 0.10, f"Labelme polygons ({st['n_poly']})", size=6.5)
arrow(1.47, cy, 1.80, cy)
img(1.85, cy - IH / 2, IW, IH, gt, cmap="gray")
text(1.85 + IW / 2, cy - IH / 2 - 0.10, "Binary mask: 255 = damage, 0 = other", size=6.5)
arrow(3.02, cy, 3.35, cy)
box(3.40, cy - 0.48, 3.88, 0.96)
text(3.55, cy + 0.30, "Mask audit", weight="bold", ha="left")
for k, s in enumerate(["325 masks, strictly binary, sizes match RGB, none empty",
                       "Damage = median 1.76% of image (range 0.07–9.95%)",
                       "Original annotations preserved; 2 mechanical fixes in a new version"]):
    text(3.55, cy + 0.08 - k * 0.19, "•  " + s, color=MUTED, ha="left", size=6.5)

# ---------------------------------------------------------------- band 3: modelling
y3, h3 = 2.62, 2.14
band(y3, h3, 3, "MODELLING", ORANGE, ORANGE_BG)
cy = y3 + 0.88
# grouped split
box(0.22, cy - 0.70, 1.42, 1.48)
tw = 0.25
for k in range(5):
    img(0.30 + k * (tw + 0.02), cy + 0.22, tw, tw * 0.75, thumbs[k])
fc = ["#9fb3c8", "#9fb3c8", "#9fb3c8", "#9fb3c8", "#1f6fb2"]
for k in range(5):
    bg.add_patch(Rectangle((0.30 + k * (tw + 0.02), cy + 0.11), tw, 0.07, fc=fc[k], ec="none", zorder=3))
text(0.93, cy + 0.50, "1 fold = validation, 4 = training", size=5.5, color=MUTED)
text(0.93, cy - 0.20, "Grouped split", weight="bold", size=6.5)
text(0.93, cy - 0.35, "by capture block", color=MUTED, size=6)
text(0.93, cy - 0.48, "5 CV folds (534 images)", color=MUTED, size=6)
text(0.93, cy - 0.60, "+ held-out test (99)", color=MUTED, size=6)
arrow(1.66, cy, 1.88, cy)


def unet(x, y, w, h, enc="#4a90d9", dec="#4a90d9", film=False):
    hs = [1.0, 0.78, 0.58, 0.40, 0.26]
    n = len(hs); cw = w / (2 * n + 1.2)
    for k, s in enumerate(hs):           # encoder
        bg.add_patch(Rectangle((x + k * cw * 1.05, y + h / 2 - s * h / 2), cw * 0.75, s * h, fc=enc, ec="white", lw=0.4, zorder=3))
    xb = x + n * cw * 1.05
    bg.add_patch(Rectangle((xb, y + h / 2 - 0.08 * h), cw * 1.1, 0.16 * h, fc="#cbd2d9", ec="white", lw=0.4, zorder=3))
    for k, s in enumerate(hs[::-1]):     # decoder
        xd = xb + cw * 1.25 + k * cw * 1.05
        bg.add_patch(Rectangle((xd, y + h / 2 - s * h / 2), cw * 0.75, s * h, fc=dec, ec="white", lw=0.4, zorder=3))
    for k, s in enumerate(hs[:4]):       # skips
        xs0 = x + k * cw * 1.05 + cw * 0.75
        xs1 = xb + cw * 1.25 + (n - 1 - k) * cw * 1.05
        yy = y + h / 2 + s * h / 2 - 0.03
        bg.add_line(Line2D([xs0, xs1], [yy, yy], lw=0.4, color="#7b8794", ls=(0, (2, 1)), zorder=2))


# phase 1/2
box(1.92, cy - 0.70, 1.86, 1.48, fc="white", ec=BLUE, lw=0.8, ls="--")
bg.add_patch(FancyBboxPatch((1.92, cy + 0.58), 1.86, 0.20, boxstyle="round,pad=0,rounding_size=0.05", fc=BLUE, ec="none", zorder=2))
text(2.85, cy + 0.68, "Phases 1–2: RGB only", weight="bold", color="white", size=6.5)
img(2.00, cy - 0.06, 0.52, 0.39, rgb)
text(2.26, cy - 0.17, "RGB patch", size=5.5, color=MUTED); text(2.26, cy - 0.29, "320×320×3", size=5.5, color=MUTED)
unet(2.60, cy - 0.32, 1.12, 0.78)
text(2.85, cy - 0.53, "MobileNetV2 (ImageNet) U-Net", weight="bold", size=6.5)
arrow(3.80, cy, 3.98, cy)
# phase 3
box(4.02, cy - 0.70, 2.02, 1.48, fc="white", ec=ORANGE, lw=0.8, ls="--")
bg.add_patch(FancyBboxPatch((4.02, cy + 0.58), 2.02, 0.20, boxstyle="round,pad=0,rounding_size=0.05", fc=ORANGE, ec="none", zorder=2))
text(5.03, cy + 0.68, "Phase 3: RGB + thermal", weight="bold", color="white", size=6.5)
img(4.10, cy + 0.06, 0.48, 0.36, rgb)
ax, _ = img(4.10, cy - 0.36, 0.48, 0.36, np.nan_to_num(Ta, nan=np.nanmin(Ta)), cmap="inferno")
text(4.34, cy - 0.46, "registered", size=5.5, color=MUTED); text(4.34, cy - 0.57, "thermal", size=5.5, color=MUTED)
unet(4.70, cy - 0.20, 1.26, 0.70)
# thermal branch + FiLM arrows
for k in range(5):
    xk = 4.70 + k * (1.26 / (2 * 5 + 1.2)) * 1.05 + 0.03
    bg.add_patch(Rectangle((xk, cy - 0.40), 0.07, 0.10, fc="#ed8936", ec="none", zorder=3))
    arrow(xk + 0.035, cy - 0.29, xk + 0.035, cy - 0.20, color=ORANGE)
text(5.33, cy - 0.53, "thermal branch → FiLM (5 scales)", size=5.5, color=ORANGE)
arrow(6.06, cy, 6.20, cy)
# test lock
box(6.24, cy - 0.70, 1.04, 1.48)
bg.add_patch(Rectangle((6.66, cy + 0.18), 0.20, 0.17, fc=INK, ec="none", zorder=3))
bg.add_patch(matplotlib.patches.Arc((6.76, cy + 0.35), 0.13, 0.20, theta1=0, theta2=180, lw=1.6, color=INK, zorder=3))
text(6.76, cy - 0.03, "Held-out test", weight="bold", size=6.5)
text(6.76, cy - 0.20, "53 damaged +", color=MUTED, size=6)
text(6.76, cy - 0.34, "46 healthy", color=MUTED, size=6)
text(6.76, cy - 0.50, "evaluated once", color=MUTED, size=6)

# ---------------------------------------------------------------- band 4: outputs
y4, h4 = 0.12, 2.36
band(y4, h4, 4, "OUTPUTS AND EVALUATION", SLATE, SLATE_BG)
cy = y4 + 1.22
pw, ph = 0.94, 0.705
xs = [0.24, 1.40, 2.56, 3.72, 4.88]
labels = ["Probability map", "Threshold 0.35", "Regions <100 px removed", "Leaf mask (CIELAB a*)", "Damage % of leaf"]
ax, im = img(xs[0], cy - ph / 2, pw, ph, prob, cmap="viridis", vmin=0, vmax=1)
img(xs[1], cy - ph / 2, pw, ph, pre, cmap="gray")
img(xs[2], cy - ph / 2, pw, ph, pred, cmap="gray")
img(xs[3], cy - ph / 2, pw, ph, leaf, cmap=matplotlib.colors.ListedColormap(["white", "#2f855a"]))
ov = overlay(rgb, leaf & ~pred, (0.18, 0.62, 0.30), 0.35)
ov = overlay((ov * 255).astype(np.uint8), pred, (0.90, 0.10, 0.10), 0.75)
ax, _ = img(xs[4], cy - ph / 2, pw, ph, ov)
cs = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
for c in cs:
    c = c[:, 0, :]
    ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color="#ffd400", lw=0.45)
for k, x in enumerate(xs):
    text(x + pw / 2, cy + ph / 2 + 0.10, labels[k], weight="bold", size=5.9)
    if k < 4:
        arrow(x + pw + 0.03, cy, xs[k + 1] - 0.03, cy)
text(xs[4] + pw / 2, cy - ph / 2 - 0.12, f"{st['pred_leaf']:.1f}% (reference {st['gt_leaf']:.1f}%)", size=6, color=INK, weight="bold")
text(xs[4] + pw / 2, cy - ph / 2 - 0.26, "red = predicted;", size=5.5, color=MUTED)
text(xs[4] + pw / 2, cy - ph / 2 - 0.38, "yellow = reference outline", size=5.5, color=MUTED)
text(xs[0] + pw / 2, cy - ph / 2 - 0.12, "5-fold ensemble mean", size=6, color=MUTED)
text(xs[3] + pw / 2, cy - ph / 2 - 0.12, "per-image Otsu", size=6, color=MUTED)
arrow(xs[4] + pw + 0.03, cy, 6.12, cy)
box(6.15, cy - 0.78, 1.15, 1.56, fc="white", ec=GREEN, lw=0.8)
text(6.725, cy + 0.58, "Evaluation", weight="bold", color=GREEN)
for k, s in enumerate(["Dice, IoU,", "precision, recall", "Boundary-tolerant", "Dice (3 px)", "Region recall /", "precision", "Healthy false alarms", "Paired bootstrap +", "permutation tests"]):
    text(6.725, cy + 0.36 - k * 0.135, s, size=5.6, color=MUTED)
text(W / 2, y4 + 0.14, "Example: held-out test image D152 (Dice 0.52; test median 0.48). Main RGB model (e00), threshold and minimum region chosen on out-of-fold predictions.",
     size=5.6, color=MUTED)

os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig01_workflow.pdf"))
fig.savefig(os.path.join(out, "fig01_workflow.png"), dpi=600)
print("ok")
