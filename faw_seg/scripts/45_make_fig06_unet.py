"""Figure 6 -- MobileNetV2 U-Net architecture (from src/phase1/experiment_06/model_experiment_06_mobilenetv2_unet.py).
Usage: python make_fig06.py <fig01_D152_data.npz> <out_dir>
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

D = np.load(sys.argv[1], allow_pickle=True); out = sys.argv[2]
rgb = D["rgb"][80:400, 160:480]; prob = D["prob"][80:400, 160:480]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "pdf.fonttype": 42})
INK, MUTED = "#1f2933", "#52606d"
ENC, ENC_E = "#c9ced4", "#7b8794"          # frozen encoder (grey)
DEC, DEC_E = "#9cc3e6", "#1f6fb2"          # trainable decoder (blue)
HEAD, HEAD_E = "#f6c89f", "#c05621"
SKIP = "#2f855a"

W, H = 7.48, 4.55
fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")


def block(cx, cy, w, h, fc, ec, l1, l2=None, l3=None):
    bg.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=fc, ec=ec, lw=0.8, zorder=3))
    bg.text(cx, cy + (0.07 if l2 else 0), l1, ha="center", va="center", fontsize=6.3, fontweight="bold", zorder=4)
    if l2:
        bg.text(cx, cy - 0.07, l2, ha="center", va="center", fontsize=5.6, color=MUTED, zorder=4)
    if l3:
        bg.text(cx, cy - h / 2 - 0.08, l3, ha="center", va="top", fontsize=5.4, color=MUTED, zorder=4)


def arrow(x0, y0, x1, y1, color=INK, lw=0.9, ls="-"):
    bg.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=7, linestyle=ls), zorder=2)


# levels: (spatial size, encoder layer, enc channels, decoder filters)
levels = [(160, "block_1_expand_relu", 96, 64),
          (80, "block_3_expand_relu", 144, 96),
          (40, "block_6_expand_relu", 192, 128),
          (20, "block_13_expand_relu", 576, 256)]
ys = [3.30, 2.62, 1.94, 1.26]
ex, dx = 1.95, 4.75; bw, bh = 1.45, 0.42; dw = 2.05

# input image
ax = fig.add_axes([0.10 / W, 3.62 / H, 0.62 / W, 0.62 / H]); ax.imshow(rgb); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_linewidth(0.4)
bg.text(0.41, 3.55, "RGB patch\n320 × 320 × 3", ha="center", va="top", fontsize=5.8)
block(ex, 3.93, bw, 0.32, "white", ENC_E, "Rescale [0, 1] → [−1, 1]")
arrow(0.76, 3.93, ex - bw / 2 - 0.02, 3.93)
arrow(ex, 3.93 - 0.16, ex, ys[0] + bh / 2 + 0.02)

for (s, name, c, f), y in zip(levels, ys):
    block(ex, y, bw, bh, ENC, ENC_E, f"{s} × {s} × {c}", name.replace("_", "\\_") if False else name)
for y0, y1 in zip(ys[:-1], ys[1:]):
    arrow(ex, y0 - bh / 2, ex, y1 + bh / 2 + 0.02)
# bottleneck
yb = 0.52
block((ex + dx) / 2, yb, bw, bh, ENC, ENC_E, "10 × 10 × 1280", "out_relu (bottleneck)")
bg.annotate("", xy=((ex + dx) / 2 - bw / 2 - 0.02, yb), xytext=(ex, ys[-1] - bh / 2),
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK, mutation_scale=7, connectionstyle="angle,angleA=-90,angleB=180"), zorder=2)
bg.annotate("", xy=(dx, ys[-1] - bh / 2 - 0.02), xytext=((ex + dx) / 2 + bw / 2, yb),
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK, mutation_scale=7, connectionstyle="angle,angleA=0,angleB=-90"), zorder=2)

# decoder
for k, ((s, name, c, f), y) in enumerate(zip(levels, ys)):
    block(dx, y, dw, bh, DEC, DEC_E, f"Decoder {4 - k}: {s} × {s} × {f}", f"up ×2 → concat skip → 2 × [Conv 3×3 – BN – ReLU]")
    arrow(ex + bw / 2, y, dx - dw / 2 - 0.02, y, color=SKIP, lw=0.8)
    bg.text((ex + bw / 2 + dx - dw / 2) / 2, y + 0.06, f"skip ({c} ch)", ha="center", va="bottom", fontsize=5.4, color=SKIP)
for y0, y1 in zip(ys[::-1][:-1], ys[::-1][1:]):
    arrow(dx, y0 + bh / 2, dx, y1 - bh / 2 - 0.02)

# head
hx = 6.62
block(dx, 3.93, dw, 0.36, HEAD, HEAD_E, "Head: 320 × 320 × 1", "up ×2 → Conv 3×3 (32) → Conv 1×1 + sigmoid")
arrow(dx, ys[0] + bh / 2, dx, 3.93 - 0.18)
ax = fig.add_axes([(hx + 0.10) / W, 3.62 / H, 0.62 / W, 0.62 / H]); ax.imshow(prob, cmap="viridis", vmin=0, vmax=1)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_linewidth(0.4)
arrow(dx + dw / 2, 3.93, hx + 0.08, 3.93)
bg.text(hx + 0.41, 3.55, "Damage probability\n320 × 320 × 1", ha="center", va="top", fontsize=5.8)

# column titles + legend
bg.text(ex, 4.40, "Encoder: MobileNetV2 (ImageNet, frozen)", ha="center", fontsize=6.8, fontweight="bold", color=MUTED)
bg.text(dx, 4.40, "Decoder (trainable)", ha="center", fontsize=6.8, fontweight="bold", color=DEC_E)
lx, ly = 6.10, 0.80
for k, (fc, ec, lab) in enumerate([(ENC, ENC_E, "frozen (ImageNet weights)"), (DEC, DEC_E, "trainable decoder"),
                                   (HEAD, HEAD_E, "output head")]):
    bg.add_patch(Rectangle((lx, ly - k * 0.17), 0.16, 0.10, fc=fc, ec=ec, lw=0.6))
    bg.text(lx + 0.22, ly - k * 0.17 + 0.05, lab, va="center", fontsize=5.8)
bg.add_line(plt.Line2D([lx, lx + 0.16], [ly - 3 * 0.17 + 0.05] * 2, color=SKIP, lw=0.9))
bg.text(lx + 0.22, ly - 3 * 0.17 + 0.05, "skip connection", va="center", fontsize=5.8)

os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig06_unet.pdf"))
fig.savefig(os.path.join(out, "fig06_unet.png"), dpi=600)
print("ok")
