"""Fusion architecture figure (from src/faw/fusion.py, e10).
Usage: python make_fig_fusion.py <fig01_D152_data.npz> <out_dir>
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib.patheffects as pe

D = np.load(sys.argv[1], allow_pickle=True); out = sys.argv[2]
rgb = D["rgb"][80:400, 160:480]
Ta = D["Ta"][80:400, 160:480].astype(np.float32)
v = np.isfinite(Ta); z = np.zeros_like(Ta); z[v] = np.clip((Ta[v] - Ta[v].mean()) / (Ta[v].std() + 1e-6), -5, 5)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
INK, MUTED = "#1f2933", "#52606d"
ENC, ENC_E = "#c9ced4", "#7b8794"
TH, TH_E = "#fbd3b0", "#c05621"
DEC, DEC_E = "#9cc3e6", "#1f6fb2"
HEAD, HEAD_E = "#e6e9ed", "#52606d"
SKIP = "#2f855a"

W, H = 7.48, 5.45
fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")


def block(cx, cy, w, h, fc, ec, l1, l2=None):
    bg.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=fc, ec=ec, lw=0.8, zorder=3))
    bg.text(cx, cy + (0.065 if l2 else 0), l1, ha="center", va="center", fontsize=5.9, fontweight="bold", zorder=4)
    if l2:
        bg.text(cx, cy - 0.075, l2, ha="center", va="center", fontsize=5.3, color=MUTED, zorder=4)


def arrow(x0, y0, x1, y1, color=INK, lw=0.85, cs=None, bridge=False):
    kw = dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=7)
    if cs: kw["connectionstyle"] = cs
    if bridge: kw["path_effects"] = [pe.Stroke(linewidth=4, foreground="white"), pe.Normal()]
    bg.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=kw, zorder=6 if bridge else 2)


ex, tx, fx, dx = 1.25, 3.05, 4.30, 5.95
ew, tw, dw, bh = 1.30, 1.42, 2.00, 0.36
ys = [4.15, 3.45, 2.75, 2.05, 1.30]
rgb_lv = [("160 × 160 × 96", "block_1_expand_relu"), ("80 × 80 × 144", "block_3_expand_relu"),
          ("40 × 40 × 192", "block_6_expand_relu"), ("20 × 20 × 576", "block_13_expand_relu"),
          ("10 × 10 × 1280", "out_relu (bottleneck)")]
th_ch = [16, 32, 48, 64, 96]; sizes = [160, 80, 40, 20, 10]
dec = [(160, 64), (80, 96), (40, 128), (20, 256)]

# inputs
for (img, cmap, cx, lab) in [(rgb, None, ex, "RGB patch 320 × 320 × 3"), (z, "inferno", tx, "Thermal 320 × 320 × 2\n(z-scored temperature + valid mask)")]:
    a = fig.add_axes([(cx - 0.30) / W, 4.75 / H, 0.60 / W, 0.60 / H]); a.imshow(img, cmap=cmap); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_linewidth(0.4)
    bg.text(cx + 0.36, 5.05, lab, ha="left", va="center", fontsize=5.6)
    arrow(cx, 4.73, cx, (ys[0] + bh / 2 + 0.02) if cx == ex else (ys[0] - 0.31 + 0.17), bridge=(cx != ex), color=INK if cx == ex else TH_E)

for k, y in enumerate(ys):
    yt = y - 0.31
    block(ex, y, ew, bh, ENC, ENC_E, *rgb_lv[k])
    block(tx, yt, tw, 0.30, TH, TH_E, f"Thermal {k + 1}: {sizes[k]} × {sizes[k]} × {th_ch[k]}", "2×[Conv3×3–BN–ReLU], first stride 2")
    if k < 4:
        arrow(ex, y - bh / 2, ex, ys[k + 1] + bh / 2 + 0.02)
        arrow(tx, yt - 0.15, tx, ys[k + 1] - 0.31 + 0.17, color=TH_E, bridge=True)
    # FiLM node
    bg.add_patch(Circle((fx, y), 0.13, fc="white", ec=TH_E, lw=1.0, zorder=4))
    bg.text(fx, y, "FiLM", ha="center", va="center", fontsize=4.8, fontweight="bold", color=TH_E, zorder=5)
    arrow(ex + ew / 2, y, fx - 0.14, y)
    arrow(tx + tw / 2, yt, fx - 0.06, y - 0.12, color=TH_E, cs="angle3,angleA=0,angleB=70")
    bg.text(tx + tw / 2 + 0.08, yt + 0.10, "γ, β", fontsize=5.3, color=TH_E, ha="left", va="bottom", zorder=5)

# decoder chain
for k, (s, f) in enumerate(dec):
    block(dx, ys[k], dw, bh, DEC, DEC_E, f"Decoder {4 - k}: {s} × {s} × {f}", "up ×2 → concat → 2×[Conv3×3–BN–ReLU]")
    arrow(fx + 0.14, ys[k], dx - dw / 2 - 0.02, ys[k], color=SKIP)
for k in range(3, 0, -1):
    arrow(dx, ys[k] + bh / 2, dx, ys[k - 1] - bh / 2 - 0.02)
arrow(fx + 0.14, ys[4], dx, ys[3] - bh / 2 - 0.02, cs="angle,angleA=0,angleB=-90")
block(dx, 4.90, dw, 0.36, HEAD, HEAD_E, "Head: damage probability 320 × 320 × 1", "up ×2 → Conv3×3 (32) → Conv1×1 + sigmoid")
arrow(dx, ys[0] + bh / 2, dx, 4.90 - 0.20)

# titles
bg.text(ex, 0.70, "RGB encoder\nMobileNetV2 (frozen)", ha="center", va="top", fontsize=6.3, fontweight="bold", color=ENC_E)
bg.text(tx, 0.70, "Thermal branch\n(trained from scratch)", ha="center", va="top", fontsize=6.3, fontweight="bold", color=TH_E)
bg.text(fx, 0.70, "Fusion at\n5 scales", ha="center", va="top", fontsize=6.3, fontweight="bold", color=TH_E)
bg.text(dx, 0.70, "Decoder (as in RGB model)", ha="center", va="top", fontsize=6.3, fontweight="bold", color=DEC_E)

# equation inset
bx, by, bw2, bh2 = 4.95, 0.08, 2.40, 0.36
bg.add_patch(FancyBboxPatch((bx, by), bw2, bh2, boxstyle="round,pad=0,rounding_size=0.04", fc="#fff8f1", ec=TH_E, lw=0.7))
bg.text(bx + bw2 / 2, by + 0.235, r"$F'_s = F_s \odot (1 + \gamma_s(T_s)) + \beta_s(T_s)$", ha="center", va="center", fontsize=7)
bg.text(bx + bw2 / 2, by + 0.085, "γ, β: 1×1 convolutions, zero-initialised", ha="center", va="center", fontsize=5.4, color=MUTED)

os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig_fusion.pdf"))
fig.savefig(os.path.join(out, "fig_fusion.png"), dpi=600)
print("ok")
