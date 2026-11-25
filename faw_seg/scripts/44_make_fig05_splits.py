"""Figure 5 -- capture-block-grouped data split (from outputs/splits/splits.csv).
Usage: python make_fig05.py <splits.csv> <out_dir>
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Arc

t = pd.read_csv(sys.argv[1]); out = sys.argv[2]
t["time"] = pd.to_datetime(t["rgb_time"]); t = t.sort_values("time").reset_index(drop=True)
t["day"] = t["time"].dt.strftime("%-d %b %Y")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
FOLD = {"fold0": "#3b6ea8", "fold1": "#2a9d8f", "fold2": "#7fb069", "fold3": "#e0a537", "fold4": "#8e6bb0"}
LIGHT = {k: matplotlib.colors.to_hex(0.35 * np.array(matplotlib.colors.to_rgb(v)) + 0.65) for k, v in FOLD.items()}
TEST = "#8a939c"; INK = "#1f2933"; MUTED = "#52606d"

W, H = 7.48, 3.55
fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")

# ---------------- (a) image strips by capture day
bg.text(0.10, H - 0.12, "(a)", fontsize=8, fontweight="bold", va="top")
bg.text(0.38, H - 0.12, "All 633 images in capture order, coloured by split; white gaps separate capture blocks",
        fontsize=7, va="top", color=INK)
x0, x1 = 1.55, W - 0.15; cw = (x1 - x0) / 325; sh = 0.26
days = [d for d in t["day"].unique()]
for r, day in enumerate(days):
    g = t[t["day"] == day].reset_index(drop=True)
    y = H - 0.62 - r * 0.42
    cls = g["class"].iloc[0]
    bg.text(x0 - 0.08, y + sh / 2 + 0.05, day, ha="right", va="center", fontsize=6.8, fontweight="bold")
    bg.text(x0 - 0.08, y + sh / 2 - 0.08, f"{cls}, n = {len(g)}", ha="right", va="center", fontsize=6, color=MUTED)
    for i, row in g.iterrows():
        c = TEST if row["split"] == "test" else FOLD[row["split"]]
        bg.add_patch(Rectangle((x0 + i * cw, y), cw, sh, fc=c, ec="none"))
    # block boundaries
    b = np.where(g["group"].values[1:] != g["group"].values[:-1])[0] + 1
    for i in b:
        bg.add_line(plt.Line2D([x0 + i * cw] * 2, [y, y + sh], color="white", lw=0.35))

# legend
ly = H - 1.72
items = [(f"Fold {k + 1}", FOLD[f"fold{k}"]) for k in range(5)] + [("Held-out test", TEST)]
counts = t.groupby("split").size()
lx = x0
for k, (lab, c) in enumerate(items):
    key = f"fold{k}" if k < 5 else "test"
    bg.add_patch(Rectangle((lx, ly), 0.16, 0.11, fc=c, ec="none"))
    bg.text(lx + 0.21, ly + 0.055, f"{lab} ({counts[key]})", va="center", fontsize=6.3)
    lx += 0.95 if k < 5 else 0

# ---------------- (b) CV runs grid
bg.text(0.10, H - 1.98, "(b)", fontsize=8, fontweight="bold", va="top")
bg.text(0.38, H - 1.98, "Five cross-validation runs (rows) over the same five folds (columns)", fontsize=7, va="top")
gx0, gy0 = 1.00, 0.18; cwid, ch = 0.50, 0.20
for k in range(5):
    bg.text(gx0 + k * cwid + cwid / 2, gy0 + 5 * ch + 0.12, f"Fold {k + 1}", ha="center", fontsize=6.3)
for r in range(5):
    y = gy0 + (4 - r) * ch
    bg.text(gx0 - 0.08, y + ch / 2, f"Run {r + 1}", ha="right", va="center", fontsize=6.3)
    for k in range(5):
        f = f"fold{k}"
        fc = FOLD[f] if k == r else LIGHT[f]
        bg.add_patch(Rectangle((gx0 + k * cwid + 0.02, y + 0.02), cwid - 0.04, ch - 0.04, fc=fc, ec="none"))
        if k == r:
            bg.text(gx0 + k * cwid + cwid / 2, y + ch / 2, "val.", ha="center", va="center", fontsize=5.8, color="white", fontweight="bold")
lx = gx0 + 5 * cwid + 0.25
bg.add_patch(Rectangle((lx, gy0 + 0.86), 0.16, 0.11, fc=FOLD["fold0"], ec="none"))
bg.text(lx + 0.22, gy0 + 0.915, "Validation fold: early stopping,", va="center", fontsize=6.3)
bg.text(lx + 0.22, gy0 + 0.78, "checkpoint choice, OOF predictions", va="center", fontsize=6.3)
bg.add_patch(Rectangle((lx, gy0 + 0.52), 0.16, 0.11, fc=LIGHT["fold0"], ec="none"))
bg.text(lx + 0.22, gy0 + 0.575, "Training folds: weight updates", va="center", fontsize=6.3)
bg.text(lx, gy0 + 0.30, "Pooled OOF (272 damaged + 262 healthy):", fontsize=6.3, va="center", color=MUTED)
bg.text(lx, gy0 + 0.16, "threshold / minimum-region selection,", fontsize=6.3, va="center", color=MUTED)
bg.text(lx, gy0 + 0.02, "paired model comparison", fontsize=6.3, va="center", color=MUTED)
tx, ty, tw, th = W - 1.45, gy0 - 0.05, 1.30, 1.10
bg.add_patch(FancyBboxPatch((tx, ty), tw, th, boxstyle="round,pad=0,rounding_size=0.05", fc="#eef0f2", ec=TEST, lw=0.8))
bg.add_patch(Rectangle((tx + tw / 2 - 0.09, ty + th - 0.38), 0.18, 0.15, fc=INK, ec="none"))
bg.add_patch(Arc((tx + tw / 2, ty + th - 0.23), 0.12, 0.18, theta1=0, theta2=180, lw=1.4, color=INK))
nd=(t[t.split=='test']['class']=='damaged').sum(); nh=(t[t.split=='test']['class']=='healthy').sum()
bg.text(tx + tw / 2, ty + 0.48, "Held-out test", ha="center", fontsize=6.8, fontweight="bold")
bg.text(tx + tw / 2, ty + 0.32, f"{nd} damaged + {nh} healthy", ha="center", fontsize=6.2, color=MUTED)
bg.text(tx + tw / 2, ty + 0.17, "evaluated once,", ha="center", fontsize=6.2, color=MUTED)
bg.text(tx + tw / 2, ty + 0.05, "after all 5 runs", ha="center", fontsize=6.2, color=MUTED)

os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig05_splits.pdf"))
fig.savefig(os.path.join(out, "fig05_splits.png"), dpi=600)
print("ok", t.group.nunique())
