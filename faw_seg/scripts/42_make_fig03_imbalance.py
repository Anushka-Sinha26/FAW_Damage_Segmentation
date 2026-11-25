"""Figure 3 -- class imbalance: damage area and number of damage regions per damaged image.
Usage: python make_fig03.py <fig03_mask_stats.csv> <out_dir>
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t = pd.read_csv(sys.argv[1]); out = sys.argv[2]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
BAR, EDGE, MED = "#4a7fb0", "white", "#c0392b"

fig, (a, b) = plt.subplots(1, 2, figsize=(7.48, 2.7), gridspec_kw=dict(wspace=0.28))

pct = t["pct"].values
a.hist(pct, bins=np.arange(0, 10.5, 0.5), color=BAR, edgecolor=EDGE, linewidth=0.6)
m = np.median(pct)
a.axvline(m, color=MED, ls="--", lw=1.0)
a.text(m + 0.15, a.get_ylim()[1] * 0.92, f"median {m:.2f}%", color=MED, fontsize=7, ha="left", va="top")
a.set_xlim(0, 10); a.set_xticks(range(0, 11, 2))
a.set_xlabel("Damage area (% of image pixels)"); a.set_ylabel("Number of images")
a.set_title("(a)", loc="left", fontweight="bold", fontsize=8)

reg = t["regions"].values
b.hist(reg, bins=np.arange(0.5, reg.max() + 1.5, 1), color=BAR, edgecolor=EDGE, linewidth=0.6)
mr = np.median(reg)
b.axvline(mr, color=MED, ls="--", lw=1.0)
b.text(mr + 0.4, b.get_ylim()[1] * 0.92, f"median {mr:.0f}", color=MED, fontsize=7, ha="left", va="top")
b.set_xlim(0, reg.max() + 1); b.set_xticks([1, 5, 10, 15, 20, int(reg.max())])
b.set_xlabel("Damage regions per image (8-connected)"); b.set_ylabel("Number of images")
b.set_title("(b)", loc="left", fontweight="bold", fontsize=8)

for ax in (a, b):
    ax.yaxis.grid(True, lw=0.4, color="#d9dde2"); ax.set_axisbelow(True)

fig.text(0.5, -0.09, f"n = {len(t)} damaged images", ha="center", fontsize=6.5, color="#52606d")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig03_imbalance.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(out, "fig03_imbalance.png"), dpi=600, bbox_inches="tight")
print("ok", m, mr)
