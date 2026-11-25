"""Figure 4 -- annotation examples (real Labelme polygons, annotation v2).
Usage: python make_fig04.py <fig04_candidates.npz> <out_dir>
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

d = np.load(sys.argv[1]); out = sys.argv[2]
meta = json.loads(str(d["meta"]))
T = {m[1]: m for m in meta if m[0] == "T"}
C = {m[1]: m for m in meta if m[0] == "C"}
TYP = "D051"
CROPS = [("C47", "Shot-hole"), ("C209", "Window-pane (scraped) patch"), ("C150", "Irregular, ragged hole")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
W = 7.48
pw = 2.30; ph = pw * 0.75; gx = 0.14; left = (W - 3 * pw - 2 * gx) / 2
cs = 1.72; gy = 0.42; top = 0.28; bot = 0.30
H = top + ph + gy + cs + bot
fig = plt.figure(figsize=(W, H))
RED, VTX = "#e53e3e", "#ffd400"


def panel(x, y, w, h, a, title, letter, cmap=None, polys=(), vertices=False):
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.imshow(a, cmap=cmap, interpolation="nearest" if a.dtype == bool else "antialiased")
    for p in polys:
        p = np.asarray(p)
        ax.add_patch(Polygon(p, closed=True, fill=False, ec=RED, lw=0.8))
        if vertices:
            ax.plot(p[:, 0], p[:, 1], "o", ms=1.6, mfc=VTX, mec="black", mew=0.25)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_edgecolor("#7b8794")
    ax.text(0.02, 0.97, f"({letter})", transform=ax.transAxes, ha="left", va="top", fontsize=7, fontweight="bold",
            color="white", bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.55))
    ax.set_title(title, fontsize=7, fontweight="bold", pad=3)
    return ax


rgb = d["T_" + TYP]; m = d["TM_" + TYP]; P = T[TYP][2]
y1 = H - top - ph
nv = int(np.median([len(p) for p in P]))
panel(left, y1, pw, ph, rgb, f"RGB image ({TYP})", "a")
panel(left + pw + gx, y1, pw, ph, rgb, f"Labelme polygons ({len(P)}; median {nv} vertices)", "b", polys=P, vertices=True)
panel(left + 2 * (pw + gx), y1, pw, ph, m, "Binary mask (white = damage)", "c", cmap="gray")

y2 = bot
for k, (cid, name) in enumerate(CROPS):
    x = left + k * (pw + gx) + (pw - cs) / 2
    cm = C[cid]
    panel(x, y2, cs, cs, d[cid], name, "def"[k], polys=[cm[3]], vertices=True)
    fig.text((x + cs / 2) / W, (y2 - 0.12) / H, f"{cm[2]}, 160 × 160 px crop", ha="center", va="center", fontsize=6, color="#52606d")

os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig04_annotation.pdf"))
fig.savefig(os.path.join(out, "fig04_annotation.png"), dpi=600)
print("ok", W, H)
