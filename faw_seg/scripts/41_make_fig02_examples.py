"""Figure 2 -- example images (RGB, ground-truth overlay, registered thermal, binary mask).
Usage: python make_fig02.py <fig02_candidates.npz> <out_dir>
"""
import sys, os
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(sys.argv[1]); out = sys.argv[2]
rows = [("D177", "Small shot-holes"), ("D083", "Large ragged damage"), ("H200", "Healthy leaf")]
cols = ["RGB image", "Ground truth (red) on RGB", "Registered thermal", "Binary mask"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})

W = 7.48
pw = 1.60; ph = pw * 0.75; gx = 0.10; gy = 0.30; left = 0.42; top = 0.30; cbw = 0.07
H = top + 3 * ph + 2 * gy + 0.25
fig = plt.figure(figsize=(W, H))
letters = "abcdefghijkl"
inferno = plt.get_cmap("inferno").copy(); inferno.set_bad("#d9dde2")

for r, (sid, name) in enumerate(rows):
    rgb = d[sid + "_rgb"]; m = d[sid + "_m"]; t = d[sid + "_t"].astype(np.float32)
    y = H - top - (r + 1) * ph - r * gy
    ov = rgb.astype(np.float32) / 255
    ov[m] = 0.35 * ov[m] + 0.65 * np.array([0.92, 0.10, 0.10])
    cs = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    panels = [(rgb, None), (ov, None), (np.ma.masked_invalid(t), inferno), (m, "gray")]
    for c, (a, cmap) in enumerate(panels):
        x = left + c * (pw + gx) + (cbw + 0.28 if c == 3 else 0)
        ax = fig.add_axes([x / W, y / H, pw / W, ph / H])
        im = ax.imshow(a, cmap=cmap, interpolation="antialiased" if c < 3 else "nearest")
        if c == 1:
            for cc in cs:
                cc = cc[:, 0, :]
                ax.plot(np.r_[cc[:, 0], cc[0, 0]], np.r_[cc[:, 1], cc[0, 1]], color="#ffd400", lw=0.35)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_edgecolor("#7b8794")
        ax.text(0.02, 0.97, f"({letters[r * 4 + c]})", transform=ax.transAxes, ha="left", va="top",
                fontsize=7, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.55))
        if r == 0:
            ax.set_title(cols[c], fontsize=7, fontweight="bold", pad=3)
        if c == 2:
            cax = fig.add_axes([(x + pw + 0.04) / W, y / H, cbw / W, ph / H])
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=5.5, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
            cb.set_label("apparent °C", fontsize=5.5, labelpad=1)
    pct = 100 * m.mean()
    sub = f"{sid}: damage {pct:.2f}% of image" if pct > 0 else f"{sid}: empty mask"
    fig.text((left - 0.12) / W, (y + ph / 2) / H, f"{name}\n{sub}", rotation=90, ha="center", va="center", fontsize=6.5)

os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig02_examples.pdf"))
fig.savefig(os.path.join(out, "fig02_examples.png"), dpi=600)
print("ok", W, H)
