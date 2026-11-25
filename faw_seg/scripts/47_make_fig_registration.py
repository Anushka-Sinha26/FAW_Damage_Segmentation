"""Registration check figure (RGB-thermal), from real CV images and a re-run registration search.
Usage: python make_fig_registration.py <fig_reg_data.npz> <out_dir>
"""
import sys, os, json
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

d = np.load(sys.argv[1]); out = sys.argv[2]
res = json.loads(str(d["res"]))
IDS = ["D177", "D083", "D051"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
cm = plt.get_cmap("inferno").copy(); cm.set_bad("#d9dde2")
FW, FH, CX, CY = 640 / 1.3123, 480 / 1.3123, 320 - 9, 240 + 5

W = 7.48
pw = 2.05; ph = pw * 0.75; gx = 0.12; left = 0.32; top = 0.22; gy = 0.12
rows_h = 3 * ph + 2 * gy
bottom_h = 1.95
H = top + rows_h + 0.30 + bottom_h
fig = plt.figure(figsize=(W, H))
cols = ["RGB + thermal footprint (dashed)", "Registered apparent temperature", "RGB leaf outline on thermal"]


def ax_at(c, r):
    x = left + c * (pw + gx) + (0.42 if c >= 2 else 0)
    y = H - top - (r + 1) * ph - r * gy
    a = fig.add_axes([x / W, y / H, pw / W, ph / H]); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_linewidth(0.4); s.set_edgecolor("#7b8794")
    return a, x, y


def contours(a, mask, color, lw):
    for c in cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
        c = c[:, 0, :]
        a.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=color, lw=lw)


letters = iter("abcdefghijklmn")
for r, sid in enumerate(IDS):
    rgb, t, leaf, gt = d[sid + "_rgb"], d[sid + "_t"], d[sid + "_leaf"], d[sid + "_gt"]
    tm = np.ma.masked_invalid(t)
    vmin, vmax = np.nanpercentile(t, 1), np.nanpercentile(t, 99)
    # (1) footprint
    a, x, y = ax_at(0, r); a.imshow(rgb)
    a.add_patch(Rectangle((CX - FW / 2, CY - FH / 2), FW, FH, fill=False, ec="#ffd400", lw=0.9, ls="--"))
    a.set_ylabel(sid, fontsize=7, fontweight="bold")
    # (2) registered temperature + colourbar
    a2, x2, y2 = ax_at(1, r); im = a2.imshow(tm, cmap=cm, vmin=vmin, vmax=vmax)
    cax = fig.add_axes([(x2 + pw + 0.04) / W, y2 / H, 0.06 / W, ph / H])
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    cb.set_label("°C", fontsize=5.5, labelpad=1)
    # (3) leaf outline (RGB-derived) on thermal, clipped to footprint
    a3, _, _ = ax_at(2, r); a3.imshow(tm, cmap=cm, vmin=vmin, vmax=vmax)
    lm = cv2.morphologyEx(leaf.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    lm = (lm > 0) & np.isfinite(t)
    for c in cv2.findContours(lm.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]:
        if cv2.contourArea(c) < 400: continue
        c = c[:, 0, :]; a3.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color="#5ce1e6", lw=0.45)
    for aa in (a, a2, a3):
        aa.text(0.02, 0.97, f"({next(letters)})", transform=aa.transAxes, ha="left", va="top", fontsize=6.5,
                fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.12", fc="black", ec="none", alpha=0.55))
    if r == 0:
        for c, aa in enumerate((a, a2, a3)):
            aa.set_title(cols[c], fontsize=6.8, fontweight="bold", pad=3)

# ---- bottom: registration search
yb = 0.42; hb = bottom_h - 0.80
ax1 = fig.add_axes([0.75 / W, yb / H, 3.0 / W, hb / H])
ax1.plot(d["widths"], d["wcorr"], color="#1f6fb2", lw=1.0, marker="o", ms=2)
ax1.axvline(FW, color="#c05621", ls="--", lw=0.9)
ax1.axhline(res["stretch"], color="#7b8794", ls=":", lw=0.9)
ax1.set_ylim(res["stretch"] - 0.03, max(d["wcorr"]) + 0.03)
ax1.text(FW - 1.5, max(d["wcorr"]) + 0.02, f"camera calibration ({FW:.1f} px)", color="#c05621", fontsize=5.8, ha="right", va="center")
ax1.text(d["widths"][-1], res["stretch"] + 0.006, "no registration (full-frame stretch)", color="#52606d", fontsize=5.8, va="bottom", ha="right")
ax1.set_xlabel("Thermal footprint width on RGB image (px)"); ax1.set_ylabel("Mean leaf–temperature\nanti-correlation")
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_title(f"({next(letters)}) Footprint width (offset at calibration)", loc="left", fontsize=6.8, fontweight="bold")

ax2 = fig.add_axes([4.75 / W, yb / H, 1.75 / W, hb / H])
offs = d["offs"]
im = ax2.imshow(d["grid"], origin="lower", cmap="viridis", extent=[offs[0] - 2.5, offs[-1] + 2.5, offs[0] - 2.5, offs[-1] + 2.5])
ax2.plot(-9, 5, marker="x", color="#ff7a1a", ms=6, mew=1.4, label="camera calibration")
b = res["best"]
ax2.plot(b[2], b[3], marker="+", color="white", ms=7, mew=1.2, label="data optimum")
ax2.set_xlabel("x offset (RGB px)"); ax2.set_ylabel("y offset (RGB px)")
ax2.legend(fontsize=5.3, loc="lower right", frameon=True, framealpha=0.8, handletextpad=0.3)
cax = fig.add_axes([(4.75 + 1.75 + 0.06) / W, yb / H, 0.07 / W, hb / H])
cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
ax2.set_title(f"({next(letters)}) Offset (width at calibration)", loc="left", fontsize=6.8, fontweight="bold")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig_registration.pdf"))
fig.savefig(os.path.join(out, "fig_registration.png"), dpi=600)
print("ok", res)
