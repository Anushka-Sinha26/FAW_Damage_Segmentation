"""Registration search + data for the registration figure (run from faw_seg/).
Re-estimates the thermal footprint on the RGB grid by maximising the anti-correlation between the RGB leaf
mask and apparent temperature (leaves are cooler than soil) on 24 random CV damaged images (seed 0), at
half resolution. Writes outputs/paper_figures/fig_reg_data.npz used by 47_make_fig_registration.py.
Result (2026-10-25): calibration 0.434; optimum 0.460 at width 492 px, offset (-10, +12); no registration 0.211.
"""
import sys, json
import numpy as np, pandas as pd, cv2
sys.path.insert(0, "src")
from faw.thermal import apparent_temperature, to_rgb_grid
from faw.leaf import leaf_mask

sp = pd.read_csv("outputs/splits/splits.csv")
cv = sp[(sp["class"] == "damaged") & (sp.split != "test")]
sel = cv.sample(24, random_state=0)
S = 2
L, T = [], []
for _, r in sel.iterrows():
    rgb = cv2.cvtColor(cv2.imread(f"../{r.rgb_path}"), cv2.COLOR_BGR2RGB)
    t, _ = apparent_temperature(f"../{r.flir_path}"); l, _ = leaf_mask(rgb)
    L.append(cv2.resize(l.astype(np.float32), (640 // S, 480 // S), interpolation=cv2.INTER_AREA)); T.append(t)


def warp(t, r2i, ox, oy):
    fw, fh = 640 / r2i, 480 / r2i; cx, cy = 320 + ox, 240 + oy
    M = np.float32([[fw / 320 / S, 0, (cx - fw / 2) / S], [0, fh / 240 / S, (cy - fh / 2) / S]])
    return cv2.warpAffine(t, M, (640 // S, 480 // S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


def score(r2i, ox, oy):
    cs = []
    for l, t in zip(L, T):
        w = warp(t, r2i, ox, oy); v = np.isfinite(w)
        cs.append(-np.corrcoef(l[v], w[v])[0, 1])
    return float(np.mean(cs))


widths = np.arange(440, 545, 4); wcorr = [score(640 / w, -9, 5) for w in widths]
offs = np.arange(-30, 31, 5); grid = np.array([[score(1.3123, ox, oy) for ox in offs] for oy in offs])
bw = int(widths[int(np.argmax(wcorr))])
best = max(((score(640 / w, ox, oy), w, ox, oy) for w in range(bw - 8, bw + 9, 2) for ox in range(-20, 11, 2) for oy in range(-6, 21, 2)))
stretch = float(np.mean([-np.corrcoef(l.ravel(), cv2.resize(t, (640 // S, 480 // S)).ravel())[0, 1] for l, t in zip(L, T)]))
res = dict(calib=score(1.3123, -9, 5), best=best, stretch=stretch, ids=[str(i) for i in sel.sample_id])
print(res)
out = dict(widths=widths, wcorr=np.array(wcorr), offs=offs, grid=grid, res=json.dumps(res))
for sid in ["D177", "D083", "D051"]:
    r = sp[sp.sample_id == sid].iloc[0]
    rgb = cv2.cvtColor(cv2.imread(f"../{r.rgb_path}"), cv2.COLOR_BGR2RGB)
    Tm, _ = apparent_temperature(f"../{r.flir_path}"); t, _ = to_rgb_grid(Tm)
    stem = r.rgb_path.split("/")[-1].rsplit(".", 1)[0]
    out[sid + "_rgb"] = rgb; out[sid + "_t"] = t.astype(np.float32); out[sid + "_leaf"] = leaf_mask(rgb)[0]
    out[sid + "_gt"] = cv2.imread(f"../mask_annotation/masks_binary_v2/{stem}.png", 0) > 127
np.savez_compressed("outputs/paper_figures/fig_reg_data.npz", **out)
print("saved")
