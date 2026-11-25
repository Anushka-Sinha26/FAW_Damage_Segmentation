"""Image analysis for the web app: damage segmentation, binary mask, leaf-area %, damaged regions.

Models (trained in faw_seg, see project docs):
  RGB photo         -> e00 main RGB model (5-fold ensemble), threshold/min-region from config_e00_baseline.yaml
  FLIR radiometric  -> embedded visual photo + registered thermal -> e10 RGB-thermal FiLM model (5-fold ensemble),
                       settings from config_e10_fusion_film.yaml
"""
import base64
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

FAW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAW / "src"))
from faw import flir, thermal, leaf, seg  # noqa: E402
from faw.data import thermal_input  # noqa: E402

PATCH, STRIDE = 320, 160
GRID = [["upper-left", "upper-centre", "upper-right"], ["middle-left", "centre", "middle-right"],
        ["lower-left", "lower-centre", "lower-right"]]


class InputError(ValueError):
    pass


def _positions(n, p=PATCH, s=STRIDE):
    pos = list(range(0, max(n - p, 0) + 1, s))
    if pos[-1] != n - p:
        pos.append(n - p)
    return pos


def sliding_predict(predict_fn, rgb_u8, th=None):
    """Overlap-averaged probabilities for any image >= 320x320 (same 320/160 protocol as evaluation)."""
    H, W = rgb_u8.shape[:2]
    pos = [(y, x) for y in _positions(H) for x in _positions(W)]
    rb = np.stack([rgb_u8[y:y + PATCH, x:x + PATCH] for y, x in pos]).astype(np.float32) / 255.0
    tb = np.stack([th[y:y + PATCH, x:x + PATCH] for y, x in pos]).astype(np.float32) if th is not None else None
    p = predict_fn(rb, tb)
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    for k, (y, x) in enumerate(pos):
        acc[y:y + PATCH, x:x + PATCH] += p[k]
        cnt[y:y + PATCH, x:x + PATCH] += 1
    return acc / cnt


def read_upload(data):
    """Return (rgb_u8 at working resolution, thermal_input or None, info dict, original (w, h))."""
    recs = {}
    try:
        recs = flir.read_records_bytes(data)
    except Exception:
        recs = {}
    if recs and flir.raw_thermal(recs) is not None:
        vis = flir.embedded_visual(recs)
        if vis is None:
            raise InputError("This FLIR file has thermal data but no built-in visual photo. The model needs the "
                             "colour photo, so please upload the RGB image or a FLIR file that includes it.")
        rgb = np.asarray(vis.convert("RGB"))
        t, p = thermal.apparent_temperature_from_records(recs)
        if rgb.shape[:2] != (480, 640):
            rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
        tr, _ = thermal.to_rgb_grid(t, p.get("real2ir", 1.3123), p.get("offx", -9), p.get("offy", 5))
        info = {"mode": "rgb_thermal", "source": "FLIR radiometric file (built-in photo + thermal)",
                "temp_range_c": [float(np.nanmin(t)), float(np.nanmax(t))]}
        return rgb, thermal_input(tr), info, (640, 480)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise InputError("Could not read the file as an image. Upload a JPG/PNG photo or a FLIR radiometric JPG.")
    if recs:
        raise InputError("This looks like a FLIR thermal picture without radiometric data. Upload the colour (RGB) "
                         "photo, or the original FLIR radiometric JPG from the camera.")
    rgb = np.asarray(img.convert("RGB"))
    oh, ow = rgb.shape[:2]
    if abs(ow / oh - 4 / 3) < 0.04:
        work = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
    else:
        s = 480 / min(oh, ow)
        work = cv2.resize(rgb, (max(PATCH, round(ow * s)), max(PATCH, round(oh * s))), interpolation=cv2.INTER_AREA)
    info = {"mode": "rgb", "source": "RGB photo"}
    if (ow, oh) != (640, 480):
        info["resized_note"] = (f"Image resized from {ow}x{oh} to {work.shape[1]}x{work.shape[0]} for analysis. The "
                                "model was trained on 640x480 FLIR E8 visual photos; other cameras or distances may "
                                "reduce accuracy.")
    return work, None, info, (ow, oh)


def regions(mask, leaf_area_px):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    H, W = mask.shape
    out = []
    for k in range(1, n):
        x, y, w, h, a = (int(v) for v in st[k])
        cx, cy = cen[k]
        out.append({"area_px": a, "pct_of_image": 100 * a / (H * W), "pct_of_leaf": 100 * a / max(leaf_area_px, 1),
                    "bbox": [x, y, w, h], "centroid": [float(cx), float(cy)],
                    "location": GRID[min(2, int(3 * cy / H))][min(2, int(3 * cx / W))]})
    out.sort(key=lambda r: -r["area_px"])
    for i, r in enumerate(out, 1):
        r["id"] = i
    return out


def png_b64(arr_bgr_or_gray):
    ok, buf = cv2.imencode(".png", arr_bgr_or_gray)
    return base64.b64encode(buf.tobytes()).decode()


def analyze(data, predictors, settings):
    """predictors: {'rgb': fn(rgb_batch, None), 'rgb_thermal': fn(rgb_batch, th_batch)}; settings: {mode: (thr, min_px)}."""
    rgb, th, info, orig_wh = read_upload(data)
    mode = info["mode"]
    if mode not in predictors:
        raise InputError(f"No model loaded for mode '{mode}'.")
    thr, min_px = settings[mode]
    prob = sliding_predict(predictors[mode], rgb, th)
    dmg = seg.remove_small(prob >= thr, min_px)
    lm, _ = leaf.leaf_mask(rgb)
    leaf_area = int((lm | dmg).sum())
    pct_image = 100 * dmg.mean()
    pct_leaf = leaf.damage_pct_of_leaf(dmg, lm)
    regs = regions(dmg, leaf_area)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ov = bgr.copy()
    ov[dmg] = (0.45 * ov[dmg] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
    cn, _ = cv2.findContours(dmg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ov, cn, -1, (0, 255, 255), 1)
    for r in regs[:30]:
        x, y, w, h = r["bbox"]
        cv2.rectangle(ov, (x, y), (x + w, y + h), (255, 255, 255), 1)
        cv2.putText(ov, str(r["id"]), (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    leaf_vis = bgr.copy()
    leaf_vis[~lm] = (leaf_vis[~lm] * 0.2).astype(np.uint8)
    heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    mask_work = (dmg * 255).astype(np.uint8)
    mask_orig = cv2.resize(mask_work, orig_wh, interpolation=cv2.INTER_NEAREST) if (mask_work.shape[1], mask_work.shape[0]) != orig_wh else mask_work
    out = {
        "info": info, "threshold": thr, "min_region_px": min_px, "size_analyzed": [rgb.shape[1], rgb.shape[0]],
        "damage_pct_of_image": pct_image, "damage_pct_of_leaf": pct_leaf,
        "leaf_pct_of_image": 100 * lm.mean(), "damaged_px": int(dmg.sum()), "n_regions": len(regs), "regions": regs,
        "images": {"original": png_b64(bgr), "overlay": png_b64(ov), "mask": png_b64(mask_work),
                   "mask_original_size": png_b64(mask_orig), "leaf": png_b64(leaf_vis), "probability": png_b64(heat)},
    }
    if th is not None:
        t = th[..., 0].astype(np.float32)
        tv = np.zeros_like(bgr)
        v = th[..., 1] > 0
        tn = np.clip((t + 2.5) / 5 * 255, 0, 255).astype(np.uint8)
        tv[v] = cv2.applyColorMap(tn, cv2.COLORMAP_INFERNO)[v]
        out["images"]["thermal"] = png_b64(tv)
    return out
