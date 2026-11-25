"""Radiometric thermal for FLIR E8 files and registration onto the 640x480 visual (RGB) grid.

Apparent temperature: FLIR Planck equation with the file's own calibration constants
(T = B / ln(R1 / (R2 (raw + O)) + F) - 273.15), emissivity 1 (blackbody-equivalent),
no atmospheric correction. Reproduces the project's earlier validated maps (max |diff| 2e-6 C).

Registration: the thermal frame covers the central part of the visual frame. Camera PiP metadata
(record 42): Real2IR = 1.3123, offset (-9, +5) visual px -> thermal footprint width 640/Real2IR
= 487.7 px, centred at (320-9, 240+5). Confirmed by an independent search maximising the
leaf-mask vs temperature correlation on 24 CV images (optimum width 492 px, offset (-10, +10)).
Residual parallax (object-distance dependent) of a few pixels remains.
"""
import struct

import cv2
import numpy as np

from faw import flir

H, W = 480, 640


def camera_params(recs):
    ci = recs[flir.REC_CAMINFO]
    f = lambda o: struct.unpack("<f", ci[o:o + 4])[0]  # noqa: E731
    p = {"R1": f(0x58), "B": f(0x5C), "F": f(0x60), "R2": f(0x30C), "O": struct.unpack("<i", ci[0x308:0x30C])[0]}
    pip = recs.get(42)
    if pip is not None:
        p["real2ir"] = struct.unpack("<f", pip[0:4])[0]
        p["offx"], p["offy"] = struct.unpack("<hh", pip[4:8])
    return p


def apparent_temperature(path):
    return apparent_temperature_from_records(flir.read_records(path))


def apparent_temperature_from_records(recs):
    raw = flir.raw_thermal(recs)
    if raw is None:
        return None, None
    p = camera_params(recs)
    t = p["B"] / np.log(p["R1"] / (p["R2"] * (raw.astype(np.float64) + p["O"])) + p["F"]) - 273.15
    return t.astype(np.float32), p


def to_rgb_grid(temp, real2ir=1.3123, offx=-9, offy=5):
    """Warp a 240x320 thermal map onto the 480x640 RGB grid. Returns (temp_on_rgb, valid_mask)."""
    fw = W / real2ir
    fh = H / real2ir
    cx, cy = W / 2 + offx, H / 2 + offy
    sx, sy = fw / temp.shape[1], fh / temp.shape[0]
    M = np.float32([[sx, 0, cx - fw / 2], [0, sy, cy - fh / 2]])
    t = cv2.warpAffine(temp, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    valid = np.isfinite(t)
    return t, valid
