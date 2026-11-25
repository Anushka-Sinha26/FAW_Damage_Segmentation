"""Minimal reader for FLIR E-series radiometric JPEGs (FFF records inside APP1 segments)."""
import io
import struct

import numpy as np
from PIL import Image

REC_RAW = 1
REC_EMBEDDED = 14
REC_CAMINFO = 32


def _fff_bytes(data):
    i, chunks = 2, {}
    while i < len(data) - 4 and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xDA:
            break
        seg_len = int.from_bytes(data[i + 2:i + 4], "big")
        seg = data[i + 4:i + 2 + seg_len]
        if marker == 0xE1 and seg[:5] == b"FLIR\x00":
            chunks[seg[6]] = seg[8:]
        i += 2 + seg_len
    return b"".join(chunks[k] for k in sorted(chunks))


def read_records(path):
    with open(path, "rb") as f:
        return read_records_bytes(f.read())


def read_records_bytes(data):
    fff = _fff_bytes(data)
    if fff[:4] != b"FFF\x00":
        return {}
    dir_off, n = struct.unpack(">II", fff[0x18:0x20])
    recs = {}
    for k in range(n):
        e = fff[dir_off + 32 * k: dir_off + 32 * k + 20]
        rtype, _, _, _, off, length = struct.unpack(">HHIIII", e)
        if rtype:
            recs[rtype] = fff[off:off + length]
    return recs


def embedded_visual(recs):
    r = recs.get(REC_EMBEDDED)
    if r is None:
        return None
    try:
        im = Image.open(io.BytesIO(r[0x20:]))
        im.load()
        return im.convert("RGB")
    except Exception:
        return None


def raw_thermal(recs):
    r = recs.get(REC_RAW)
    if r is None:
        return None
    w, h = struct.unpack("<HH", r[2:6])
    payload = r[0x20:]
    if payload[:4] == b"\x89PNG":
        a = np.array(Image.open(io.BytesIO(payload))).astype(np.uint16)
        a = a.byteswap()
    else:
        a = np.frombuffer(payload[:w * h * 2], dtype="<u2").reshape(h, w)
    return a
