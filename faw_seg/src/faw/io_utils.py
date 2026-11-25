import re
from datetime import datetime

import numpy as np
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png")


def list_images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def parse_num(name):
    m = re.findall(r"(\d+)", name)
    return int(m[-1]) if m else None


def exif_time(path):
    with Image.open(path) as im:
        e = im.getexif()
        t = e.get_ifd(0x8769).get(36867) or e.get(306)
    return datetime.strptime(t, "%Y:%m:%d %H:%M:%S") if t else None


def dhash(img, size=8):
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.float32)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def ncc(img_a, img_b, size=(160, 120)):
    a = np.asarray(img_a.convert("L").resize(size, Image.BILINEAR), dtype=np.float64)
    b = np.asarray(img_b.convert("L").resize(size, Image.BILINEAR), dtype=np.float64)
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    return float((a * b).mean())
