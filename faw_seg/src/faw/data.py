"""Data loading for cross-validated training.

Splits come from outputs/splits/splits.csv (step 02, frozen). Masks are read from
cfg paths.masks (annotation v2). Healthy images get an all-zero mask.
The 'test' split is never loaded here unless explicitly requested.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def load_records(cfg, splits):
    df = pd.read_csv(cfg["p"]["outputs"] / "splits" / "splits.csv")
    df = df[df["split"].isin(splits)]
    recs = []
    for r in df.to_dict("records"):
        dmg = r["class"] == "damaged"
        recs.append({
            "sample_id": r["sample_id"], "class": r["class"], "split": r["split"],
            "img": cfg["root"] / r["rgb_path"],
            "mask": (cfg["p"]["masks"] / (Path(r["rgb_path"]).stem + ".png")) if dmg else None,
            "flir": (cfg["root"] / r["flir_path"]) if isinstance(r.get("flir_path"), str) and r["flir_path"] else None,
        })
    return recs


def read_rgb(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if a.shape != (480, 640, 3):
        raise ValueError(f"Unexpected image size {a.shape}: {path}")
    return a


def read_mask(path):
    m = np.asarray(Image.open(path).convert("L"))
    if m.shape != (480, 640):
        raise ValueError(f"Unexpected mask size {m.shape}: {path}")
    if not set(np.unique(m).tolist()) <= {0, 255}:
        raise ValueError(f"Non-binary mask: {path}")
    return (m > 127).astype(np.uint8)


def thermal_input(t16):
    """2-channel thermal input: per-image z-scored apparent temperature (0 outside footprint), valid mask."""
    t = t16.astype(np.float32)
    v = np.isfinite(t)
    out = np.zeros((480, 640, 2), np.float32)
    if v.sum() > 100:
        out[..., 0][v] = np.clip((t[v] - t[v].mean()) / (t[v].std() + 1e-3), -5, 5)
        out[..., 1] = v
    return out.astype(np.float16)


def load_thermal(cfg, recs):
    d = cfg["p"]["outputs"] / "thermal_aligned"
    return [thermal_input(np.load(d / f"{r['sample_id']}.npy")) for r in recs]


def load_arrays(recs):
    imgs = [read_rgb(r["img"]) for r in recs]
    masks = [read_mask(r["mask"]) if r["mask"] is not None else np.zeros((480, 640), np.uint8) for r in recs]
    return imgs, masks


def make_train_dataset(tf, d_imgs, d_masks, h_imgs, cfg, seed, d_th=None, h_th=None):
    T, A = cfg["train"], cfg["train"]["aug"]
    P = T["patch"]
    hf = T["healthy_fraction"] if h_imgs else 0.0
    zero = np.zeros((480, 640, 1), np.uint8)

    def gen():
        rng = np.random.default_rng(seed)
        while True:
            if hf and rng.random() < hf:
                yield h_imgs[rng.integers(len(h_imgs))], zero
            else:
                i = rng.integers(len(d_imgs))
                yield d_imgs[i], d_masks[i][..., None]

    sig = (tf.TensorSpec((480, 640, 3), tf.uint8), tf.TensorSpec((480, 640, 1), tf.uint8))
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)

    if d_th is not None:
        if A.get("mode", "extended") != "exp06":
            raise ValueError("Thermal fusion training is implemented for the exp06 augmentation recipe only")

        def gen_t():
            rng = np.random.default_rng(seed)
            while True:
                if hf and rng.random() < hf:
                    j = rng.integers(len(h_imgs))
                    yield h_imgs[j], h_th[j], zero
                else:
                    i = rng.integers(len(d_imgs))
                    yield d_imgs[i], d_th[i], d_masks[i][..., None]

        sig_t = (tf.TensorSpec((480, 640, 3), tf.uint8), tf.TensorSpec((480, 640, 2), tf.float16),
                 tf.TensorSpec((480, 640, 1), tf.uint8))

        def aug_t(img_u8, th, m_u8):
            img = tf.image.convert_image_dtype(img_u8, tf.float32)
            x = tf.concat([img, tf.cast(th, tf.float32), tf.cast(m_u8, tf.float32)], -1)
            x = tf.image.random_crop(x, [P, P, 6])
            u = tf.random.uniform([4])
            x = tf.cond(u[0] < 0.5, lambda: tf.image.flip_left_right(x), lambda: x)
            x = tf.cond(u[1] < 0.2, lambda: tf.image.flip_up_down(x), lambda: x)
            k = tf.random.uniform([], 1, 4, dtype=tf.int32)
            x = tf.cond(u[2] < 0.5, lambda: tf.image.rot90(x, k), lambda: x)
            img, t2, m = x[..., :3], x[..., 3:5], x[..., 5:]
            v = tf.random.uniform([2])
            img = tf.cond(v[0] < 0.4, lambda: tf.image.random_brightness(img, 0.08), lambda: img)
            img = tf.cond(v[1] < 0.4, lambda: tf.image.random_contrast(img, 0.90, 1.10), lambda: img)
            return (tf.clip_by_value(img, 0.0, 1.0), t2), tf.cast(m > 0.5, tf.float32)

        dst = tf.data.Dataset.from_generator(gen_t, output_signature=sig_t)
        dst = dst.map(aug_t, num_parallel_calls=tf.data.AUTOTUNE).batch(T["batch_size"]).prefetch(tf.data.AUTOTUNE)
        n_crops = len(d_imgs) * T["crops_per_damaged_image"] / max(1e-6, 1.0 - hf)
        return dst, int(np.ceil(n_crops / T["batch_size"]))

    def aug_exp06(img_u8, m_u8):
        """Phase 1 Exp 06 recipe (src/phase1/dataset_patches.py): random 320 crop at native scale,
        hflip p0.5, vflip p0.2, rot90 (k=1..3) p0.5, brightness +-0.08 p0.4, contrast 0.9-1.1 p0.4."""
        img = tf.image.convert_image_dtype(img_u8, tf.float32)
        m = tf.cast(m_u8, tf.float32)
        x = tf.image.random_crop(tf.concat([img, m], -1), [P, P, 4])
        u = tf.random.uniform([4])
        x = tf.cond(u[0] < 0.5, lambda: tf.image.flip_left_right(x), lambda: x)
        x = tf.cond(u[1] < 0.2, lambda: tf.image.flip_up_down(x), lambda: x)
        k = tf.random.uniform([], 1, 4, dtype=tf.int32)
        x = tf.cond(u[2] < 0.5, lambda: tf.image.rot90(x, k), lambda: x)
        img, m = x[..., :3], x[..., 3:]
        v = tf.random.uniform([2])
        img = tf.cond(v[0] < 0.4, lambda: tf.image.random_brightness(img, 0.08), lambda: img)
        img = tf.cond(v[1] < 0.4, lambda: tf.image.random_contrast(img, 0.90, 1.10), lambda: img)
        return tf.clip_by_value(img, 0.0, 1.0), tf.cast(m > 0.5, tf.float32)

    def aug(img_u8, m_u8):
        img = tf.image.convert_image_dtype(img_u8, tf.float32)
        m = tf.cast(m_u8, tf.float32)
        # Random scale = crop a (P/s x P/s) window, then resize it to P x P.
        # Same effect as resizing the whole image first, but ~4x less work per sample.
        s = tf.random.uniform([], A["scale"][0], A["scale"][1])
        c = tf.minimum(tf.cast(tf.round(P / s), tf.int32), 480)
        x = tf.image.random_crop(tf.concat([img, m], -1), [c, c, 4])
        x = tf.concat([tf.image.resize(x[..., :3], [P, P], method="bilinear"),
                       tf.image.resize(x[..., 3:], [P, P], method="nearest")], -1)
        if A["hflip"]:
            x = tf.image.random_flip_left_right(x)
        if A["vflip"]:
            x = tf.image.random_flip_up_down(x)
        img, m = x[..., :3], x[..., 3:]
        img = tf.image.random_brightness(img, A["brightness"])
        img = tf.image.random_contrast(img, A["contrast"][0], A["contrast"][1])
        img = tf.image.random_saturation(img, A["saturation"][0], A["saturation"][1])
        img = tf.image.random_hue(img, A["hue"])
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, tf.cast(m > 0.5, tf.float32)

    fn = aug_exp06 if A.get("mode", "extended") == "exp06" else aug
    ds = ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE).batch(T["batch_size"]).prefetch(tf.data.AUTOTUNE)
    n_crops = len(d_imgs) * T["crops_per_damaged_image"] / max(1e-6, 1.0 - hf)
    steps = int(np.ceil(n_crops / T["batch_size"]))
    return ds, steps
