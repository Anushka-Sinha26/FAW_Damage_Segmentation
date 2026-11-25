"""Quick speed check (about 2 minutes): where does training time go?
  A) data pipeline alone   B) one training step on ready-made tensors
Usage (from project root): python faw_seg\\scripts\\09_speed_check.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

FAW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAW / "src"))
from faw.config import load_config  # noqa: E402
from faw import data  # noqa: E402
import tensorflow as tf  # noqa: E402

cfg = load_config()
T = cfg["train"]
print(f"CPU threads visible: {os.cpu_count()} | TF {tf.__version__} | batch {T['batch_size']}")
tr = data.load_records(cfg, ["fold1", "fold2"])
d = [r for r in tr if r["class"] == "damaged"][:40]
h = [r for r in tr if r["class"] == "healthy"][:20]
di, dm = data.load_arrays(d)
hi, _ = data.load_arrays(h)
ds, _ = data.make_train_dataset(tf, di, dm, hi, cfg, 0)
it = iter(ds)
next(it)
t = time.time()
for _ in range(30):
    xb, yb = next(it)
dt = (time.time() - t) / 30
print(f"A) data pipeline: {dt:.3f} s per batch")

sys.path.insert(0, str(cfg["root"] / cfg["paths"]["phase1_model_code"]))
import model_experiment_06_mobilenetv2_unet as M  # noqa: E402
model = M.build_mobilenetv2_unet(input_shape=(T["patch"], T["patch"], 3), encoder_trainable=False)
M.compile_model(model, learning_rate=1e-3)
model.train_on_batch(xb, yb)
t = time.time()
for _ in range(15):
    model.train_on_batch(xb, yb)
mt = (time.time() - t) / 15
print(f"B) model training step: {mt:.3f} s per batch")
print("Bottleneck:", "DATA PIPELINE" if dt > 0.7 * mt else "MODEL COMPUTE (CPU)")
