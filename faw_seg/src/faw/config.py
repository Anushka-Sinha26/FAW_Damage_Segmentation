import os
import random
from pathlib import Path

import numpy as np
import yaml

FAW_ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    import os
    path = Path(path) if path else Path(os.environ.get("FAW_CONFIG", FAW_ROOT / "configs" / "config.yaml"))
    if not path.is_absolute() and not path.exists():
        path = FAW_ROOT / "configs" / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = (FAW_ROOT / cfg["project_root"]).resolve()
    cfg["root"] = root
    cfg["p"] = {k: (root / v).resolve() for k, v in cfg["paths"].items()}
    return cfg


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
