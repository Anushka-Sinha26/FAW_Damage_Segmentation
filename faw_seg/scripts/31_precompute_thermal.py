"""Step 31 (Phase 3) - Precompute registered thermal maps on the RGB grid for every image.
Output: outputs/thermal_aligned/<sample_id>.npy  (float16, 480x640, apparent temperature in C,
NaN outside the thermal footprint or when the file has no radiometric data) + index CSV.
Pairing: RGB<->FLIR by EXIF capture time (splits.csv flir_path). Registration: faw.thermal.
Usage (project root): python faw_seg\\scripts\\31_precompute_thermal.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

FAW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAW / "src"))
from faw.config import load_config  # noqa: E402
from faw import data, thermal  # noqa: E402


def main():
    cfg = load_config()
    K = cfg["splits"]["n_folds"]
    out = cfg["p"]["outputs"] / "thermal_aligned"
    out.mkdir(exist_ok=True)
    recs = data.load_records(cfg, [f"fold{k}" for k in range(K)] + ["test"])
    rows = []
    for i, r in enumerate(recs, 1):
        t, p = (None, None)
        if r["flir"] is not None and r["flir"].exists():
            t, p = thermal.apparent_temperature(r["flir"])
        if t is None:
            tr = np.full((480, 640), np.nan, np.float32)
            status = "no_radiometric_data"
        else:
            tr, _ = thermal.to_rgb_grid(t, p.get("real2ir", 1.3123), p.get("offx", -9), p.get("offy", 5))
            status = "ok"
        np.save(out / f"{r['sample_id']}.npy", tr.astype(np.float16))
        v = np.isfinite(tr)
        rows.append({"sample_id": r["sample_id"], "class": r["class"], "split": r["split"], "flir": str(r["flir"]),
                     "status": status, "valid_pct": round(100 * v.mean(), 2),
                     "t_min": float(np.nanmin(tr)) if v.any() else None, "t_max": float(np.nanmax(tr)) if v.any() else None})
        if i % 100 == 0:
            print(f"{i}/{len(recs)}")
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    bad = [r["sample_id"] for r in rows if r["status"] != "ok"]
    print(f"Done: {len(rows)} maps, {len(bad)} without radiometric data: {bad}")


if __name__ == "__main__":
    main()
