"""Step 01 - Dataset audit (read-only on the original data).

Checks: inventory, sizes, mask validity, RGB<->FLIR pairing (filename, EXIF time,
embedded visual photo correlation), radiometric availability, near-duplicates.
Outputs go to faw_seg/outputs/audit/.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faw.config import load_config  # noqa: E402
from faw import flir  # noqa: E402
from faw.io_utils import list_images, parse_num, exif_time, dhash, hamming, ncc  # noqa: E402


def index_by_num(folder):
    out = {}
    for p in list_images(folder):
        n = parse_num(p.stem)
        if n in out:
            raise ValueError(f"Duplicate id {n} in {folder}: {out[n].name}, {p.name}")
        out[n] = p
    return out


def mask_stats(path, min_px):
    m = np.array(Image.open(path))
    vals = np.unique(m)
    binary = set(vals.tolist()) <= {0, 255}
    fg = m > 127
    n, _, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    n -= 1
    sizes = st[1:, cv2.CC_STAT_AREA] if n else np.array([0])
    return {
        "mask_w": m.shape[1], "mask_h": m.shape[0], "mask_binary": binary,
        "damage_frac": float(fg.mean()), "n_components": int(n),
        "n_tiny_components": int((sizes < min_px).sum()) if n else 0,
        "min_component_px": int(sizes.min()) if n else 0,
        "median_component_px": float(np.median(sizes)) if n else 0.0,
    }


def save_sheet(rgb, vis, therm, out, title):
    tiles = [rgb.resize((320, 240))]
    tiles.append(vis.resize((320, 240)) if vis else Image.new("RGB", (320, 240)))
    if therm is not None:
        t = therm.astype(np.float32)
        t = (255 * (t - np.percentile(t, 1)) / (np.ptp(t) + 1e-6)).clip(0, 255).astype(np.uint8)
        tiles.append(Image.fromarray(t).convert("RGB").resize((320, 240)))
    else:
        tiles.append(Image.new("RGB", (320, 240)))
    sheet = Image.new("RGB", (960, 240))
    for i, t in enumerate(tiles):
        sheet.paste(t, (320 * i, 0))
    sheet.save(out / f"{title}.png")


def main():
    cfg = load_config()
    P, A = cfg["p"], cfg["audit"]
    out = P["outputs"] / "audit"
    (out / "pair_sheets").mkdir(parents=True, exist_ok=True)

    rows = []
    for cls in ["damaged", "healthy"]:
        rgb = index_by_num(P[f"rgb_{cls}"])
        fl_by_time = {}
        for fp in list_images(P[f"flir_{cls}"]):
            fl_by_time.setdefault(exif_time(fp), []).append(fp)
        masks = index_by_num(P["masks"]) if cls == "damaged" else {}
        used = set()
        for n in sorted(rgb):
            r = {"class": cls, "num": n, "sample_id": f"{cls[0].upper()}{n:03d}",
                 "rgb_path": str(rgb[n].relative_to(cfg["root"]).as_posix()),
                 "mask_path": str(masks[n].relative_to(cfg["root"]).as_posix()) if n in masks else ""}
            with Image.open(rgb[n]) as im:
                r["rgb_w"], r["rgb_h"] = im.size
                rgb_img = im.convert("RGB")
            r["rgb_time"] = exif_time(rgb[n])
            r["dhash"] = dhash(rgb_img, A["dhash_size"])
            cands = fl_by_time.get(r["rgb_time"], [])
            fp = cands[0] if len(cands) == 1 else None
            r["flir_path"] = str(fp.relative_to(cfg["root"]).as_posix()) if fp else ""
            r["pair_method"] = "exif_time" if fp else ("ambiguous" if cands else "none")
            r["same_filename"] = bool(fp) and fp.name == rgb[n].name
            if fp:
                used.add(fp)
                with Image.open(fp) as im:
                    r["flir_w"], r["flir_h"] = im.size
                r["flir_time"] = exif_time(fp)
                recs = flir.read_records(fp)
                vis = flir.embedded_visual(recs)
                raw = flir.raw_thermal(recs)
                r["flir_radiometric"] = raw is not None
                r["raw_min"] = int(raw.min()) if raw is not None else None
                r["raw_max"] = int(raw.max()) if raw is not None else None
                r["embedded_visual"] = f"{vis.size[0]}x{vis.size[1]}" if vis else ""
                if vis is not None:
                    r["ncc_rgb_vs_embedded"] = ncc(rgb_img, vis, tuple(A["ncc_size"]))
                    if r["ncc_rgb_vs_embedded"] < A["ncc_pair_threshold"]:
                        save_sheet(rgb_img, vis, raw, out / "pair_sheets",
                                   f"{r['sample_id']}_ncc{r['ncc_rgb_vs_embedded']:.2f}")
            if n in masks:
                r.update(mask_stats(masks[n], A["min_component_px"]))
                r["mask_size_matches_rgb"] = (r["mask_w"], r["mask_h"]) == (r["rgb_w"], r["rgb_h"])
            rows.append(r)
        for fp in list_images(P[f"flir_{cls}"]):
            if fp not in used:
                rows.append({"class": cls, "sample_id": f"unpaired_{fp.stem}", "rgb_path": "",
                             "flir_path": str(fp.relative_to(cfg["root"]).as_posix()), "pair_method": "none"})
        missing_masks = sorted(set(masks) - set(rgb))
        if missing_masks:
            print(f"WARNING masks without RGB image: {missing_masks}")

    df = pd.DataFrame(rows)
    df["time_match"] = df["pair_method"].eq("exif_time")
    df.to_csv(out / "inventory.csv", index=False)

    dups = []
    for (i, a), (j, b) in combinations(df[df["dhash"].notna()][["sample_id", "dhash"]].itertuples(index=False), 2):
        d = hamming(int(a), int(b))
        if d <= A["near_dup_hamming"]:
            dups.append({"a": i, "b": j, "hamming": d})
    pd.DataFrame(dups, columns=["a", "b", "hamming"]).to_csv(out / "near_duplicates.csv", index=False)

    dmg = df[df["class"] == "damaged"]
    summary = {}
    for cls, g in df.groupby("class"):
        t = g["rgb_time"].dropna().sort_values()
        gaps = t.diff().dt.total_seconds().dropna()
        summary[cls] = {
            "n_rgb": int((g["rgb_path"] != "").sum()),
            "n_flir": int((g["flir_path"] != "").sum()),
            "rgb_without_flir": g.loc[g["flir_path"] == "", "sample_id"].tolist(),
            "flir_without_rgb": g.loc[g["rgb_path"] == "", "sample_id"].tolist(),
            "rgb_sizes": (g["rgb_w"].astype("Int64").astype(str) + "x" + g["rgb_h"].astype("Int64").astype(str)).value_counts().to_dict(),
            "flir_sizes": (g["flir_w"].astype("Int64").astype(str) + "x" + g["flir_h"].astype("Int64").astype(str)).value_counts().to_dict(),
            "flir_radiometric": int(g["flir_radiometric"].sum()),
            "exif_time_match": int(g["time_match"].sum()),
            "same_filename_pairs": int(g["same_filename"].fillna(False).astype(bool).sum()),
            "ncc_median": float(g["ncc_rgb_vs_embedded"].median()),
            "ncc_below_threshold": int((g["ncc_rgb_vs_embedded"] < A["ncc_pair_threshold"]).sum()),
            "capture_dates": t.dt.date.astype(str).value_counts().sort_index().to_dict(),
            "gap_seconds_quantiles": {q: float(gaps.quantile(q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
        }
    summary["masks"] = {
        "n_masks": int((dmg["mask_path"] != "").sum()),
        "damaged_without_mask": dmg.loc[dmg["mask_path"] == "", "sample_id"].tolist(),
        "non_binary": int((~dmg["mask_binary"].astype("boolean").fillna(True)).sum()),
        "size_mismatch": int((~dmg["mask_size_matches_rgb"].astype("boolean").fillna(True)).sum()),
        "empty": int((dmg["damage_frac"] == 0).sum()),
        "damage_frac_pct": {k: float(v * 100) for k, v in dmg["damage_frac"].describe(
            percentiles=[.1, .25, .5, .75, .9]).drop("count").items()},
        "components_median": float(dmg["n_components"].median()),
        "masks_with_tiny_components": int((dmg["n_tiny_components"] > 0).sum()),
    }
    summary["near_duplicate_pairs"] = len(dups)
    with open(out / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
