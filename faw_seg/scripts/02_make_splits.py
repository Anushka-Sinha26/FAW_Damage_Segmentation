"""Step 02 - Leakage-safe, stratified, grouped data splits.

Images are grouped into temporally contiguous capture blocks (plus near-duplicate
links) so repeated views of the same plant never cross partitions. Damaged images
are stratified by damage-fraction tertile. One held-out test set + K CV folds.
Output: faw_seg/outputs/splits/splits.csv (one row per image, column 'split'
= test | fold0..fold{K-1}).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from faw.config import load_config  # noqa: E402


def make_blocks(g, gap_s, max_size):
    g = g.sort_values("rgb_time").copy()
    gap = g["rgb_time"].diff().dt.total_seconds().fillna(np.inf)
    block, size, ids = -1, 0, []
    for d in gap:
        if d > gap_s or size >= max_size:
            block, size = block + 1, 0
        ids.append(block)
        size += 1
    g["block"] = ids
    return g


def merge_near_dups(df, dups):
    parent = {b: b for b in df["group"].unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    gid = dict(zip(df["sample_id"], df["group"]))
    for a, b in dups[["a", "b"]].itertuples(index=False):
        if a in gid and b in gid:
            parent[find(gid[a])] = find(gid[b])
    df["group"] = df["group"].map(find)
    return df


def assign(groups, test_frac, k, rng):
    """groups: DataFrame [group, n, stratum]. Returns dict group->split."""
    out = {}
    for _, s in groups.groupby("stratum"):
        s = s.sample(frac=1, random_state=rng.integers(1e9))
        target, taken = test_frac * s["n"].sum(), 0
        rest = []
        for g, n in s[["group", "n"]].itertuples(index=False):
            if taken + n / 2 <= target:
                out[g], taken = "test", taken + n
            else:
                rest.append((g, n))
        counts = np.zeros(k)
        for g, n in sorted(rest, key=lambda x: -x[1]):
            f = int(np.argmin(counts + rng.random(k) * 1e-3))
            out[g], counts[f] = f"fold{f}", counts[f] + n
    return out


def main():
    cfg = load_config()
    S = cfg["splits"]
    rng = np.random.default_rng(cfg["seed"])
    audit = cfg["p"]["outputs"] / "audit"
    df = pd.read_csv(audit / "inventory.csv", parse_dates=["rgb_time"])
    df = df[df["rgb_path"].notna() & (df["rgb_path"] != "")].copy()
    dups = pd.read_csv(audit / "near_duplicates.csv")

    parts = []
    for cls, g in df.groupby("class"):
        g = make_blocks(g, S["block_gap_seconds"], S["max_block_size"])
        g["group"] = cls[0].upper() + "_b" + g["block"].astype(str).str.zfill(3)
        parts.append(g)
    df = merge_near_dups(pd.concat(parts), dups)

    df["stratum"] = "healthy"
    d = df["class"] == "damaged"
    gmean = df[d].groupby("group")["damage_frac"].transform("mean")
    edges = np.quantile(df.loc[d].groupby("group")["damage_frac"].mean(), np.linspace(0, 1, S["strata_bins"] + 1))
    df.loc[d, "stratum"] = "dmg_q" + pd.Series(
        np.clip(np.searchsorted(edges, gmean, side="right") - 1, 0, S["strata_bins"] - 1),
        index=gmean.index).astype(str)

    groups = df.groupby("group").agg(n=("sample_id", "size"), stratum=("stratum", "first")).reset_index()
    mapping = assign(groups, S["test_fraction"], S["n_folds"], rng)
    df["split"] = df["group"].map(mapping)

    out = cfg["p"]["outputs"] / "splits"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["sample_id", "class", "rgb_path", "flir_path", "mask_path", "rgb_time",
            "damage_frac", "group", "stratum", "split"]
    df[cols].sort_values(["class", "sample_id"]).to_csv(out / "splits.csv", index=False)

    assert df["split"].notna().all()
    assert df.groupby("group")["split"].nunique().max() == 1, "group crosses splits"

    tab = pd.crosstab(df["split"], df["class"])
    dstats = df[d].groupby("split")["damage_frac"].agg(["mean", "median"]).mul(100).round(2)
    rep = {
        "n_groups": {c: int(df[df["class"] == c]["group"].nunique()) for c in ["damaged", "healthy"]},
        "group_size": df.groupby("group").size().describe()[["min", "50%", "max"]].to_dict(),
        "counts": tab.to_dict(),
        "damage_pct_by_split": dstats.to_dict(),
        "strata_by_split": pd.crosstab(df["split"], df["stratum"]).to_dict(),
    }
    with open(out / "split_report.json", "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(tab.assign(total=tab.sum(1)).to_string())
    print("\nDamage % of image by split (damaged only):\n", dstats.to_string())
    print("\nStrata by split:\n", pd.crosstab(df["split"], df["stratum"]).to_string())
    print("\nGroups:", rep["n_groups"], "group size:", rep["group_size"])
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
