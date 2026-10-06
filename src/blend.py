"""Rank-average blend over saved OOF/test predictions.

    ../.venv/bin/python src/blend.py                 # all runs in submissions/
    ../.venv/bin/python src/blend.py lgb_base cat_fe  # named subset

Blends on ranks because the models are calibrated differently and AUC depends
only on ordering. Reports the OOF AUC of every member, of the equal-weight
blend, and of a greedy forward selection that adds members only while CV
improves - so the final blend is never worse than its best single member.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
OUT = HERE / "submissions"
TARGET = "satisfaction"


def rank01(a):
    return (rankdata(a) - 1) / max(len(a) - 1, 1)


def main():
    y = pd.read_csv(DATA / "train.csv", usecols=[TARGET])[TARGET].astype(int).values
    ids = pd.read_csv(DATA / "test.csv", usecols=["id"])["id"]

    tags = sys.argv[1:] or sorted(p.stem[4:] for p in OUT.glob("oof_*.npy"))
    tags = [t for t in tags if (OUT / f"test_{t}.npy").exists()]
    if len(tags) < 2:
        raise SystemExit(f"need >=2 runs with both oof_ and test_ files, have {tags}")

    oof = {t: rank01(np.load(OUT / f"oof_{t}.npy")) for t in tags}
    tst = {t: rank01(np.load(OUT / f"test_{t}.npy")) for t in tags}
    solo = {t: roc_auc_score(y, oof[t]) for t in tags}

    print("members:")
    for t in sorted(tags, key=solo.get, reverse=True):
        print(f"  {t:22s} {solo[t]:.5f}")

    equal = np.mean([oof[t] for t in tags], axis=0)
    print(f"\nequal-weight blend of {len(tags)}: {roc_auc_score(y, equal):.5f}")

    # Greedy forward selection with replacement (a member may enter more than
    # once, which acts as integer weighting).
    chosen = [max(tags, key=solo.get)]
    best = solo[chosen[0]]
    while True:
        cands = {t: roc_auc_score(y, np.mean([oof[c] for c in chosen + [t]], axis=0))
                 for t in tags}
        t, s = max(cands.items(), key=lambda kv: kv[1])
        if s <= best + 1e-6:
            break
        chosen.append(t); best = s
    weights = pd.Series(chosen).value_counts().sort_values(ascending=False)
    print(f"\ngreedy blend: {best:.5f}")
    for t, w in weights.items():
        print(f"  {t:22s} x{w}")

    blend_test = np.mean([tst[t] for t in chosen], axis=0)
    name = OUT / f"sub_blend_cv{best:.5f}.csv"
    pd.DataFrame({"id": ids, TARGET: blend_test}).to_csv(name, index=False)
    print("\nwrote", name.name)


if __name__ == "__main__":
    main()
