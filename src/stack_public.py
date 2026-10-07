"""Stack our members with the public OOF libraries.

Every public library used here is on the community split
StratifiedKFold(5, shuffle=True, random_state=42) over train.csv in file order
(10-fold members are still one OOF prediction per row), and we use the same
split, so all OOFs are directly comparable and stackable.

Sources (all public, all credited in PLAN.md):
  submissions/                              ours
  data/public_oof/cdeotte124/               Chris Deotte's aggregation of 124 public members
  data/public_oof/sachith7/*.npz            22 members, {oof, test} arrays
  data/public_oof/s6e10-tabpfn-member/*.csv TabPFN-3.5 / TabICL members
  data/public_oof/s6e10-tfm-oof-and-test-predictions/*.npy  tabular foundation models

Gating, because a single leaked member would look superb in CV and sink the LB:
  * exact row counts (699,635 / 299,844), finite, in [0, 1] or logit-like
  * OOF AUC within [AUC_MIN, AUC_MAX]; above the max is not credible for this
    task and is treated as leakage
  * near-duplicates (|corr| > DUP_CORR on logits) are collapsed to the best one

Stacker: logistic regression on logits, C chosen by an outer 5-fold CV over the
OOF matrix so the reported number never evaluates a fit on its own rows.
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
PUB = DATA / "public_oof"
OUT = HERE / "submissions"
TARGET = "satisfaction"
N_TRAIN, N_TEST = 699_635, 299_844
AUC_MIN, AUC_MAX = 0.940, 0.9625
DUP_CORR = 0.9995


def logit(p):
    p = np.asarray(p, dtype=np.float64).ravel()
    if p.min() >= 0 and p.max() <= 1:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    return p  # already a logit / decision value


def _csv_vec(path):
    d = pd.read_csv(path)
    cols = [c for c in d.columns if c.lower() not in ("id", "unnamed: 0")]
    return d[cols[-1]].to_numpy()


def load_members():
    """-> dict name -> (oof_logit, test_logit)"""
    M = {}

    # ours
    for f in sorted(OUT.glob("oof_*.npy")):
        t = OUT / f"test_{f.name[4:]}"
        if t.exists():
            M[f"ours/{f.stem[4:]}"] = (np.load(f), np.load(t))

    # cdeotte 124
    d = PUB / "cdeotte124"
    for f in sorted(d.glob("oof_*.npy")):
        t = d / f"test_{f.name[4:]}"
        if t.exists():
            M[f"cd/{f.stem[4:]}"] = (np.load(f, allow_pickle=True), np.load(t, allow_pickle=True))

    # sachith7 npz
    for f in sorted((PUB / "sachith7").glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        if "oof" in z.files and "test" in z.files:
            M[f"s7/{f.stem}"] = (z["oof"], z["test"])

    # tabpfn / tabicl csv: pair oof_X.csv with test_X.csv
    d = PUB / "s6e10-tabpfn-member"
    for f in sorted(d.glob("oof_*.csv")):
        t = d / f"test_{f.name[4:]}"
        if t.exists():
            M[f"tp/{f.stem[4:]}"] = (_csv_vec(f), _csv_vec(t))

    # tfm npy
    d = PUB / "s6e10-tfm-oof-and-test-predictions"
    for f in sorted(d.glob("oof_*.npy")):
        t = d / f"test_{f.name[4:]}"
        if t.exists():
            M[f"tfm/{f.stem[4:]}"] = (np.load(f, allow_pickle=True), np.load(t, allow_pickle=True))

    return M


def main():
    y = pd.read_csv(DATA / "train.csv", usecols=[TARGET])[TARGET].astype(int).values
    ids = pd.read_csv(DATA / "test.csv", usecols=["id"])["id"]
    assert len(y) == N_TRAIN and len(ids) == N_TEST

    raw = load_members()
    print(f"loaded {len(raw)} candidate members")

    # ---- gate -------------------------------------------------------------
    kept, rejected = {}, []
    for name, (o, t) in raw.items():
        o, t = np.asarray(o, dtype=np.float64).ravel(), np.asarray(t, dtype=np.float64).ravel()
        if o.shape[0] != N_TRAIN or t.shape[0] != N_TEST:
            rejected.append((name, f"shape {o.shape[0]}/{t.shape[0]}")); continue
        if not (np.isfinite(o).all() and np.isfinite(t).all()):
            rejected.append((name, "non-finite")); continue
        auc = roc_auc_score(y, o)
        if auc < 0.5:
            o, t, auc = -o, -t, 1 - auc          # inverted score, flip it
        if not (AUC_MIN <= auc <= AUC_MAX):
            rejected.append((name, f"auc {auc:.5f} outside gate")); continue
        kept[name] = (logit(o), logit(t), auc)
    print(f"gate: kept {len(kept)}, rejected {len(rejected)}")
    for n, why in rejected:
        print(f"  rejected {n:40s} {why}")

    # ---- dedupe near-identical members (keep the higher AUC) --------------
    names = sorted(kept, key=lambda n: -kept[n][2])
    O = np.column_stack([kept[n][0] for n in names])
    C = np.corrcoef(O, rowvar=False)
    keep_idx = []
    for i in range(len(names)):
        if all(abs(C[i, j]) < DUP_CORR for j in keep_idx):
            keep_idx.append(i)
    names = [names[i] for i in keep_idx]
    print(f"dedupe (|corr| >= {DUP_CORR}): {len(names)} members remain")

    O = np.column_stack([kept[n][0] for n in names])
    T = np.column_stack([kept[n][1] for n in names])
    aucs = np.array([kept[n][2] for n in names])
    print(f"\nbest single member: {names[0]}  {aucs[0]:.5f}")
    print(f"members by source: " + ", ".join(
        f"{s}={sum(n.startswith(s + '/') for n in names)}" for s in ("ours", "cd", "s7", "tp", "tfm")))

    # ---- nested-CV LR over C grid -----------------------------------------
    outer = list(StratifiedKFold(5, shuffle=True, random_state=0).split(O, y))
    mu, sd = O.mean(0), O.std(0) + 1e-9
    Z, ZT = (O - mu) / sd, (T - mu) / sd

    best_c, best_cv = None, -1
    for Cval in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        oof = np.zeros(len(y))
        for tr, va in outer:
            lr = LogisticRegression(C=Cval, max_iter=2000).fit(Z[tr], y[tr])
            oof[va] = lr.decision_function(Z[va])
        cv = roc_auc_score(y, oof)
        print(f"  C={Cval:<6} nested CV {cv:.5f}")
        if cv > best_cv:
            best_c, best_cv = Cval, cv

    lr = LogisticRegression(C=best_c, max_iter=2000).fit(Z, y)
    print(f"\nLR stack: nested CV {best_cv:.5f}  (C={best_c}, {len(names)} members)")
    w = pd.Series(lr.coef_[0], index=names).sort_values(key=np.abs, ascending=False)
    print("top weights:")
    for n, v in w.head(15).items():
        print(f"  {n:44s} {v:+.3f}  (solo {kept[n][2]:.5f})")

    pred = lr.decision_function(ZT)
    pred = 1 / (1 + np.exp(-pred))
    path = OUT / f"sub_pubstack_cv{best_cv:.5f}.csv"
    pd.DataFrame({"id": ids, TARGET: pred}).to_csv(path, index=False)
    print("\nwrote", path.name)


if __name__ == "__main__":
    main()
