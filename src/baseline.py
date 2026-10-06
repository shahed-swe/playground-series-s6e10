"""LightGBM baseline with stratified 5-fold CV.

Purpose: establish a CV number and a first leaderboard point so that the
CV-vs-LB relationship is known before any feature work. Every later change is
measured as a delta against this, on the same folds (fixed seed).

    ../.venv/bin/python src/baseline.py              # PS train only
    ../.venv/bin/python src/baseline.py --original   # + source-dataset rows

The --original flag appends the real Airline Passenger Satisfaction rows to
the training folds only. They are never used for validation: the test set is
synthetic, so validating on real rows would measure the wrong distribution.
"""

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
OUT = HERE / "submissions"
OUT.mkdir(exist_ok=True)

TARGET = "satisfaction"
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
SEED = 42
N_FOLDS = 5

PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    seed=SEED,
    num_threads=-1,
)


def load_original():
    """Source dataset, mapped onto the PS schema."""
    frames = []
    for name in ("train.csv", "test.csv"):
        f = DATA / "original" / name
        if f.exists():
            frames.append(pd.read_csv(f))
    o = pd.concat(frames, ignore_index=True)
    o = o.drop(columns=[c for c in ("Unnamed: 0", "id") if c in o.columns])
    # Source target is a string; PS is boolean.
    o[TARGET] = o[TARGET].astype(str).str.strip().eq("satisfied")
    return o


def prepare(df):
    df = df.copy()
    for c in CATS:
        df[c] = df[c].astype("category")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", action="store_true",
                    help="append source-dataset rows to the training folds")
    ap.add_argument("--orig-flag", action="store_true",
                    help="like --original, but add an is_original indicator so the "
                         "model can separate the real and synthetic distributions")
    ap.add_argument("--rounds", type=int, default=5000)
    args = ap.parse_args()
    use_orig = args.original or args.orig_flag

    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    features = [c for c in train.columns if c not in ("id", TARGET)]

    orig = load_original()[features + [TARGET]] if use_orig else None
    if args.orig_flag:
        # Plain append hurt CV on all 5 folds (0.95903 -> 0.95868). The
        # indicator lets trees condition on provenance; test rows are all 0.
        train["is_original"] = 0
        test["is_original"] = 0
        orig["is_original"] = 1
        features = features + ["is_original"]
    tag = "lgb_origflag" if args.orig_flag else ("lgb_orig" if args.original else "lgb_base")
    print(f"train {train.shape} | test {test.shape}"
          + (f" | original {orig.shape}" if orig is not None else ""))

    # Align categorical codes across every frame so LightGBM sees one mapping.
    all_cat = pd.concat([train[CATS], test[CATS]] + ([orig[CATS]] if orig is not None else []))
    cat_levels = {c: sorted(all_cat[c].dropna().unique()) for c in CATS}

    def encode(df):
        df = df.copy()
        for c in CATS:
            df[c] = pd.Categorical(df[c], categories=cat_levels[c])
        return df

    X = encode(train[features]); y = train[TARGET].astype(int).values
    X_test = encode(test[features])
    X_orig = encode(orig[features]) if orig is not None else None
    y_orig = orig[TARGET].astype(int).values if orig is not None else None

    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        if X_orig is not None:
            X_tr = pd.concat([X_tr, X_orig], ignore_index=True)
            y_tr = np.concatenate([y_tr, y_orig])
        dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=CATS)
        dva = lgb.Dataset(X.iloc[va_idx], y[va_idx], categorical_feature=CATS, reference=dtr)

        m = lgb.train(PARAMS, dtr, num_boost_round=args.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va_idx] = m.predict(X.iloc[va_idx], num_iteration=m.best_iteration)
        pred += m.predict(X_test, num_iteration=m.best_iteration) / N_FOLDS
        best_iters.append(m.best_iteration)
        print(f"  fold {fold}: auc {roc_auc_score(y[va_idx], oof[va_idx]):.5f}  "
              f"iters {m.best_iteration}  {time.time()-t0:.0f}s", flush=True)

    cv = roc_auc_score(y, oof)
    print(f"\n{tag}: OOF AUC {cv:.5f}  (mean best iter {np.mean(best_iters):.0f})  "
          f"{(time.time()-t0)/60:.1f} min")

    np.save(OUT / f"oof_{tag}.npy", oof)
    sub = pd.DataFrame({"id": test["id"], TARGET: pred})
    path = OUT / f"sub_{tag}_cv{cv:.5f}.csv"
    sub.to_csv(path, index=False)
    print("wrote", path.name)


if __name__ == "__main__":
    main()
