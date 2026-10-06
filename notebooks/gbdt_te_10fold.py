"""LightGBM + XGBoost with in-fold target encoding, teacher and digits, 10 folds.

Port of src/experiment.py + src/encoders.py to run on Kaggle. Two reasons:
  * 10 folds measured +0.0001 to +0.0002 on every model (sachith7), and
    10-fold members were one of only two things that added to the public
    stack (kratosyan).
  * The laptop is starved by other applications (load ~15); single 3-minute
    runs could not finish in 30. Kaggle CPU is quiet.

Everything is cross-fitted inside each fold exactly as locally. Same seed, so
these OOFs stack with the local ones.
"""

import glob
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

T0 = time.time()
TARGET = "satisfaction"
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
TE_COLS = ["Flight Distance", "Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
SEED = 42
N_FOLDS = 10


def find(name):
    for root in ("/kaggle/input/competitions", "/kaggle/input/datasets", "/kaggle/input"):
        for d in range(1, 5):
            hits = glob.glob(root + "/*" * d + "/" + name)
            if hits:
                return hits[0]
    raise FileNotFoundError(name)


# ------------------------------------------------------------- encoders

def _te_map(keys, y, prior, m):
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + m * prior) / (g["count"] + m)


def target_encode(train_keys, y_train, other_keys, n_inner=5, m=20, seed=42):
    train_keys = pd.Series(train_keys).reset_index(drop=True)
    y_train = np.asarray(y_train, dtype=float)
    prior = y_train.mean()
    enc_train = np.full(len(train_keys), prior, dtype=float)
    for fit_idx, enc_idx in KFold(n_inner, shuffle=True, random_state=seed).split(train_keys):
        mp = _te_map(train_keys.iloc[fit_idx].values, y_train[fit_idx], prior, m)
        enc_train[enc_idx] = train_keys.iloc[enc_idx].map(mp).fillna(prior).values
    full = _te_map(train_keys.values, y_train, prior, m)
    return enc_train, [pd.Series(k).map(full).fillna(prior).values for k in other_keys]


# ------------------------------------------------------------- data

train = pd.read_csv(find("train.csv"))
test = pd.read_csv(find("test.csv"))
y = train[TARGET].astype(int).values

train["teacher_logit"] = np.load(find("teacher_train.npy"))
test["teacher_logit"] = np.load(find("teacher_test.npy"))

for df in (train, test):
    fd = df["Flight Distance"].astype(int)
    df["fd_d0"], df["fd_d1"], df["fd_d2"] = fd % 10, (fd // 10) % 10, (fd // 100) % 10
    df["fd_mod50"], df["fd_mod100"] = fd % 50, fd % 100
    for c in TE_COLS:
        counts = pd.concat([train[c], test[c]]).fillna(-1).value_counts()
        df[f"freq_{c}"] = df[c].fillna(-1).map(counts).fillna(0).values

features = [c for c in train.columns if c not in ("id", TARGET)]
levels = {c: sorted(pd.concat([train[c], test[c]]).astype(str).unique()) for c in CATS}
for df in (train, test):
    for c in CATS:
        df[c] = pd.Categorical(df[c].astype(str), categories=levels[c])
X, X_te = train[features], test[features]
print(f"{len(features)} features; {N_FOLDS} folds")


# ------------------------------------------------------------- models

def run_lgb(X_tr, y_tr, X_va, y_va, X_tt):
    p = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
             min_child_samples=50, feature_fraction=0.7, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=2.0, verbose=-1, seed=SEED, num_threads=-1)
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=CATS)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=CATS, reference=dtr)
    m = lgb.train(p, dtr, 20000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    return (m.predict(X_va, num_iteration=m.best_iteration),
            m.predict(X_tt, num_iteration=m.best_iteration), m.best_iteration)


def run_xgb(X_tr, y_tr, X_va, y_va, X_tt):
    p = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.03,
             max_depth=8, min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
             reg_lambda=2.0, tree_method="hist", seed=SEED)
    dtr = xgb.DMatrix(X_tr, y_tr, enable_categorical=True)
    dva = xgb.DMatrix(X_va, y_va, enable_categorical=True)
    dtt = xgb.DMatrix(X_tt, enable_categorical=True)
    m = xgb.train(p, dtr, 20000, evals=[(dva, "va")], early_stopping_rounds=300,
                  verbose_eval=False)
    it = (0, m.best_iteration + 1)
    return m.predict(dva, iteration_range=it), m.predict(dtt, iteration_range=it), m.best_iteration


for name, runner in (("lgb", run_lgb), ("xgb", run_xgb)):
    tag = f"{name}_te_freq_teacher_digits_f{N_FOLDS}"
    oof, pred, iters = np.zeros(len(X)), np.zeros(len(X_te)), []
    for fold, (tr, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y)):
        X_tr, X_va, X_tt = X.iloc[tr].copy(), X.iloc[va].copy(), X_te.copy()
        for c in TE_COLS:
            e_tr, (e_va, e_tt) = target_encode(
                train[c].iloc[tr].fillna(-1).values, y[tr],
                [train[c].iloc[va].fillna(-1).values, test[c].fillna(-1).values], seed=SEED)
            X_tr[f"te_{c}"], X_va[f"te_{c}"], X_tt[f"te_{c}"] = e_tr, e_va, e_tt
        p_va, p_tt, it = runner(X_tr, y[tr], X_va, y[va], X_tt)
        oof[va] = p_va; pred += p_tt / N_FOLDS; iters.append(it)
        print(f"  {tag} fold {fold}: {roc_auc_score(y[va], p_va):.5f}  iters {it}  "
              f"{(time.time()-T0)/60:.1f} min", flush=True)
    cv = roc_auc_score(y, oof)
    print(f"\n{tag}: OOF AUC {cv:.5f}  (mean iters {np.mean(iters):.0f})\n")
    np.save(f"/kaggle/working/oof_{tag}.npy", oof)
    np.save(f"/kaggle/working/test_{tag}.npy", pred)
    pd.DataFrame({"id": test["id"], TARGET: pred}).to_csv(
        f"/kaggle/working/sub_{tag}_cv{cv:.5f}.csv", index=False)

print(f"done in {(time.time()-T0)/60:.1f} min")
