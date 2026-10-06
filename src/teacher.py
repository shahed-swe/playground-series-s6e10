"""Teacher model trained on the ORIGINAL airline dataset only.

Appending original rows to PS training hurt CV (-0.00035 here, -0.00038 in
the forum ablation) because the synthetic generator drifted Flight Distance
and Age. But a model trained on the original data and used as a *feature*
transfers: the forum measured +0.00059 from exactly this.

The teacher never sees PS labels, so its predictions for PS train rows carry
no leakage and need no cross-fitting. We still report the teacher's own CV on
original data as a sanity check.
"""

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
TARGET = "satisfaction"
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]

t0 = time.time()
orig = pd.concat([pd.read_csv(DATA / "original" / f) for f in ("train.csv", "test.csv")],
                 ignore_index=True)
orig = orig.drop(columns=[c for c in ("Unnamed: 0", "id", "Inflight service") if c in orig.columns])
orig[TARGET] = orig[TARGET].astype(str).str.strip().eq("satisfied").astype(int)

ps_train = pd.read_csv(DATA / "train.csv")
ps_test = pd.read_csv(DATA / "test.csv")
features = [c for c in ps_test.columns if c != "id"]
assert set(features) <= set(orig.columns), set(features) - set(orig.columns)

levels = {c: sorted(pd.concat([orig[c], ps_train[c], ps_test[c]]).astype(str).unique()) for c in CATS}
def enc(df):
    df = df[features].copy()
    for c in CATS:
        df[c] = pd.Categorical(df[c].astype(str), categories=levels[c])
    return df

X, y = enc(orig), orig[TARGET].values
print(f"original: {X.shape}, positive rate {y.mean():.4f}")

params = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=63,
              min_child_samples=40, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=2.0, verbose=-1, seed=42, num_threads=-1)

# Sanity CV on the original data itself.
oof = np.zeros(len(X)); iters = []
for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
    m = lgb.train(params, lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=CATS),
                  num_boost_round=5000,
                  valid_sets=[lgb.Dataset(X.iloc[va], y[va], categorical_feature=CATS)],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration); iters.append(m.best_iteration)
print(f"teacher CV on original data: AUC {roc_auc_score(y, oof):.5f}  (iters {np.mean(iters):.0f})")

# Final teacher on all original rows, fixed round count from the CV.
final = lgb.train(params, lgb.Dataset(X, y, categorical_feature=CATS),
                  num_boost_round=int(np.mean(iters) * 1.1))
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
tr_logit = logit(final.predict(enc(ps_train)))
te_logit = logit(final.predict(enc(ps_test)))
np.save(OUT / "teacher_train.npy", tr_logit); np.save(OUT / "teacher_test.npy", te_logit)

# How well does the teacher alone transfer to PS? This is the number that
# tells us whether the drift is survivable.
print(f"teacher alone on PS train: AUC {roc_auc_score(ps_train[TARGET].astype(int), tr_logit):.5f}")
print(f"saved teacher_train/test.npy  {(time.time()-t0)/60:.1f} min")
