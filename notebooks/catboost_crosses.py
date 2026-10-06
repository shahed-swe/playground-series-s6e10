"""CatBoost with categorical numeric copies + 39 rating-context crosses. GPU.

This is the recipe wangxintong111 measured at 0.96080 OOF on the same
StratifiedKFold(5, shuffle=True, random_state=42) split: every 0-5 rating as
categorical, exact-value categorical copies of Age / Flight Distance / delays
(CatBoost's ordered target statistics then do the target encoding natively),
plus each rating crossed with Class, Type of Travel and Customer Type.

It is the most *different* strong model available - CatBoost CTRs rather than
explicit TE, depth-6 symmetric trees rather than leaf-wise - so it is the
member most likely to add to a stack of LightGBM/XGBoost. It is also far too
slow on a laptop CPU, hence Kaggle GPU.

Adds the original-data teacher logit as well, uploaded alongside as a dataset.
Writes OOF + test predictions in the same format as src/experiment.py.
"""

import glob
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T0 = time.time()
TARGET = "satisfaction"
SEED = 42
N_FOLDS = 5
RATINGS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
PASSENGER = ["Gender", "Customer Type", "Type of Travel", "Class"]
BACKGROUND = ["Class", "Type of Travel", "Customer Type"]
NUMERIC = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]


def find(name):
    for root in ("/kaggle/input/competitions", "/kaggle/input/datasets", "/kaggle/input"):
        for d in range(1, 5):
            hits = glob.glob(root + "/*" * d + "/" + name)
            if hits:
                return hits[0]
    raise FileNotFoundError(name)


train = pd.read_csv(find("train.csv"))
test = pd.read_csv(find("test.csv"))
y = train[TARGET].astype(int).values

teacher_tr = teacher_te = None
try:
    teacher_tr = np.load(find("teacher_train.npy"))
    teacher_te = np.load(find("teacher_test.npy"))
    print("teacher logits attached")
except FileNotFoundError:
    print("no teacher dataset attached; running without")


def cat_str(s):
    return s.astype("string").fillna("__MISSING__").astype(str)


def make(df, teacher):
    X = pd.DataFrame(index=df.index)
    cats = []
    for c in PASSENGER + RATINGS:
        X[c] = cat_str(df[c]); cats.append(c)
    for c in NUMERIC:
        X[c] = df[c].astype(float)
        X[c + "__cat"] = cat_str(df[c]); cats.append(c + "__cat")
    for r in RATINGS:
        for b in BACKGROUND:
            n = f"{r}__X__{b}"
            X[n] = X[r] + "|" + X[b]; cats.append(n)
    if teacher is not None:
        X["teacher_logit"] = teacher
    return X, cats


X, cats = make(train, teacher_tr)
X_te, _ = make(test, teacher_te)
print(f"{X.shape[1]} features, {len(cats)} categorical")

params = dict(loss_function="Logloss", eval_metric="AUC", task_type="GPU",
              depth=6, learning_rate=0.05, l2_leaf_reg=3, iterations=6000,
              od_type="Iter", od_wait=150, random_seed=SEED, verbose=500)

oof, pred, iters = np.zeros(len(X)), np.zeros(len(X_te)), []
for fold, (tr, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y)):
    m = CatBoostClassifier(**params)
    m.fit(Pool(X.iloc[tr], y[tr], cat_features=cats),
          eval_set=Pool(X.iloc[va], y[va], cat_features=cats), use_best_model=True)
    oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    pred += m.predict_proba(X_te)[:, 1] / N_FOLDS
    iters.append(m.get_best_iteration())
    print(f"fold {fold}: auc {roc_auc_score(y[va], oof[va]):.5f}  iters {iters[-1]}  "
          f"{(time.time()-T0)/60:.1f} min", flush=True)

cv = roc_auc_score(y, oof)
tag = "cat_crosses" + ("_teacher" if teacher_tr is not None else "")
print(f"\n{tag}: OOF AUC {cv:.5f}  (mean iters {np.mean(iters):.0f})  {(time.time()-T0)/60:.1f} min")
np.save(f"/kaggle/working/oof_{tag}.npy", oof)
np.save(f"/kaggle/working/test_{tag}.npy", pred)
pd.DataFrame({"id": test["id"], TARGET: pred}).to_csv(f"/kaggle/working/sub_{tag}_cv{cv:.5f}.csv", index=False)
print("saved")
