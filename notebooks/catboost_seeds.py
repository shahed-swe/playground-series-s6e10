"""CatBoost crosses + teacher, two more seeds, for a 3-seed average. GPU.

Seed 42 is already done (OOF 0.96117, notebooks/catboost_crosses.py). CatBoost
carries 0.78 of the stack weight, so variance reduction on it is the most
reliable remaining gain: seed-averaging a dominant GBDT member is typically
worth +0.0001 to +0.0003 and costs nothing but GPU minutes.

Identical features and parameters to the seed-42 run; only random_seed changes.
The CV split itself stays at random_state=42 so OOFs remain stackable.
"""

import glob
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T0 = time.time()
TARGET = "satisfaction"
SPLIT_SEED = 42
MODEL_SEEDS = [7, 2026]
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
teacher_tr, teacher_te = np.load(find("teacher_train.npy")), np.load(find("teacher_test.npy"))


def cat_str(s):
    return s.astype("string").fillna("__MISSING__").astype(str)


def make(df, teacher):
    X = pd.DataFrame(index=df.index); cats = []
    for c in PASSENGER + RATINGS:
        X[c] = cat_str(df[c]); cats.append(c)
    for c in NUMERIC:
        X[c] = df[c].astype(float)
        X[c + "__cat"] = cat_str(df[c]); cats.append(c + "__cat")
    for r in RATINGS:
        for b in BACKGROUND:
            X[f"{r}__X__{b}"] = X[r] + "|" + X[b]; cats.append(f"{r}__X__{b}")
    X["teacher_logit"] = teacher
    return X, cats


X, cats = make(train, teacher_tr)
X_te, _ = make(test, teacher_te)
folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SPLIT_SEED).split(X, y))
print(f"{X.shape[1]} features, {len(cats)} categorical; seeds {MODEL_SEEDS}")

for seed in MODEL_SEEDS:
    params = dict(loss_function="Logloss", eval_metric="AUC", task_type="GPU",
                  depth=6, learning_rate=0.05, l2_leaf_reg=3, iterations=6000,
                  od_type="Iter", od_wait=150, random_seed=seed, verbose=0)
    oof, pred, iters = np.zeros(len(X)), np.zeros(len(X_te)), []
    for fold, (tr, va) in enumerate(folds):
        m = CatBoostClassifier(**params)
        m.fit(Pool(X.iloc[tr], y[tr], cat_features=cats),
              eval_set=Pool(X.iloc[va], y[va], cat_features=cats), use_best_model=True)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        pred += m.predict_proba(X_te)[:, 1] / N_FOLDS
        iters.append(m.get_best_iteration())
        print(f"  seed {seed} fold {fold}: {roc_auc_score(y[va], oof[va]):.5f}  iters {iters[-1]}  "
              f"{(time.time()-T0)/60:.1f} min", flush=True)
    cv = roc_auc_score(y, oof)
    tag = f"cat_crosses_teacher_s{seed}"
    print(f"\n{tag}: OOF AUC {cv:.5f}  (mean iters {np.mean(iters):.0f})\n")
    np.save(f"/kaggle/working/oof_{tag}.npy", oof)
    np.save(f"/kaggle/working/test_{tag}.npy", pred)

print(f"done in {(time.time()-T0)/60:.1f} min")
