"""Model-agnostic CV runner for feature and model experiments.

    ../.venv/bin/python src/experiment.py --model lgb --fe
    ../.venv/bin/python src/experiment.py --model xgb --fe
    ../.venv/bin/python src/experiment.py --model cat --fe

Every run uses the same stratified 5-fold split (seed 42) as baseline.py, so
CV numbers are directly comparable across runs. OOF and test predictions are
saved per run so blend.py can rank-average any subset.

Original-data append was tested in baseline.py and did not help (plain append
-0.00035 on all folds; with an indicator +0.00003, mixed signs). It is not used
here.
"""

import argparse
import time
from pathlib import Path

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
RATINGS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
SEED = 42
N_FOLDS = 5


def add_features(df):
    """Aggregates over the 0-5 survey ratings plus a few domain interactions.

    In this dataset a rating of 0 means 'not applicable', not 'terrible', so
    the count of zeros is a distinct signal from the mean.
    """
    df = df.copy()
    r = df[RATINGS]
    df["rating_mean"] = r.mean(axis=1)
    df["rating_std"] = r.std(axis=1)
    df["rating_min"] = r.min(axis=1)
    df["rating_max"] = r.max(axis=1)
    df["rating_n0"] = (r == 0).sum(axis=1)
    df["rating_n5"] = (r == 5).sum(axis=1)
    df["rating_n_low"] = (r <= 2).sum(axis=1)
    # Service-quality vs logistics split: the two groups load on satisfaction
    # differently in the source data.
    svc = ["Inflight wifi service", "Food and drink", "Online boarding",
           "Seat comfort", "Inflight entertainment", "On-board service",
           "Leg room service", "Cleanliness"]
    df["svc_mean"] = df[svc].mean(axis=1)
    df["logistics_mean"] = df[["Departure/Arrival time convenient",
                               "Ease of Online booking", "Gate location",
                               "Baggage handling", "Checkin service"]].mean(axis=1)
    # Delays
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(
        df["Departure Delay in Minutes"])
    df["delay_total"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["delay_diff"] = df["Arrival Delay in Minutes"] - df["Departure Delay in Minutes"]
    df["any_delay"] = (df["delay_total"] > 0).astype(int)
    df["log_distance"] = np.log1p(df["Flight Distance"])
    # Travel-type x class is the strongest categorical interaction in the source.
    df["travel_class"] = df["Type of Travel"].astype(str) + "|" + df["Class"].astype(str)
    df["loyal_travel"] = df["Customer Type"].astype(str) + "|" + df["Type of Travel"].astype(str)
    return df


def run_lgb(X_tr, y_tr, X_va, y_va, X_te, cats, lr):
    import lightgbm as lgb
    params = dict(objective="binary", metric="auc", learning_rate=lr,
                  num_leaves=127, min_child_samples=50, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                  verbose=-1, seed=SEED, num_threads=-1)
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cats)
    dva = lgb.Dataset(X_va, y_va, categorical_feature=cats, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=20000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(300, verbose=False)])
    return (m.predict(X_va, num_iteration=m.best_iteration),
            m.predict(X_te, num_iteration=m.best_iteration), m.best_iteration)


def run_xgb(X_tr, y_tr, X_va, y_va, X_te, cats, lr):
    import xgboost as xgb
    params = dict(objective="binary:logistic", eval_metric="auc", learning_rate=lr,
                  max_depth=8, min_child_weight=5, subsample=0.8,
                  colsample_bytree=0.7, reg_lambda=2.0, tree_method="hist",
                  enable_categorical=True, max_cat_to_onehot=1, seed=SEED)
    dtr = xgb.DMatrix(X_tr, y_tr, enable_categorical=True)
    dva = xgb.DMatrix(X_va, y_va, enable_categorical=True)
    dte = xgb.DMatrix(X_te, enable_categorical=True)
    m = xgb.train(params, dtr, num_boost_round=20000, evals=[(dva, "va")],
                  early_stopping_rounds=300, verbose_eval=False)
    it = (0, m.best_iteration + 1)
    return m.predict(dva, iteration_range=it), m.predict(dte, iteration_range=it), m.best_iteration


def run_cat(X_tr, y_tr, X_va, y_va, X_te, cats, lr):
    from catboost import CatBoostClassifier, Pool
    # CatBoost wants string categoricals, not pandas category dtype.
    def to_str(df):
        df = df.copy()
        for c in cats:
            df[c] = df[c].astype(str)
        return df
    X_tr, X_va, X_te = to_str(X_tr), to_str(X_va), to_str(X_te)
    m = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC",
                           learning_rate=lr, depth=8, l2_leaf_reg=3,
                           iterations=20000, od_type="Iter", od_wait=300,
                           random_seed=SEED, verbose=False, thread_count=-1)
    m.fit(Pool(X_tr, y_tr, cat_features=cats), eval_set=Pool(X_va, y_va, cat_features=cats),
          use_best_model=True)
    return (m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1],
            m.get_best_iteration())


RUNNERS = {"lgb": run_lgb, "xgb": run_xgb, "cat": run_cat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=RUNNERS, required=True)
    ap.add_argument("--fe", action="store_true", help="add engineered features")
    ap.add_argument("--ratings-cat", action="store_true",
                    help="treat the 0-5 survey ratings as categorical rather than "
                         "ordinal, so trees can isolate 0 = 'not applicable'")
    ap.add_argument("--te", action="store_true",
                    help="cross-fitted target encoding of exact Flight Distance, Age "
                         "and the two delays, computed inside each fold")
    ap.add_argument("--freq", action="store_true",
                    help="frequency encoding of the same columns over train+test")
    ap.add_argument("--teacher", action="store_true",
                    help="add logit from a model trained on the original dataset "
                         "(submissions/teacher_train.npy / teacher_test.npy)")
    ap.add_argument("--digits", action="store_true",
                    help="decimal digits of Flight Distance as features (+0.00036 in "
                         "the forum ablation; the generator leaks structure in them)")
    ap.add_argument("--folds", type=int, default=N_FOLDS,
                    help="CV folds; 10 measured +0.0001-0.0002 over 5")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    if args.fe:
        train, test = add_features(train), add_features(test)

    # Columns with many exact values that the synthetic generator ties to the
    # target. Flight Distance (3,474 values) is the big one.
    TE_COLS = ["Flight Distance", "Age", "Departure Delay in Minutes",
               "Arrival Delay in Minutes"]

    if args.teacher:
        # Teacher saw no PS labels, so its logit needs no cross-fitting.
        train["teacher_logit"] = np.load(OUT / "teacher_train.npy")
        test["teacher_logit"] = np.load(OUT / "teacher_test.npy")

    if args.digits:
        for df in (train, test):
            fd = df["Flight Distance"].astype(int)
            df["fd_d0"], df["fd_d1"], df["fd_d2"] = fd % 10, (fd // 10) % 10, (fd // 100) % 10
            df["fd_mod50"], df["fd_mod100"] = fd % 50, fd % 100

    if args.freq:
        from encoders import frequency_encode
        for c in TE_COLS:
            keys_all = pd.concat([train[c], test[c]]).fillna(-1)
            train[f"freq_{c}"], test[f"freq_{c}"] = frequency_encode(
                keys_all, train[c].fillna(-1), test[c].fillna(-1))

    cats = CATS + (["travel_class", "loyal_travel"] if args.fe else [])
    if args.ratings_cat:
        # Keep the numeric copies too when --fe is on: the aggregates were
        # computed from them and the ordinal signal is still real.
        cats = cats + RATINGS
    features = [c for c in train.columns if c not in ("id", TARGET)]
    levels = {c: sorted(pd.concat([train[c], test[c]]).astype(str).unique()) for c in cats}
    for df in (train, test):
        for c in cats:
            df[c] = pd.Categorical(df[c].astype(str), categories=levels[c])

    X, y = train[features], train[TARGET].astype(int).values
    X_te = test[features]
    flags = "".join(f"_{f}" for f in ("fe", "te", "freq", "teacher", "digits") if getattr(args, f))
    tag = args.tag or f"{args.model}{flags}_lr{args.lr}" + (f"_f{args.folds}" if args.folds != N_FOLDS else "")
    print(f"{tag}: {len(features)} features, {len(cats)} categorical"
          + (f" | +{len(TE_COLS)} target-encoded in-fold" if args.te else ""))

    oof, pred, iters = np.zeros(len(X)), np.zeros(len(X_te)), []
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        X_tr, X_va, X_tt = X.iloc[tr], X.iloc[va], X_te
        if args.te:
            # Cross-fitted within the training portion of THIS fold only, so no
            # validation label ever reaches a training-row encoding.
            from encoders import target_encode
            X_tr, X_va, X_tt = X_tr.copy(), X_va.copy(), X_te.copy()
            for c in TE_COLS:
                k_tr = train[c].iloc[tr].fillna(-1).values
                k_va = train[c].iloc[va].fillna(-1).values
                k_tt = test[c].fillna(-1).values
                e_tr, (e_va, e_tt) = target_encode(k_tr, y[tr], [k_va, k_tt], seed=SEED)
                X_tr[f"te_{c}"], X_va[f"te_{c}"], X_tt[f"te_{c}"] = e_tr, e_va, e_tt
        p_va, p_te, it = RUNNERS[args.model](X_tr, y[tr], X_va, y[va], X_tt, cats, args.lr)
        oof[va] = p_va; pred += p_te / args.folds; iters.append(it)
        print(f"  fold {fold}: auc {roc_auc_score(y[va], p_va):.5f}  iters {it}  "
              f"{time.time()-t0:.0f}s", flush=True)

    cv = roc_auc_score(y, oof)
    print(f"\n{tag}: OOF AUC {cv:.5f}  (mean iters {np.mean(iters):.0f})  "
          f"{(time.time()-t0)/60:.1f} min")
    np.save(OUT / f"oof_{tag}.npy", oof)
    np.save(OUT / f"test_{tag}.npy", pred)
    pd.DataFrame({"id": test["id"], TARGET: pred}).to_csv(
        OUT / f"sub_{tag}_cv{cv:.5f}.csv", index=False)
    print("saved oof/test/sub for", tag)


if __name__ == "__main__":
    main()
