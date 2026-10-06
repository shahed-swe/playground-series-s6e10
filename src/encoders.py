"""Leak-free encoders applied inside the CV loop.

Target encoding has to be cross-fitted: a training row must never see its own
label in its encoding, or the model learns the leak instead of the signal and
CV collapses on the leaderboard. Within an outer fold, training rows are
encoded by an inner K-fold over the training portion; validation and test rows
are encoded from the full training portion.

Smoothing pulls rare values toward the global prior so a Flight Distance seen
three times does not get an encoding of 1.0 or 0.0.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def _te_map(keys, y, prior, m):
    """value -> smoothed mean target, (n*mean + m*prior) / (n + m)."""
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + m * prior) / (g["count"] + m)


def target_encode(train_keys, y_train, other_keys, n_inner=5, m=20, seed=42):
    """Cross-fitted smoothed target encoding.

    train_keys : Series of category keys for the training rows of this fold
    y_train    : their labels
    other_keys : list of Series to encode from the full training portion
                 (validation rows, test rows)
    Returns (encoded_train, [encoded_other, ...]) as float arrays.
    """
    train_keys = pd.Series(train_keys).reset_index(drop=True)
    y_train = np.asarray(y_train, dtype=float)
    prior = y_train.mean()

    enc_train = np.full(len(train_keys), prior, dtype=float)
    kf = KFold(n_inner, shuffle=True, random_state=seed)
    for fit_idx, enc_idx in kf.split(train_keys):
        mp = _te_map(train_keys.iloc[fit_idx].values, y_train[fit_idx], prior, m)
        enc_train[enc_idx] = train_keys.iloc[enc_idx].map(mp).fillna(prior).values

    full_map = _te_map(train_keys.values, y_train, prior, m)
    enc_other = [pd.Series(k).map(full_map).fillna(prior).values for k in other_keys]
    return enc_train, enc_other


def frequency_encode(all_keys, *series):
    """Count of each value across train+test. Rarity is signal in synthetic data."""
    counts = pd.Series(all_keys).value_counts()
    return [pd.Series(s).map(counts).fillna(0).values for s in series]
