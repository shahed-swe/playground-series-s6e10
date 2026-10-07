# Predicting Airline Satisfaction — Playground Series S6E10

Kaggle Playground Series, Season 6 Episode 10. Binary classification of
passenger satisfaction, scored by ROC AUC. Deadline **31 Oct 2026**.

See **[PLAN.md](PLAN.md)** for status and open work.

## The data

| | rows | columns |
|---|---|---|
| train | 699,635 | 22 features + `satisfaction` |
| test | 299,844 | 22 features |

- Target `satisfaction` is boolean, **44.4% positive** — mildly imbalanced, not
  enough to need special handling for AUC.
- 14 of the features are 0–5 survey ratings (wifi, seat comfort, cleanliness…).
- 4 string categoricals, all low-cardinality: Gender (2), Customer Type (2),
  Type of Travel (2), Class (3).
- Numeric: Age, Flight Distance, Departure Delay, Arrival Delay.
- Missing values: only `Arrival Delay in Minutes`, 0.04%.

Playground Series data is synthetic, generated from a real source dataset. The
source here is *Airline Passenger Satisfaction* (teejmahal20), fetched into
`data/original/`. Appending the original rows to training is a standard
Playground gain and the first thing to test.

## Results so far

| Model | CV AUC | Public LB |
|---|---|---|
| LightGBM baseline | 0.95903 | 0.95840 |
| LightGBM + in-fold TE + freq + teacher | 0.96059 | 0.96017 |
| CatBoost crosses + teacher (Kaggle GPU) | 0.96117 | — |
| LR stack, 8 own members (nested CV) | 0.96121 | 0.96056 |
| 3-seed CatBoost + 10-fold LGB blend | 0.96129 | 0.96057 |
| **LR stack, 137 members: ours + public OOF libraries** | **0.96210** | **0.96174** |

Rank **44 / 988** as of 7 Oct. Leader 0.96177.

The final step stacks the public OOF libraries — chiefly Chris Deotte's
124-member aggregation and his tabular-foundation-model members — with our own
12 members, all on the identical CV split. Every member is gated on a plausible
OOF AUC (a leaked member would look superb in CV and sink the board), near-
duplicates are collapsed, and the stacker is scored by nested CV. Credit for the
members belongs to their authors; see PLAN.md.

## What moved the score

Not what I expected. Hand-crafted rating aggregates *hurt* (−0.00023), ratings
as categorical did nothing, original rows appended hurt (−0.00035). What worked,
all measured on the same folds and all first reported by others on the forum:

- **Cross-fitted target encoding of exact Flight Distance / Age / delays:**
  +0.00111. The synthetic generator leaves a learnable per-value target rate.
- **A teacher model trained on the original dataset, used as a logit feature:**
  part of a further +0.00036. Rows mislead because the generator drifted Flight
  Distance and Age; a model trained on them still transfers.
- **CatBoost with categorical copies of the numerics + 39 rating × context
  crosses** (wangxintong111): 0.96117 — CatBoost's ordered target statistics do
  the target encoding natively.
- **Logistic regression on logits** as the stacker, scored by nested CV.

Verified clean: zero duplicate rows within or across train/test, no id-ordering
leak. The leaderboard is genuine modelling.

## Leaderboard shape

Top public scores span **0.96172–0.96177**. The field converges fast; rank is
decided by target encoding, a teacher, 10 folds, and stacking across model
families — not by architecture or by hand-crafted features.

## Layout

```
data/              competition CSVs + data/original/ (gitignored, re-downloadable)
src/               shared code
notebooks/
submissions/       generated submission files (gitignored)
```

## Running

Uses the shared workspace venv one level up. Every run shares the same
`StratifiedKFold(5, shuffle=True, random_state=42)` split — the community
split — so OOFs are directly comparable and stackable, including with public
OOF libraries.

```bash
../.venv/bin/python src/teacher.py                              # once
../.venv/bin/python src/experiment.py --model lgb --te --freq --teacher
../.venv/bin/python src/blend.py                                # rank blend + LR stack
```

Heavy models (CatBoost with categorical numerics, 10-fold runs) are Kaggle
notebooks under `notebooks/`; the laptop is routinely starved by other apps.
