# Plan — Predicting Airline Satisfaction (PS S6E10)

**Status as of 6 Oct 2026.** Entered, no submissions yet. Deadline **31 Oct**
(25 days). 10 submissions/day, max team size 3, file-upload submissions.

## Done
- Entered. Data downloaded and profiled (clean, no real missingness).
- Original source dataset fetched for the append trick.
- **Baseline** — LightGBM, 5-fold stratified CV, seed 42, early stopping.

## Results log

| Run | CV AUC | Public LB | Notes |
|---|---|---|---|
| `lgb_base` | 0.95903 | **0.95840** | lr 0.05, 127 leaves, ~285 rounds. **CV runs ~0.0006 optimistic.** |
| `lgb_orig` | 0.95868 | — | original rows appended. **Worse on all 5 folds** (−0.00035). |
| `lgb_origflag` | 0.95906 | — | + `is_original` indicator. +0.00003, mixed fold signs — a wash. **Original data dropped.** |
| `lgb_fe_lr0.03` | 0.95889 | — | engineered features + lr 0.03 + ff 0.7 + l2 2.0. **Worse on 4/5 folds.** Confounded — too many changes at once; ablating. |
| `xgb_fe_lr0.03` | 0.95869 | — | XGBoost, same features. Blend member. |

Leaderboard top: 0.96177. Gap from baseline LB: **0.0034**.

## Next, in order
1. ~~Baseline~~ done; LB score pending for CV calibration.
2. ~~Original data~~ — tested both ways, no gain. Dropped.
3. **Feature work** — rating aggregates (mean/std/count of 5s and 0s), delay
   difference, distance per rating; test each against CV.
4. **Model diversity** — XGBoost + CatBoost alongside LightGBM, rank-average.
5. **Tuning** — Optuna on the best single model once features settle.

## Constraints
- Leaderboard is tight (top 8 within 0.00005). Trust CV; use LB sparingly.
- Keep a fixed fold seed so every delta is comparable.
