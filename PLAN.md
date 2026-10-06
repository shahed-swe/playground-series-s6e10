# Plan — Predicting Airline Satisfaction (PS S6E10)

**Status as of 6 Oct 2026.** Entered, no submissions yet. Deadline **31 Oct**
(25 days). 10 submissions/day, max team size 3, file-upload submissions.

## Done
- Entered. Data downloaded and profiled (clean, no real missingness).
- Original source dataset fetched for the append trick.

## Next, in order
1. **Baseline** — LightGBM, 5-fold stratified CV, native categoricals. Get a
   CV AUC and a first submission on the board so CV-vs-LB calibration is known.
2. **Original data append** — add source rows to train, measure CV delta.
3. **Feature work** — rating aggregates (mean/std/count of 5s and 0s), delay
   difference, distance per rating; test each against CV.
4. **Model diversity** — XGBoost + CatBoost alongside LightGBM, rank-average.
5. **Tuning** — Optuna on the best single model once features settle.

## Constraints
- Leaderboard is tight (top 8 within 0.00005). Trust CV; use LB sparingly.
- Keep a fixed fold seed so every delta is comparable.
