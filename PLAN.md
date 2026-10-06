# Plan — Predicting Airline Satisfaction (PS S6E10)

**Status as of 6 Oct 2026.** Public LB **0.96056**, rank **350 / 936**. Deadline **31 Oct**
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
| `lgb_nofe_lr0.03` | **0.95912** | — | **Ablation: new params, no features. Best single so far.** Params +0.00009; engineered features −0.00023. **Features dropped.** |
| `lgb_ratcat` | 0.95890 | — | ratings as categorical. No gain. Dropped. |
| `blend` (4 LGB/XGB) | 0.95922 | — | greedy rank blend. **+0.0001 over best single** — blending correlated GBDTs will not close the 0.0033 gap. |
| `teacher` (LGB on original 130k rows) | 0.99515 on original; **0.95463 alone on PS train** | — | A new, differently-sourced signal only 0.0045 below the best PS model. Saved as logit feature. |
| **`lgb_te_freq_teacher_lr0.03`** | **0.96059** | **0.96017** | **+0.00147 over best single, better on all 5 folds.** In-fold TE of Flight Distance/Age/delays + freq + teacher logit. Encoder passed 5 leak tests. |
| `lgb_te_lr0.03` | 0.96023 | — | **Ablation: TE alone = +0.00111.** Teacher + freq contribute the remaining +0.00036. |
| `cat_nofe_lr0.1` | 0.95862 | — | CatBoost, plain features, 18 min CPU. Weak alone; needs categorical numeric copies + GPU to shine. |
| `xgb_te_freq_teacher` | killed | — | overlapped with two other runs; thrashed past the 30-min cap. Rerun alone. |
| **`cat_crosses_teacher`** (Kaggle GPU) | **0.96117** | — | **Best single.** Ratings categorical + categorical numeric copies + 39 rating×context crosses + teacher logit. 40 min on T4; all folds ≥0.9602. |
| **`stack_lr`** (8 members, nested CV) | **0.96121** | **0.96056** | LR on logits; CatBoost weight 0.78. +0.00004 over CatBoost alone — GBDT members saturated. |
| `lgb_te_freq_teacher_digits_f10` (Kaggle CPU) | **0.96076** | — | 10 folds + digits: +0.00017 over the 5-fold run. 25 min for both models. |
| `xgb_te_freq_teacher_digits_f10` (Kaggle CPU) | **0.96074** | — | XGBoost with TE: 0.95869 → 0.96074. Strong second family. |
| `cat_crosses_teacher_s7` / `_s2026` (GPU) | 0.96113 / 0.96112 | — | two more seeds, 67 min. Within 0.00005 of seed 42 — stable. |
| **`blend` 3×CatBoost + LGB-f10** | **0.96129** | submitted | rank blend = LR stacker (tie). 3-seed CatBoost alone 0.96122. **GBDT family saturated: +0.00002/step.** |
| `stack_lr` (10 members) | 0.96127 | not submitted | +0.00006 with the 10-fold members. Holding for CatBoost seeds to submit one larger step. |
| `lgb_*_digits` (local) | killed ×2 | — | starved: machine load 15 from Chrome/NordVPN/VS Code, not from training. Retry when the machine is free, or on Kaggle. |
| `cat_fe_lr0.05` | killed | — | 2/5 folds (0.95866, 0.95755) in 14 min under CPU contention; would have hit the 30-min limit. Relaunched alone, no features, lr 0.1. |

Leaderboard top: 0.96177. Gap from current LB: **0.0012**.

## Next, in order
1. ~~Baseline~~ done; LB score pending for CV calibration.
2. ~~Original data~~ — tested both ways, no gain. Dropped.
3. ~~Feature work~~ — hand-crafted aggregates hurt. **The right features were
   target encodings of exact values + a teacher** (see recipe below): +0.00147.
   Ablating TE / teacher / freq contributions next.
4. **Model diversity** — XGBoost with TE+teacher locally (rerun alone);
   CatBoost with categorical numeric copies + 39 crosses on **Kaggle GPU**
   (`notebooks/catboost_crosses.py`, pushed 6 Oct). Then LR-on-logits stack.
5. ~~10 folds~~ done on Kaggle: +0.00017. **Tuning** — Optuna on CatBoost
   once seeds are in; GBDT members otherwise saturated.

## Ruled out (checked, 6 Oct)
- **No leak.** 0 exact duplicate feature-rows within train or train↔test;
  0.02% match even ignoring delays. id-vs-target AUC 0.50007; positive rate
  flat over id. The 0.0033 gap to the top is modelling, not structure.
- **Original data** — no gain plain or with indicator.
- **Hand-crafted rating aggregates** — −0.00023 in a clean ablation.
- **Ratings as categorical** — no gain.
- **Blending correlated GBDTs** — +0.0001.

## The recipe (from forum ablations, 6 Oct) — measured per-step deltas

| Lever | Δ CV AUC | Who measured it |
|---|---|---|
| **Cross-fitted target encoding of exact Flight Distance** | **+0.00135** | sachith7 (745908) |
| **Teacher trained on original data → logit feature** | **+0.00059** | sachith7 |
| 39 rating × {Class, Travel, Customer} categorical crosses, CatBoost | +0.00034, 5/5 folds | wangxintong111 (745892) |
| Flight-distance digit features | +0.00036 | karttikjangid05 (745098) |
| Route-profile / expected-rating aux features | ~+0.00014 each | sachith7 |
| 10 folds vs 5 | +0.0001–0.0002 | sachith7 |
| Logistic-regression stack on logits | best stacker; nested CV 0.96200 | kratosyan (746148) |
| Original rows appended | **−0.00038** | karttikjangid05 — matches ours |

Why: Flight Distance has 3,474 exact values and the synthetic generator leaves
a learnable per-value target rate. The generator drifted Flight Distance and
Age relative to the original (starkhushi, 745932), which is why appending
original *rows* hurts while a *teacher* trained on them transfers.

The community split is `StratifiedKFold(5, shuffle=True, random_state=42)` —
identical to ours, so all numbers are directly comparable. CatBoost with
categorical copies of the numerics reaches **0.96049 OOF as a control** — that
is native target statistics doing the TE automatically.

## Compute note
Three concurrent trainings with `num_threads=-1` thrash the CPU: a 3-min LightGBM
run hit the 30-min background cap. **Max two at once locally.** Then, on 6 Oct
afternoon, load averaged ~15 from Chrome / NordVPN / VS Code alone and even a
*single* 3-min run could not finish in 30. **Heavy training now goes to Kaggle
GPU** — the CatBoost recipe ran there in 40 min while the laptop starved.

## Constraints
- Leaderboard is tight (top 8 within 0.00005). Trust CV; use LB sparingly.
- Keep a fixed fold seed so every delta is comparable.
