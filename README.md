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

## Leaderboard shape

Top 8 public scores span **0.96172–0.96177**. This is a tabular GBDT problem
where the field converges quickly and rank is decided by tuning, ensembling and
the original-data trick — not by model architecture.

## Layout

```
data/              competition CSVs + data/original/ (gitignored, re-downloadable)
src/               shared code
notebooks/
submissions/       generated submission files (gitignored)
```

## Running

Uses the shared workspace venv one level up:

```bash
../.venv/bin/python src/baseline.py
```
