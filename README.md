# Wimbledon Match Outcome Predictor

A machine-learning project that predicts the outcome of Wimbledon singles matches.
Given two players' rankings, ranking points, and the tournament round, the model
estimates the probability that **Player_1** beats **Player_2**.

It is a binary classification problem: the target is **1** when Player_1 wins and
**0** when Player_2 wins.

> All pipeline shapes, scores, and parameters below were read directly from the
> committed `wimbledon.ipynb` cell outputs, so the numbers here match what the
> notebook actually produces.

---

## Project Structure

```
tennis-ML/
├── Learning_Resource.md    # Step-by-step learning guide (conceptual)
├── README.md               # This technical reference
├── data/
│   └── atp_tennis.csv       # ATP match dataset (input)
├── wimbledon.ipynb          # Full workflow: load -> clean -> train -> tune -> save
└── predict.py               # Loads the saved model and exposes prediction functions
```

The trained model is the handoff between the two code artifacts: the notebook
serializes the tuned model to disk, and `predict.py` loads it to score new matchups.
(The `models/` directory is created at save time and is not committed.)

`Learning_Resource.md` is a teaching document — it includes a conceptual UML
"blueprint" (`DataPipeline`, `ModelTrainer`, `Evaluator`, `Predictor`). Those are a
mental model, **not** classes in the codebase; the actual implementation is a
notebook plus a function-based `predict.py`. The learning guide states this itself.

---

## Requirements & Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

| Library | Used for |
|---|---|
| `pandas` | Loading, filtering, cleaning, feature engineering |
| `numpy` | Sentinel/NaN handling |
| `matplotlib` / `seaborn` | EDA charts |
| `scikit-learn` | Logistic Regression, Random Forest, metrics, split, `GridSearchCV`, `TimeSeriesSplit` |
| `xgboost` | The primary / final model |
| `joblib` | Serializing and loading the trained model |

Run the workflow by launching Jupyter from the project root and opening
`wimbledon.ipynb`.

---

## The Dataset

Input is `data/atp_tennis.csv`, an ATP match dataset (2000 onward) with 17 columns
including `Tournament`, `Date`, `Surface`, `Round`, `Player_1`, `Player_2`, `Winner`,
`Rank_1`, `Rank_2`, `Pts_1`, `Pts_2`, and odds.

Data-quality details that the notebook handles explicitly:

- **`Odd_2` loads as `object`/string** (one non-numeric value forces the column to
  string). It is converted with `pd.to_numeric(..., errors="coerce")` during cleaning
  (the notebook prints `Odd_2 dtype before: str` → `after: float64`).
- **`Pts_*` sentinels** are normalized; missing point values are set to `0` before
  computing differences.
- **`Rank_1` / `Rank_2` have no nulls** after filtering (the notebook prints `0` nulls
  for both), so no rows are dropped on rank.

---

## The Pipeline

Every shape below is taken from the committed notebook outputs.

| Step | Operation | Shape / Result |
|---|---|---|
| 1 | Load full ATP CSV | 17 columns |
| 2 | Filter to Wimbledon | `(3061, 17)` |
| 3 | EDA (round counts, charts) | `(3061, 17)` |
| 4 | Clean (`Odd_2` → float, sentinels, `Pts` NaN → 0) | `(3061, 17)`, no rows dropped |
| 5 | Engineer 4 columns (`rank_diff`, `pts_diff`, `round_encoded`, `target`) | `(3061, 21)` |
| 6 | Select 3 features, time-based split | Train `(2197, 3)` · Test `(864, 3)` |
| 7 | Naive baseline | Accuracy **0.688** |
| 8 | Train three models | — |
| 9 | Evaluate all three | see *Models & Results* |
| 10 | Tune XGBoost (`GridSearchCV` + `TimeSeriesSplit`) | Best CV ROC-AUC **0.7735** |
| 11 | Serialize tuned model | round-trip accuracy **0.692** |

### Train / test split

The split is **time-based**, not random. The cutoff is `2018-01-01`: rows with
`Date < 2018-01-01` train (2000-06-26 → 2017-07-16), rows from 2018 onward test
(2018-07-02 → 2025-07-13). That is a ~72/28 split (2197 / 864) and prevents future
matches from leaking into training.

---

## Features

| Feature | Definition | Notes |
|---|---|---|
| `rank_diff` | `Rank_1 - Rank_2` | Negative means Player_1 has the better (lower) rank |
| `pts_diff` | `Pts_1 - Pts_2` | Computed after `Pts` NaNs are filled with 0 |
| `round_encoded` | `Round` mapped to `1–7` | `1st Round = 1` … `The Final = 7` |

The label `target = (Winner == Player_1)` is roughly 50/50 balanced. The round
encoding (identical in the notebook and `predict.py`):

```python
{"1st Round":1, "2nd Round":2, "3rd Round":3, "4th Round":4,
 "Quarterfinals":5, "Semifinals":6, "The Final":7}
```

---

## Models & Results

All three evaluation models are seeded with `random_state=42`, so these numbers are
reproducible from the committed notebook. The naive baseline (predict Player_1 when
`rank_diff < 0`) sets the floor at **0.688**.

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Baseline (rule) | 0.688 | — |
| Logistic Regression | 0.689 | 0.764 |
| Random Forest | 0.660 | 0.730 |
| XGBoost (default) | 0.677 | 0.743 |
| **XGBoost (tuned)** | — | **0.7735** (CV) |

Notes:

- With only three features, Logistic Regression edges out the tree models on the test
  set.
- A quick (unseeded) Random Forest fit reports feature importances of roughly
  `rank_diff` ≈ 0.56, `pts_diff` ≈ 0.41, `round_encoded` ≈ 0.03 — rank difference
  dominates.
- ROC-AUC is reported alongside accuracy and must be computed from positive-class
  probabilities (`predict_proba(...)[:, 1]`), not hard labels.

### Tuning detail

`GridSearchCV` searches a **6-parameter** grid (216 candidates × 5 folds = 1080 fits),
scored on `roc_auc`, using **`TimeSeriesSplit(n_splits=5)`** as the CV so folds respect
temporal order (no leakage):

```python
param_grid = {
    "eta":              [0.05, 0.1, 0.3],   # learning rate
    "max_depth":        [3, 6, 9],
    "min_child_weight": [1, 3, 5],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma":            [0, 0.1],
}
```

Best params: `eta=0.05, max_depth=3, min_child_weight=1, subsample=0.8,
colsample_bytree=0.8, gamma=0`. The tuned model is the one serialized and used by
`predict.py`.

---

## Using `predict.py`

`predict.py` loads the saved model and turns raw match values into a prediction. It
exposes four functions.

### `load_model(model_path="models/xgb_tuned.joblib")`
Loads a model saved with `joblib`.

### `prepare_features(rank_1, rank_2, pts_1, pts_2, round_name)`
Returns `{"rank_diff", "pts_diff", "round_encoded"}`. **Unknown round names map to
`0`** (outside the trained `1–7` range — see *Known Issues*).

### `predict_match(model, rank_diff, pts_diff, round_encoded, threshold=0.5)`
Runs the model on engineered features. Returns:
```python
{"prob_player1_win": float,   # predict_proba[:, 1]
 "predicted_label": int}      # 1 if prob >= threshold else 0
```

### `predict_winner(model, rank_1, rank_2, pts_1, pts_2, round_name)`
Convenience wrapper that engineers features internally and returns:
```python
{"winner": "Player_1" | "Player_2",
 "confidence": float,    # prob of the predicted winner, rounded to 2 dp (always >= 0.5)
 "p1_win_prob": float}   # prob Player_1 wins, rounded to 2 dp
```

### Example
```python
from predict import load_model, predict_winner
model = load_model()
print(predict_winner(model, rank_1=12, rank_2=87, pts_1=3500, pts_2=1200,
                     round_name="Quarterfinals"))
```

Running `python predict.py` executes a built-in example.

> **Column order matters.** The model was trained on `rank_diff, pts_diff,
> round_encoded` in that order, and `predict_proba` column `1` is P(Player_1 wins).

---

## Known Issues

Real behaviors of the current code, kept here so the docs stay honest:

1. **Model-save path is Windows-only.** The notebook saves with a backslash literal,
   `joblib.dump(xgb_tuned, r"models\xgb_tuned.joblib")`, while `predict.py` loads
   `"models/xgb_tuned.joblib"` (forward slash). On Windows these resolve to the same
   place; on macOS/Linux the backslash version writes a file literally named
   `models\xgb_tuned.joblib` in the working directory rather than into a `models/`
   folder, so `predict.py`'s default load will not find it. **Fix:** save with a
   forward slash or `os.path.join("models", "xgb_tuned.joblib")`.

2. **No input validation in `prepare_features`.** Any unrecognized `round_name` maps
   to `0`, which is outside the trained `1–7` range and silently yields an unreliable
   prediction rather than raising.

3. **`models/` is not committed.** A fresh clone must run the notebook through Step 11
   to produce `xgb_tuned.joblib` before `predict.py` can load anything.

4. **Some hyperparameters aren't seeded.** The evaluation models in Step 9 use
   `random_state=42` (reproducible), but the quick Random Forest used for feature
   importances in Step 8 is unseeded, so those importances vary slightly per run.

> A previous version of this README listed a `predict.py` typo of
> `"models/xbg_tuned.joblib"` (`xbg`). That has been verified against the current code
> and is **not** present — both the default argument and the `__main__` example use
> the correct `"models/xgb_tuned.joblib"`. The real path issue is the
> backslash/forward-slash mismatch in item 1.
