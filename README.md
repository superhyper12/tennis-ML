# Wimbledon Match Outcome Predictor — Documentation

A machine learning project that predicts the outcome of Wimbledon tennis matches.
Given two players' rankings, ranking points, and the tournament round, the model
estimates the probability that **Player_1** beats **Player_2**.

This is a binary classification problem: the output is either **Player_1 wins (1)**
or **Player_2 wins (0)**.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Requirements & Installation](#requirements--installation)
4. [The Dataset](#the-dataset)
5. [The Pipeline](#the-pipeline)
6. [Features](#features)
7. [Models & Results](#models--results)
8. [Using `predict.py`](#using-predictpy)
9. [Known Issues](#known-issues)

---

## Overview

The project is built in two parts:

- **`wimbledon.ipynb`** — a single Jupyter notebook containing the full workflow
  from loading raw data through training, evaluating, tuning, and saving the model
  (organized as Steps 1–11).
- **`predict.py`** — a small standalone module that loads the saved model and
  exposes a reusable prediction function (Step 12).

The trained model is the handoff between the two: the notebook serializes the best
model to `models/xgb_tuned.joblib`, and `predict.py` loads it to make predictions on
new matchups.

> **Note on the existing README:** the repository's `README.md` is a *learning guide*
> that walks through the 12 build steps with external tutorial links and expected
> outputs. This document is the *technical reference* for the project as it actually
> exists in code.

---

## Project Structure

```
wimbledon-predictor/
├── Learning_Resource.md
├── data/
│   └── atp_tennis.csv      # ATP match dataset (placed here manually)
├── models/
│   └── xgb_tuned.joblib    # Trained model, written by the notebook (~200 KB)
├── wimbledon.ipynb         # Full workflow, Steps 1–11
├── predict.py              # Prediction function, Step 12
└── README.md               # Step-by-step learning guide
```

The `models/` folder starts empty and is populated when the notebook reaches the
save step. The `data/` directory holds the input CSV, which is added manually rather
than committed.

---

## Requirements & Installation

The project uses the standard Python data/ML stack.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

| Library        | Used for                                          |
| -------------- | ------------------------------------------------- |
| `pandas`       | Loading, filtering, cleaning, feature engineering |
| `numpy`        | Array operations and NaN handling                 |
| `matplotlib`   | Bar charts and histograms during EDA              |
| `seaborn`      | Distribution plots during EDA                     |
| `scikit-learn` | Logistic Regression, Random Forest, evaluation, splitting, tuning |
| `xgboost`      | The primary (final) model                         |
| `joblib`       | Serializing and loading the trained model         |

To run the workflow, launch Jupyter from the project root and open the notebook:

```bash
jupyter notebook
```

---

## The Dataset

The input is an ATP tennis match dataset (`data/atp_tennis.csv`) covering matches
from 2000 onward. It has 17 columns including `Tournament`, `Date`, `Surface`,
`Round`, the two players (`Player_1`, `Player_2`), the `Winner`, each player's rank
(`Rank_1`, `Rank_2`), ranking points (`Pts_1`, `Pts_2`), and betting odds
(`Odd_1`, `Odd_2`).

A few data-quality details matter:

- **No true NaNs on load.** Missing values are stored as a sentinel `-1` in the
  `Pts_*` and `Odd_*` columns (used for earlier years where data wasn't collected).
  These must be replaced with `NaN` before feature engineering, or they will corrupt
  the difference calculations.
- **`Odd_2` loads as `object`** instead of `float64` because one row contains a
  non-numeric value, forcing the whole column to string type. It is converted with
  `pd.to_numeric(..., errors='coerce')`.
- **`Rank_1` / `Rank_2` are always populated** — no nulls or sentinels — so no rows
  are dropped on rank.
- After filtering to Wimbledon, `Surface` is always `"Grass"`.

---

## The Pipeline

The notebook transforms the data through a fixed sequence of steps. The table below
tracks how the data shape changes:

| Step | Operation                              | Shape / State                          |
| ---- | -------------------------------------- | -------------------------------------- |
| 1    | Load full ATP CSV                      | `(67460, 17)`                          |
| 2    | Filter to Wimbledon only               | `(3061, 17)`                           |
| 3    | Exploratory data analysis (charts)     | `(3061, 17)`                           |
| 4    | Clean data (sentinels → NaN, types)    | `(3061, 17)` (no rows dropped)         |
| 5    | Add 4 engineered columns               | `(3061, 21)`                           |
| 6    | Select 3 features, split by date       | Train `(2197, 3)` · Test `(864, 3)`    |
| 7    | Naive baseline                         | Accuracy ≈ 0.688                       |
| 8    | Train three models                     | —                                      |
| 9    | Evaluate all three                     | See [results](#models--results)        |
| 10   | Tune XGBoost (GridSearchCV)            | Best CV ROC-AUC ≈ 0.777                |
| 11   | Serialize best model                   | `models/xgb_tuned.joblib`              |

### Train/test split

The split is **time-based**, not random — everything before 2018 trains, everything
from 2018 onward tests (roughly a 72/28 split). This is critical: a random split
would leak future matches into training and inflate accuracy. The latest training
date must be strictly earlier than the earliest test date.

---

## Features

From the cleaned data, three features are engineered and used for modeling:

| Feature         | Definition                            | Notes                                              |
| --------------- | ------------------------------------- | -------------------------------------------------- |
| `rank_diff`     | `Rank_1 - Rank_2`                     | Negative means Player_1 has the better (lower) rank |
| `pts_diff`      | `Pts_1 - Pts_2`                       | Sentinels replaced and filled before computing      |
| `round_encoded` | Round mapped to an ordinal `1–7`      | `1st Round = 1` … `The Final = 7`                  |

The label `target` is `1` when `Winner == Player_1` and `0` otherwise. Because the
dataset records real match pairings (both players present), the target is balanced at
roughly 50/50.

The round encoding used in both the notebook and `predict.py`:

```python
{
    "1st Round": 1,
    "2nd Round": 2,
    "3rd Round": 3,
    "4th Round": 4,
    "Quarterfinals": 5,
    "Semifinals": 6,
    "The Final": 7,
}
```

---

## Models & Results

Three models are trained on the same three features and evaluated on the held-out
test set. The naive baseline (predict Player_1 wins when `rank_diff < 0`) sets the
floor at ~0.688 accuracy.

| Model               | Accuracy | ROC-AUC |
| ------------------- | -------- | ------- |
| Baseline (rule)     | 0.688    | —       |
| Logistic Regression | 0.689    | 0.764   |
| Random Forest       | 0.664    | 0.729   |
| XGBoost (default)   | 0.684    | 0.755   |
| **XGBoost (tuned)** | —        | **0.777** (CV) |

A few observations:

- With only three features, Logistic Regression's regularization is competitive with
  the tree-based methods.
- Random Forest feature importances confirm `rank_diff` is the dominant signal
  (~0.56), `pts_diff` contributes meaningfully (~0.40), and `round_encoded` is weak
  (~0.04).
- The **tuned XGBoost model is the one saved to disk** and used by `predict.py`.
  Tuning is done with `GridSearchCV` over `learning_rate`, `max_depth`, and
  `subsample`, scored with cross-validation (ROC-AUC). For time-series data,
  `TimeSeriesSplit` is the correct CV strategy to avoid leakage.

ROC-AUC is reported alongside accuracy because it measures how well the model *ranks*
positive examples above negative ones, which is more informative than accuracy alone.
When computing it, pass the positive-class probabilities (`predict_proba(...)[:, 1]`),
not hard labels.

---

## Using `predict.py`

`predict.py` loads the saved model and turns raw match values into a prediction. It
exposes four functions.

### `load_model(model_path="models/xgb_tuned.joblib")`

Loads a model previously saved with `joblib`.

### `prepare_features(rank_1, rank_2, pts_1, pts_2, round_name)`

Converts raw match values into the three model input features, returning a dict with
`rank_diff`, `pts_diff`, and `round_encoded`. Unknown round names map to `0`.

### `predict_match(model, rank_diff, pts_diff, round_encoded, threshold=0.5)`

Runs the model on already-engineered features. Returns:

```python
{
    "prob_player1_win": float,   # probability from predict_proba[:, 1]
    "predicted_label": int,      # 1 if prob >= threshold, else 0
}
```

### `predict_winner(model, rank_1, rank_2, pts_1, pts_2, round_name)`

The main convenience function. Takes raw values, engineers the features internally,
and returns a human-readable result:

```python
{
    "winner": "Player_1" | "Player_2",
    "confidence": float,     # probability of the predicted winner, rounded to 2 dp
    "p1_win_prob": float,    # probability Player_1 wins, rounded to 2 dp
}
```

`confidence` is the probability assigned to whichever player is predicted to win, so
it is always ≥ 0.5. A confidence near 0.5 for two closely-ranked players is correct
behavior — the model is appropriately uncertain.

### Example

```python
from predict import load_model, predict_winner

model = load_model()  # loads models/xgb_tuned.joblib

result = predict_winner(
    model,
    rank_1=12,
    rank_2=87,
    pts_1=3500,
    pts_2=1200,
    round_name="Quarterfinals",
)
print(result)
# {'winner': 'Player_1', 'confidence': 0.79, 'p1_win_prob': 0.79}
```

Running the module directly executes a built-in example:

```bash
python predict.py
```

> **Inference detail:** the model was trained on columns named `rank_diff`,
> `pts_diff`, and `round_encoded` in that order. Any input passed at prediction time
> must preserve those names and that order, and `predict_proba` column `1` (not `0`)
> is the probability that Player_1 wins.

---

## Known Issues

A few inconsistencies exist in the current code that are worth fixing:

1. **Misspelled default model path.** In `predict.py`, both `load_model`'s default
   argument and the `__main__` example load `"models/xbg_tuned.joblib"` — note
   `xbg` instead of `xgb`. The notebook saves the model as `xgb_tuned.joblib`. As
   written, the default will fail to find the file. Either rename the saved model or
   correct the path to `"models/xgb_tuned.joblib"`.

2. **No input validation.** `prepare_features` maps any unrecognized `round_name` to
   `0`, which is outside the trained `1–7` range and will silently produce unreliable
   predictions rather than raising an error.

3. **README vs. code.** The README references a UML "blueprint" with `DataPipeline`,
   `ModelTrainer`, `Evaluator`, and `Predictor` classes. The actual implementation is
   a notebook plus the function-based `predict.py` — there are no such classes. The
   README is best read as a conceptual learning map, not a literal description of the
   code.
