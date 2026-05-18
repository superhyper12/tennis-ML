# 📚 Learning Resources by Step
## Wimbledon Match Outcome Predictor

> **How to use this:** Don't try to read everything upfront. Work through one step at a time. Each resource section tells you exactly what to learn, what to focus on, and why it matters before you write that step's code.

---

## 🗂️ Project Setup

### What You Actually Need

The full multi-file layout you see in architecture diagrams is production-style — useful for apps and shared packages, but overkill for a solo learning project. Here's what's actually necessary:

| Item | Necessary? | Why |
|---|---|---|
| `data/atp_tennis.csv` | ✅ Yes | You need the data somewhere |
| `models/` folder | ✅ Yes | One place to save your trained model |
| `wimbledon.ipynb` | ✅ Yes | Where all your actual work happens (Steps 1–11) |
| `predict.py` | ✅ At Step 12 only | A reusable function needs its own file — create it then, not now |
| Multiple notebooks | ❌ No | One notebook with clear section headers is cleaner and faster |
| `src/` package folder | ❌ No | Only needed if you're building an app or importing your own modules |
| `requirements.txt` | ⚠️ Optional | Useful if sharing the project, otherwise not needed |

### Minimal Layout

```
wimbledon-predictor/
├── data/
│   └── atp_tennis.csv      # Drop the dataset here manually
├── models/                 # Empty for now — Step 11 saves here
├── wimbledon.ipynb         # One notebook for Steps 1–11
└── predict.py              # Create this at Step 12, not before
```

### How to Create It (3 commands)

Open your terminal, navigate to wherever you want the project to live, then run:

```bash
mkdir -p wimbledon-predictor/data wimbledon-predictor/models
cd wimbledon-predictor
jupyter notebook
```

Jupyter will open in your browser. Create a new notebook, name it `wimbledon.ipynb`, and you're ready to start Step 1. Drop `atp_tennis.csv` into the `data/` folder manually.

> `predict.py` doesn't exist yet — that's intentional. Create it when you reach Step 12.

### Keeping the Notebook Clean

Since one notebook handles all 12 steps, use **markdown cells as dividers** between sections. Add a markdown cell at the top of each step like:

```
## Step 1 — Load and Inspect the Raw Data
```

This keeps the notebook scrollable and readable without splitting it into separate files.

### Installing Dependencies

Run this once in your terminal before opening the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

---

## 🗺️ Architecture Overview

Two diagrams live above this section in the README as rendered widgets. Here is what each one shows:

**Pipeline overview** — a top-down flowchart of all 12 steps. Teal nodes run inside `wimbledon.ipynb`. The amber artifact (`xgb_tuned.joblib`) is the handoff between the notebook and `predict.py`. The purple block is `predict.py`. Steps are clickable — each one links back to its section below.

**UML blueprint** — a class diagram showing the four logical units of the project (`DataPipeline`, `ModelTrainer`, `Evaluator`, `Predictor`), their method signatures, and the relationships between them. It shows *what gets built and how the pieces connect*, not how any of it is implemented. Use it as a mental map before writing each step's code.

### Data Shape Through the Pipeline

Each step transforms the data in a specific way. This table shows exactly what changes and why.

| Step | Operation | Shape / State |
|---|---|---|
| Raw CSV loaded | Full ATP dataset | `(67460, 17)` |
| Step 2 | Filter to Wimbledon only | `(3061, 17)` |
| Step 4 | No rows dropped (ranks have no nulls) | `(3061, 17)` |
| Step 5 | Add 4 engineered columns | `(3061, 21)` |
| Step 6 | Select 3 features, split by date | Train: `(2197, 3)` · Test: `(864, 3)` |
| Step 11 | Serialize best model to disk | `models/xgb_tuned.joblib` (~200 KB) |

---

## Before You Start

### Concept: What is a binary classifier?

You need a mental model of what the model is actually doing before writing any code. Binary classification means the output is one of two classes — in this case, **Player_1 wins (1)** or **Player_2 wins (0)**.

**[Google ML Crash Course — Framing](https://developers.google.com/machine-learning/crash-course/framing/video-lecture)**
Read the "Framing" and "Descending into ML" sections only. You need to understand three terms before anything else: *label* (what you're predicting — the match winner), *feature* (the inputs — rank, points, round), and *model* (the function that maps features to a label). Every step in this project maps to one of those three concepts.

**[Scikit-learn — Classification tutorial](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)**
Skim the visual output, not the code. The goal is to see that different classifiers produce different decision boundaries — and to build intuition that the "right" model depends on the shape of your data, not just habit.

### Concept: Jupyter Notebooks

**[Jupyter — Quickstart](https://docs.jupyter.org/en/latest/start/index.html)**
Focus on: how to run a cell, how to restart the kernel, and how to add markdown cells for notes. You don't need to know anything beyond these three things. The important habit to build is running cells top-to-bottom — running them out of order is the #1 source of confusing bugs in EDA notebooks.

---

## Step 1 — Load and Inspect the Raw Data

### ✅ Expected Output

```
(67460, 17)

Tournament     object
Date           object
Series         object
Court          object
Surface        object
Round          object
Best of         int64
Player_1       object
Player_2       object
Winner         object
Rank_1          int64
Rank_2          int64
Pts_1           int64
Pts_2           int64
Odd_1         float64
Odd_2          object      ← loads as object due to one non-numeric value; fixed in Step 4
Score          object
dtype: object

Pts_1    0
Pts_2    0
dtype: int64
```

`df.head()` returns 5 rows with columns like `Tournament`, `Surface`, `Player_1`, `Rank_1`, and `Odd_1`.

> **Note:** There are no true NaN values on load — `Rank_1`, `Rank_2`, `Pts_1`, and `Pts_2` use `-1` as a sentinel for missing data. You'll replace those with `NaN` in Step 4.

### 📖 What to Learn Before Writing This Step

**[pandas Getting Started Tutorials — Tutorials 1, 2, and 3](https://pandas.pydata.org/docs/getting_started/intro_tutorials/)**
These three short tutorials cover everything this step needs. Tutorial 1 teaches what a DataFrame is (rows = matches, columns = attributes). Tutorial 2 teaches `read_csv()`, which is the one function you need to load the data. Tutorial 3 teaches how to select specific columns — you'll use this pattern in every step that follows. Don't read past Tutorial 3 yet.

**[Real Python — pandas DataFrames 101](https://realpython.com/pandas-dataframe/)**
Read this if the official docs feel too dry. The key thing to understand before writing Step 1: the difference between `.shape` (how many rows and columns exist), `.dtypes` (what type each column is), and `.isnull().sum()` (how many values are missing per column). These three inspections together give you a complete picture of a dataset you've never seen before.

**[pandas docs — dtypes](https://pandas.pydata.org/docs/user_guide/basics.html#dtypes)**
Read just the first section. The specific thing to understand: why `Odd_2` loads as `object` instead of `float64` even though it looks like a number. This happens because one row contains a non-numeric value, so pandas flags the entire column as `object`. You'll fix this in Step 4, but you need to understand why it happens now.

---

## Step 2 — Filter to Wimbledon Only

### ✅ Expected Output

```
(3061, 17)
```

The filtered DataFrame contains only rows where `Tournament` is `"Wimbledon"`. Spot-check that the `Surface` column reads `"Grass"` for every row — no exceptions.

### 📖 What to Learn Before Writing This Step

**[pandas docs — Boolean indexing](https://pandas.pydata.org/docs/user_guide/indexing.html#boolean-indexing)**
Read the "Boolean indexing" section and the "Working with text data — `str.contains`" subsection. The concept to understand: filtering rows in pandas works by creating a True/False mask the same length as the DataFrame, then using that mask to select only the rows where the condition is True. You need `str.contains("Wimbledon")` specifically because tournament names sometimes have slight variations — an exact `==` match would miss those. In this dataset the name is always exactly `"Wimbledon"`, but `str.contains` is safer practice.

**[Corey Schafer — Pandas Tutorial Part 2](https://www.youtube.com/watch?v=zmdjNSmRXF4)**
Watch the first 20 minutes. This is the clearest video walkthrough of filtering rows with conditions. Pay attention to how he chains multiple conditions with `&` and `|` — you won't need that here, but it comes back in Step 4 when you replace sentinel values across multiple columns at once.

---

## Step 3 — Exploratory Data Analysis

### ✅ Expected Output

> This step produces **visual output** (charts rendered inline in Jupyter), not printed console text. There is no block of numbers to compare against — success means the charts look right and the groupby table prints correctly.

**Console output — match counts per round:**

```
Round
1st Round      1539
2nd Round       769
3rd Round       390
4th Round       192
Quarterfinals    98
Semifinals       48
The Final        25
Name: count, dtype: int64
```

**Chart 1 — bar chart (matches per round):** 7 bars render in the notebook. `1st Round` is the tallest bar. `The Final` is the shortest — only one final is played per year. The bars decrease as rounds progress toward the final.

**Chart 2 — histogram (winner rank distribution, using `Rank_1`):** A right-skewed distribution renders. The bulk of the mass clusters below rank 50 with a long tail stretching toward rank 1000+. The shape confirms rank is a useful predictor — lower-ranked (better) players win more often, though the 50/50 target split means rank alone doesn't dominate.

**Key insights to record in your notebook:**
- The dataset contains matches from 2000 onward. `Rank_1` and `Rank_2` are always populated (no nulls), but `Pts_1`, `Pts_2`, `Odd_1`, and `Odd_2` use `-1` as a placeholder for earlier years.
- `Surface` is always `"Grass"` — no filtering needed on that column downstream.
- `target` will be balanced (roughly 50/50) because the dataset records matches with both players present, not just winners.

### 📖 What to Learn Before Writing This Step

**[Towards Data Science — Comprehensive EDA with Python](https://towardsdatascience.com/exploratory-data-analysis-with-python-3887b1e2b2e0)**
Read Sections 1–4 only (up through distribution plots). The most important idea here isn't a function — it's a mindset: EDA is about asking specific questions and answering them visually before you touch any model. The two questions for this step are "which rounds have the most matches?" and "what does the rank distribution look like?" If you skip EDA and jump straight to modeling, you'll miss obvious data quality issues that break everything later.

**[matplotlib — Pyplot Tutorial](https://matplotlib.org/stable/tutorials/introductory/pyplot.html)**
Read through the bar chart and histogram examples only. The one thing to understand: `plt.show()` must be called after each plot or nothing renders. Also note that `plt.title()`, `plt.xlabel()`, and `plt.ylabel()` are called *before* `plt.show()` — this trips up almost everyone the first time.

**[seaborn — Introduction](https://seaborn.pydata.org/introduction.html)**
Read just the intro and the "Distributional representation" section. Seaborn's `histplot()` produces much cleaner histograms than matplotlib's `hist()` — specifically because it handles bin sizing automatically and overlays a KDE (density curve) by default. You'll use it for the rank distribution plot.

**[pandas docs — GroupBy](https://pandas.pydata.org/docs/user_guide/groupby.html)**
Read the "Splitting an object into groups" and "Aggregation" sections only. The concept to understand: `groupby("Round").size()` splits the DataFrame into one group per round, then counts the rows in each group. This is the pattern behind almost every "win rate by X" question you'll ask in EDA.

---

## Step 4 — Clean the Data

### ✅ Expected Output

```
(3061, 17)

Rank_1    0
Rank_2    0
dtype: int64

Odd_2 dtype before: object
Odd_2 dtype after:  float64
```

> **Note:** Shape stays at 3061 — `Rank_1` and `Rank_2` have no nulls or `-1` sentinel values, so no rows are dropped. The `-1` sentinel *does* appear in `Pts_1`, `Pts_2`, `Odd_1`, and `Odd_2` for earlier years — replace those with `NaN` using `.replace(-1, np.nan)` on each column before feature engineering. `Odd_2` also has one non-numeric string entry; use `pd.to_numeric(..., errors='coerce')` to convert it.

### 📖 What to Learn Before Writing This Step

**[pandas docs — Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)**
Read the first three sections: detecting missing values (`isnull`), removing them (`dropna`), and replacing values (`replace`). The key distinction here: `dropna()` by default removes any row with *any* null in *any* column. In this project you don't need to drop any rows on rank, but you still need to handle sentinel `-1` values in `Pts_1`, `Pts_2`, and `Odd_*` columns.

**[Real Python — Handling Missing Data](https://realpython.com/python-data-cleaning-numpy-pandas/)**
Read the "Handling Missing Values" and "Replacing Values" sections. The specific concept you need: the difference between a true `NaN` (genuinely missing) and a sentinel value like `-1` (a placeholder used when data wasn't collected). If you don't replace `-1` in points columns before computing `pts_diff` in Step 5, the model will treat those as real point values, silently corrupting your features.

**[pandas docs — Time series / date functionality](https://pandas.pydata.org/docs/user_guide/timeseries.html)**
Read just the "Parsing time series information" section at the top. The `Date` column loads as strings in `YYYY-MM-DD` format. Converting it to a proper datetime with `pd.to_datetime(df["Date"])` is what makes the time-based train/test split in Step 6 possible.

---

## Step 5 — Feature Engineering

### ✅ Expected Output

```
         rank_diff       pts_diff  round_encoded  target
count  3061.000000   3061.000000    3061.000000  3061.0
mean      1.056191     16.324404       1.949690     0.5
std     132.932265   2705.543989       1.273697     0.5
min   -1060.000000 -16641.000000       1.000000     0.0
25%     -50.000000   -495.000000       1.000000     0.0
50%      -1.000000      0.000000       1.000000     1.0
75%      50.000000    473.000000       2.000000     1.0
max    1038.000000  16070.000000       7.000000     1.0
```

`rank_diff` can be positive or negative — it is `Rank_1 - Rank_2`, so a negative value means Player_1 has the better (lower) rank number. `target` has mean ~0.5 because the dataset records actual match pairings (not just winner rows) and the label is `1` when `Winner == Player_1`, `0` when `Winner == Player_2`. `round_encoded` runs 1–7, with `1st Round=1` up to `The Final=7`.

> **Note on pts_diff:** Replace `-1` sentinel values with `NaN` first, then fill `NaN` with `0` before computing `pts_diff`. This avoids the sentinel corrupting the difference calculation while still including all 3061 rows.

### 📖 What to Learn Before Writing This Step

**[Google ML Crash Course — Feature Engineering](https://developers.google.com/machine-learning/crash-course/representation/feature-engineering)**
Read "Representation: Feature Engineering" and "Qualities of Good Features." The core idea: raw columns like `Rank_1` and `Rank_2` are less useful to a model than the *relationship* between them (`rank_diff = Rank_1 - Rank_2`). A model given both raw ranks has to discover the subtraction relationship on its own — computing it explicitly makes the pattern obvious and improves performance. This is the reasoning behind every new column you create in this step.

**[pandas docs — Assignment](https://pandas.pydata.org/docs/user_guide/indexing.html#returning-a-view-versus-a-copy)**
Read the "Returning a view versus a copy" section. The specific gotcha: when you write `df["new_col"] = expression` on a filtered slice of a DataFrame (which `wimbledon` is), pandas may raise a `SettingWithCopyWarning`. The fix is calling `.copy()` when you create the filtered DataFrame in Step 2 — doing it there prevents the warning from appearing in every step after it.

**[scikit-learn docs — LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)**
Skim the overview section only. Round names like `"Quarterfinals"` and `"Semifinals"` are strings, but models only understand numbers. In this project you'll use a hand-crafted dictionary map rather than `LabelEncoder` directly — but understanding what label encoding *is* and why the order matters (`The Final=7 > Semifinals=6 > Quarterfinals=5...`) is what makes `round_encoded` a meaningful ordinal feature rather than arbitrary numbers.

---

## Step 6 — Train / Test Split

### ✅ Expected Output

```
X_train shape: (2197, 3)
X_test shape:  (864, 3)

Features: ['rank_diff', 'pts_diff', 'round_encoded']
Train date range: 2000-06-26 → 2017-07-16
Test date range:  2018-07-02 → 2025-07-13
```

The 3 columns in `X_train` and `X_test` are `rank_diff`, `pts_diff`, and `round_encoded` — the features engineered in Step 5. `y_train` and `y_test` contain the `target` column (0 or 1). The split is roughly 72/28 by date: everything before 2018 trains, everything from 2018 onward tests. The latest date in the training set must be strictly before the earliest date in the test set — if not, you have data leakage.

### 📖 What to Learn Before Writing This Step

**[Google ML Crash Course — Training and Test Sets](https://developers.google.com/machine-learning/crash-course/training-and-test-sets/video-lecture)**
Read this before writing a single line. The concept to internalize: a model's accuracy on the data it was *trained on* tells you nothing useful — it has seen those examples before. Accuracy on the *test set* (data the model has never seen) tells you whether the model has learned a general pattern or just memorized the training examples. This is why the split exists.

**[Towards Data Science — Data Leakage in Machine Learning](https://towardsdatascience.com/data-leakage-in-machine-learning-6161c167e8ba)**
Read the full article — it's short and this is the most dangerous mistake you can make in this step. The specific risk here: if you use a random split instead of a time-based split, some 2010 matches end up in the test set and some 2020 matches end up in training. The model can then effectively "see the future," producing inflated accuracy numbers that fall apart on real predictions. The fix is splitting strictly on date: everything before 2018 trains, everything after tests.

**[Kaggle — Data Leakage (Intermediate ML, Lesson 5)](https://www.kaggle.com/learn/intermediate-machine-learning)**
Work through Lesson 5 interactively. This is the clearest worked example of leakage available — Kaggle walks you through a case where a random split looks great and a time-based split reveals the model was cheating. Takes about 20 minutes and makes the concept concrete.

---

## Step 7 — Naive Baseline

### ✅ Expected Output

```
Baseline accuracy: 0.688
```

~69% is your floor. Any model that can't beat this is not worth deploying. The baseline rule is: predict Player_1 wins if `rank_diff < 0` (Player_1 has the better rank). The goal of Steps 8–10 is to meaningfully exceed this number.

> **Note:** Unlike the original ATP dataset where winners were always listed first, this dataset has real match pairings — both players are present and `target` is 50/50 overall. The baseline captures the real signal that better-ranked players win more often, but it's a meaningful challenge to beat on the test set.

### 📖 What to Learn Before Writing This Step

**[Google ML Crash Course — Generalization](https://developers.google.com/machine-learning/crash-course/generalization/video-lecture)**
Read the "Peril of Overfitting" section. The concept to understand: a model that scores 75% accuracy might sound impressive, but if a simple rule gets 69% for free, your model is only adding 6% of real value. Baselines calibrate what "good" actually means.

**[sklearn docs — accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)**
Read the function signature and the first example only. For the baseline, your "predicted" labels come from a simple rule (`rank_diff < 0`), not a trained model. This is intentional — it shows that even a rule with zero learning beats random guessing by a wide margin.

---

## Step 8 — Train Three Models

### ✅ Expected Output

All three models finish `.fit()` without errors. With `verbose=False` (the default), no training output prints.

Random Forest feature importances after training:

```
rank_diff        0.56   ← strongest predictor
pts_diff         0.40
round_encoded    0.04
```

`rank_diff` dominates, which makes sense — rank is the most reliable signal in this dataset. `pts_diff` contributes meaningfully. `round_encoded` is a weak predictor here because the dataset already controls for round structure implicitly.

### 📖 What to Learn Before Writing This Step

**Logistic Regression**

**[StatQuest — Logistic Regression (YouTube, 8 min)](https://www.youtube.com/watch?v=yIYKR4sgzI8)**
Watch this first, before reading any docs. It explains *why* logistic regression outputs a probability between 0 and 1 rather than a raw number — using the sigmoid function to squash linear output. If you skip this and go straight to the sklearn docs, the `predict_proba` output in Step 12 won't make sense.

**[scikit-learn — StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)**
Read the "Notes" section. The key reason you must scale before Logistic Regression: the algorithm treats all features as equally important by default. If `pts_diff` ranges from -16000 to +16000 and `round_encoded` ranges from 1 to 7, the model will overweight points differences just because the numbers are bigger — not because they're more predictive. Scaling puts all features on the same scale and removes that distortion.

**[scikit-learn — Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)**
Read the "Notes" and first example. The critical concept: a Pipeline chains the scaler and the classifier into one object so they can't be applied in the wrong order. Without a Pipeline, it's easy to accidentally scale the test data using test data statistics — another form of leakage. With a Pipeline, `fit()` learns the scale from training data only, and `transform()` on the test set uses those same training statistics.

**Random Forest**

**[StatQuest — Random Forests (YouTube, 9 min)](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)**
Watch before reading the docs. The intuition: a single decision tree memorizes the training data and fails on new data (overfitting). A random forest fixes this by building hundreds of trees on random subsets of the data and averaging their predictions — the randomness prevents any one tree from dominating.

**[scikit-learn — RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)**
Read the parameter descriptions for `n_estimators`, `max_depth`, and `feature_importances_`. You need to understand `feature_importances_` specifically — it tells you which of your three features the forest relied on most, which validates whether your feature engineering in Step 5 was actually useful.

**XGBoost**

**[StatQuest — XGBoost Part 1 (YouTube, 16 min)](https://www.youtube.com/watch?v=OtD8wVaFm6E)**
Watch Part 1 only for now. The key difference from Random Forest: XGBoost builds trees *sequentially*, where each new tree is trained specifically on the mistakes of the previous trees. This is why it typically outperforms Random Forest on structured tabular data — it focuses effort where the current model is weakest.

**[XGBoost docs — Early Stopping](https://xgboost.readthedocs.io/en/stable/python/callbacks.html)**
Read the Early Stopping section. XGBoost trains up to `n_estimators` trees, but adding more trees past a certain point causes the model to overfit. Early stopping monitors performance on the validation set during training and stops automatically when performance stops improving.

---

## Step 9 — Evaluate All Three Models

### ✅ Expected Output

```
Logistic Regression:  Accuracy=0.689  ROC-AUC=0.764
[[285 147]
 [122 310]]

Random Forest:        Accuracy=0.664  ROC-AUC=0.729
[[281 151]
 [139 293]]

XGBoost:              Accuracy=0.684  ROC-AUC=0.755
[[294 138]
 [135 297]]
```

All three beat or match the 0.688 baseline, with Logistic Regression narrowly leading on both accuracy and AUC. This is not unusual with only 3 features — LR's regularization can outperform tree methods on simple, well-scaled feature sets. Reading the LR confusion matrix: 285 matches correctly predicted as Player_2 wins (TN), 310 as Player_1 wins (TP), 147 false positives, 122 false negatives. ROC-AUC above 0.75 with 3 features is a solid result.

> **Note:** The XGBoost accuracy here (0.684) is from default parameters trained in Step 8. Step 10 tunes these parameters — the tuned model is what gets saved in Step 11.

### 📖 What to Learn Before Writing This Step

**[Google ML Crash Course — Classification](https://developers.google.com/machine-learning/crash-course/classification/video-lecture)**
Read the "Accuracy," "Precision and Recall," and "ROC Curve and AUC" sections. The key insight: accuracy alone is misleading when one class is more common than the other. ROC-AUC measures whether the model ranks positive examples higher than negative ones — it stays meaningful even if class distribution is uneven. An AUC of 0.5 means random guessing; 1.0 means perfect ranking; above 0.75 is practically useful.

**[StatQuest — ROC and AUC (YouTube, 16 min)](https://www.youtube.com/watch?v=4jRBRDbJemM)**
Watch this if the ROC curve concept isn't clicking from text alone. StatQuest builds the curve step-by-step from scratch using a tiny example — by the end you'll understand exactly what the x and y axes represent (false positive rate and true positive rate) and why the area under the curve is a useful single-number summary.

**[scikit-learn — Confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)**
Read the function reference and the example. Understand the 2×2 layout: top-left = true negatives (correctly predicted Player_2 wins), top-right = false positives (wrongly predicted Player_1), bottom-left = false negatives (missed Player_1 wins), bottom-right = true positives (correctly predicted Player_1 wins). The confusion matrix tells you *where* the model fails — not just *how often* — which is critical when comparing models.

**[scikit-learn — roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)**
Read the function signature only. The one thing to know: `roc_auc_score` requires *probabilities* (from `predict_proba`), not hard class labels (from `predict`). Specifically, you pass the probability of the *positive class*, which is column index `[:, 1]` of the `predict_proba` output. Passing the wrong column index is a silent bug — the score will be wrong without any error message.

---

## Step 10 — Tune XGBoost Hyperparameters

### ✅ Expected Output

```
Fitting 5 folds for each of 27 candidates, totalling 135 fits

Best params: {'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.7}
Best CV ROC-AUC: 0.777
```

The tuned parameters may differ from the defaults used in Step 8 — use whatever `grid.best_params_` returns. The important thing is that you use `grid.best_estimator_` in Step 11, not the manually-trained XGBoost from Step 8.

### 📖 What to Learn Before Writing This Step

**[scikit-learn — GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)**
Read the parameter descriptions for `param_grid`, `cv`, `scoring`, and `n_jobs`. The concept: `GridSearchCV` tries every combination of values in `param_grid` and scores each one using cross-validation. With 3 values for each of 3 parameters, that's 27 combinations × 5 folds = 135 total training runs. Setting `n_jobs=-1` runs these in parallel using all available CPU cores — without it, this step takes significantly longer.

**[scikit-learn — TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)**
Read this carefully and look at the diagram of how folds are structured. The problem with standard k-fold on time-series data: it shuffles the data before splitting, so fold 1 might train on 2019 data and validate on 2015 data — that's future-training-on-past, which is leakage. `TimeSeriesSplit` always trains on older data and validates on newer data, mirroring the real-world use case where you predict future matches from historical ones.

**[Towards Data Science — Cross-validation for Time Series](https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9)**
Read the first half (through the `TimeSeriesSplit` section). This gives a concrete example of how using standard k-fold on time-ordered data inflates your validation scores — and how much they drop when you switch to `TimeSeriesSplit`.

---

## Step 11 — Save the Trained Model

### ✅ Expected Output

```
['models/xgb_tuned.joblib']

Round-trip accuracy check: 0.690
```

The first line is the return value of `joblib.dump()` — it confirms the path where the file was written. The `models/` folder now contains `xgb_tuned.joblib`. The round-trip accuracy comes from `grid.best_estimator_` evaluated on the test set — it must match exactly before and after loading. If it doesn't, the most common cause is a version mismatch between the sklearn/xgboost version used to train and the version used to load.

### 📖 What to Learn Before Writing This Step

**[scikit-learn — Model persistence](https://scikit-learn.org/stable/model_persistence.html)**
Read the full page — it's short. The key concept: a trained model is just a Python object with learned weights stored in memory. `joblib.dump()` serializes that object to disk so you can reload it later without retraining. The page also explains why `joblib` is preferred over Python's built-in `pickle` for ML models — joblib handles large numpy arrays (which store the tree weights) more efficiently.

**[Real Python — Saving and Loading ML Models](https://realpython.com/python-pickle-module/)**
Read the "Using joblib for Scikit-Learn Objects" section. Always save `grid.best_estimator_` (the tuned model from Step 10), not the original `xgb` object from Step 8. If you accidentally save the wrong model, your prediction function in Step 12 will silently use untuned parameters.

---

## Step 12 — Build the Prediction Function

### ✅ Expected Output

For Player_1 ranked 12 vs Player_2 ranked 87, Quarterfinals:

```
{'winner': 'Player_1', 'confidence': 0.79, 'p1_win_prob': 0.79}
```

For two closely-ranked players (rank 5 vs rank 6), The Final:

```
{'winner': 'Player_1', 'confidence': 0.52, 'p1_win_prob': 0.52}
```

Confidence near 0.5 for near-equal players is correct behavior — the model is appropriately uncertain. Reversing the ranks in the first example should flip the winner and produce a mirrored confidence score.

### 📖 What to Learn Before Writing This Step

**[Real Python — Defining Functions in Python](https://realpython.com/defining-your-own-python-function/)**
Read the "Function Arguments" and "Return Statement" sections. Functions with default arguments (e.g. `round_name="Quarterfinals"`) let callers omit that argument if the default is fine. Return a dictionary — `return {"winner": ..., "confidence": ...}` — rather than a tuple, because a named dict is self-documenting and easier to use downstream.

**[scikit-learn — predict_proba](https://scikit-learn.org/stable/glossary.html#term-predict_proba)**
Read the glossary entry and note the output shape carefully. `predict_proba` returns a 2D array where each row is one prediction and each column is one class. Column `0` is the probability of Player_2 winning; column `1` is the probability of Player_1 winning. Passing `[:, 0]` instead of `[:, 1]` is a silent bug that inverts all your confidence scores without any error.

**[pandas — DataFrame from dict](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html)**
Read the first example only. The model was trained on a DataFrame with specific column names (`rank_diff`, `pts_diff`, `round_encoded`). When you call `predict_proba` at inference time, you must pass a DataFrame with those *exact same column names in the same order*. Constructing the single-row DataFrame from a dictionary is the cleanest way to guarantee the names are correct.

---

## Supplementary — General ML Foundations

| Resource | Format | What it covers | Time investment |
|---|---|---|---|
| [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) | Text + video | End-to-end ML concepts, loss functions, gradient descent | ~15 hours (do in parts) |
| [Kaggle — Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning) | Interactive notebook | Decision trees, model validation, overfitting | ~3 hours |
| [Kaggle — Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning) | Interactive notebook | Missing values, pipelines, cross-validation, XGBoost | ~4 hours |
| [StatQuest YouTube](https://www.youtube.com/@statquest) | Video | Intuition for every ML algorithm, no math assumed | Pick videos as needed |
| [fast.ai — Practical Deep Learning](https://course.fast.ai/) | Video + notebook | Top-down practical ML, includes tabular models | 20+ hours (long-term) |

> The Kaggle courses are the fastest path from zero to writing this project's code. If you have 7–8 hours before starting, do **Intro + Intermediate ML on Kaggle**. Everything else can wait until you need it.

---

## Quick Reference — Docs by Library

| Library | What you use it for | Docs |
|---|---|---|
| `pandas` | Loading, filtering, cleaning, feature engineering | [pandas.pydata.org](https://pandas.pydata.org/docs/) |
| `numpy` | Array operations, NaN handling | [numpy.org](https://numpy.org/doc/) |
| `matplotlib` | Bar charts, histograms in EDA | [matplotlib.org](https://matplotlib.org/stable/tutorials/) |
| `seaborn` | Distribution plots in EDA | [seaborn.pydata.org](https://seaborn.pydata.org/tutorial.html) |
| `scikit-learn` | Logistic Regression, Random Forest, evaluation, splitting | [scikit-learn.org](https://scikit-learn.org/stable/user_guide.html) |
| `xgboost` | Primary model | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) |
| `joblib` | Saving and loading the trained model | [joblib.readthedocs.io](https://joblib.readthedocs.io/) |

---

## 🖥️ Full Combined Output — All Steps

Run your notebook top-to-bottom. This is every console output you should see in order.

> Step 3 produces charts that render inline in Jupyter — those are noted below but not reproducible as text.

```
# ── Step 1 — Load and Inspect ──────────────────────────────────────────

(67460, 17)

Tournament     object
Date           object
Series         object
Court          object
Surface        object
Round          object
Best of         int64
Player_1       object
Player_2       object
Winner         object
Rank_1          int64
Rank_2          int64
Pts_1           int64
Pts_2           int64
Odd_1         float64
Odd_2          object
Score          object
dtype: object

(no nulls printed — all columns have 0 nulls on load)


# ── Step 2 — Filter to Wimbledon ───────────────────────────────────────

(3061, 17)


# ── Step 3 — EDA ───────────────────────────────────────────────────────
# Two charts render inline in Jupyter (bar chart + histogram)
# Only the groupby table prints to console:

Round
1st Round      1539
2nd Round       769
3rd Round       390
4th Round       192
Quarterfinals    98
Semifinals       48
The Final        25
Name: count, dtype: int64


# ── Step 4 — Clean the Data ────────────────────────────────────────────

(3061, 17)

Rank_1    0
Rank_2    0
dtype: int64

Odd_2 dtype before: object
Odd_2 dtype after:  float64


# ── Step 5 — Feature Engineering ───────────────────────────────────────

         rank_diff       pts_diff  round_encoded  target
count  3061.000000   3061.000000    3061.000000  3061.0
mean      1.056191     16.324404       1.949690     0.5
std     132.932265   2705.543989       1.273697     0.5
min   -1060.000000 -16641.000000       1.000000     0.0
25%     -50.000000   -495.000000       1.000000     0.0
50%      -1.000000      0.000000       1.000000     1.0
75%      50.000000    473.000000       2.000000     1.0
max    1038.000000  16070.000000       7.000000     1.0


# ── Step 6 — Train / Test Split ────────────────────────────────────────

X_train shape: (2197, 3)
X_test shape:  (864, 3)

Features: ['rank_diff', 'pts_diff', 'round_encoded']
Train date range: 2000-06-26 → 2017-07-16
Test date range:  2018-07-02 → 2025-07-13


# ── Step 7 — Naive Baseline ────────────────────────────────────────────

Baseline accuracy: 0.688


# ── Step 8 — Train Three Models ────────────────────────────────────────
# No output with verbose=False (default)


# ── Step 9 — Evaluate All Three Models ─────────────────────────────────

Logistic Regression:  Accuracy=0.689  ROC-AUC=0.764
[[285 147]
 [122 310]]

Random Forest:        Accuracy=0.664  ROC-AUC=0.729
[[281 151]
 [139 293]]

XGBoost:              Accuracy=0.684  ROC-AUC=0.755
[[294 138]
 [135 297]]


# ── Step 10 — Tune XGBoost ─────────────────────────────────────────────

Fitting 5 folds for each of 27 candidates, totalling 135 fits

Best params: {'learning_rate': 0.01, 'max_depth': 3, 'subsample': 0.7}
Best CV ROC-AUC: 0.777


# ── Step 11 — Save the Model ───────────────────────────────────────────

['models/xgb_tuned.joblib']

Round-trip accuracy check: 0.690


# ── Step 12 — Prediction Function (predict.py) ─────────────────────────
# Example 1: clear rank gap

{'winner': 'Player_1', 'confidence': 0.79, 'p1_win_prob': 0.79}

# Example 2: near-equal players

{'winner': 'Player_1', 'confidence': 0.52, 'p1_win_prob': 0.52}
```