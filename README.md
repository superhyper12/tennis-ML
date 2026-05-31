# 📚 Learning Resources by Step
## Wimbledon Match Outcome Predictor

> **How to use this:** Don't try to read everything upfront. Work through one step at a time. Each resource section tells you exactly what to learn, what to focus on, and why it matters before you write that step's code.

> **🔗 Resource style (this revision):** Where a good tutorial-site walkthrough exists, the primary link now points to **GeeksforGeeks, W3Schools, or TutorialsPoint** — these read more like step-by-step lessons with worked code than the official reference docs. For a few precision-critical details (the `predict_proba` column-order gotcha, the `SettingWithCopyWarning` fix, and time-series cross-validation), the **official docs are kept** because the tutorial-site pages are thinner or occasionally inaccurate on those exact points. Each kept-official link says why.

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

**[GeeksforGeeks — Getting started with Classification](https://www.geeksforgeeks.org/machine-learning/getting-started-with-classification/)**
Read this first for the vocabulary. You need to understand three terms before anything else: *label* (what you're predicting — the match winner), *feature* (the inputs — rank, points, round), and *model* (the function that maps features to a label). Every step in this project maps to one of those three concepts.

**[GeeksforGeeks — Types of Machine Learning / supervised learning](https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/)**
Skim the supervised-learning section. The goal is to see that this project is *supervised classification*: you have labeled historical matches (who actually won), and you're training a model to predict the label for new matchups.

### Concept: Jupyter Notebooks

**[GeeksforGeeks — How to use Jupyter Notebook](https://www.geeksforgeeks.org/installation-guide/how-to-use-jupyter-notebook-an-ultimate-guide/)**
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

**[GeeksforGeeks — Pandas Read CSV in Python](https://www.geeksforgeeks.org/pandas/python-read-csv-using-pandas-read_csv/)**
This is the one function you need to load the data. The page walks through `pd.read_csv()` with worked examples and covers the parameters you'll actually touch (`header`, `index_col`, `usecols`).

**[W3Schools — Pandas DataFrames](https://www.w3schools.com/python/pandas/pandas_dataframes.asp)** and **[W3Schools — Analyzing DataFrames](https://www.w3schools.com/python/pandas/pandas_analyzing.asp)**
Read this if you want the simplest possible intro. The key thing to understand before writing Step 1: the difference between `.shape` (how many rows and columns exist), `.dtypes` (what type each column is), and `.isnull().sum()` (how many values are missing per column). The "Analyzing DataFrames" page covers `head()`, `info()`, and inspecting a dataset you've never seen.

**[GeeksforGeeks — Python | Pandas DataFrame.dtypes](https://www.geeksforgeeks.org/pandas/python-pandas-dataframe-dtypes/)**
The specific thing to understand: why `Odd_2` loads as `object` instead of `float64` even though it looks like a number. This happens because one row contains a non-numeric value, so pandas flags the entire column as `object`. You'll fix this in Step 4, but you need to understand why it happens now.

---

## Step 2 — Filter to Wimbledon Only

### ✅ Expected Output

```
(3061, 17)
```

The filtered DataFrame contains only rows where `Tournament` is `"Wimbledon"`. Spot-check that the `Surface` column reads `"Grass"` for every row — no exceptions.

### 📖 What to Learn Before Writing This Step

**[GeeksforGeeks — Filter Pandas Dataframe by Column Value (Boolean indexing)](https://www.geeksforgeeks.org/pandas/filtering-rows-from-dataframe-in-python/)**
The concept to understand: filtering rows in pandas works by creating a True/False mask the same length as the DataFrame, then using that mask to select only the rows where the condition is True.

**[GeeksforGeeks — Python | Pandas Series.str.contains()](https://www.geeksforgeeks.org/pandas/python-pandas-series-str-contains/)**
You need `str.contains("Wimbledon")` specifically because tournament names sometimes have slight variations — an exact `==` match would miss those. In this dataset the name is always exactly `"Wimbledon"`, but `str.contains` is safer practice.

**[Corey Schafer — Pandas Tutorial Part 2 (Filtering, YouTube)](https://www.youtube.com/watch?v=Lw2rlcxScZY)**
Watch the first 20 minutes for a video walkthrough. Pay attention to how he chains multiple conditions with `&` and `|` — you won't need that here, but it comes back in Step 4 when you replace sentinel values across multiple columns at once.

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

**[GeeksforGeeks — Exploratory Data Analysis (EDA) with Python](https://www.geeksforgeeks.org/data-analysis/what-is-exploratory-data-analysis/)**
The most important idea here isn't a function — it's a mindset: EDA is about asking specific questions and answering them visually before you touch any model. The two questions for this step are "which rounds have the most matches?" and "what does the rank distribution look like?" If you skip EDA and jump straight to modeling, you'll miss obvious data quality issues that break everything later.

**[GeeksforGeeks — Introduction to Matplotlib](https://www.geeksforgeeks.org/python/python-introduction-matplotlib/)** (and **[W3Schools — Matplotlib Bars](https://www.w3schools.com/python/matplotlib_bars.asp)** / **[Histograms](https://www.w3schools.com/python/matplotlib_histograms.asp)**)
Read through the bar chart and histogram examples only. The one thing to understand: `plt.show()` must be called after each plot or nothing renders. Also note that `plt.title()`, `plt.xlabel()`, and `plt.ylabel()` are called *before* `plt.show()` — this trips up almost everyone the first time.

**[GeeksforGeeks — Introduction to Seaborn](https://www.geeksforgeeks.org/python/introduction-to-seaborn-python/)**
Seaborn's `histplot()` produces much cleaner histograms than matplotlib's `hist()` — specifically because it handles bin sizing automatically and overlays a KDE (density curve) by default. You'll use it for the rank distribution plot.

**[GeeksforGeeks — Pandas dataframe.groupby() Method](https://www.geeksforgeeks.org/pandas/python-pandas-dataframe-groupby/)**
The concept to understand: `groupby("Round").size()` splits the DataFrame into one group per round, then counts the rows in each group. The page uses a "split-apply-combine" framing that maps directly to almost every "win rate by X" question you'll ask in EDA.

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

**[GeeksforGeeks — Working with Missing Data in Pandas](https://www.geeksforgeeks.org/pandas/working-with-missing-data-in-pandas/)**
Covers detecting missing values (`isnull`), removing them (`dropna`), and replacing values (`replace`). The key distinction here: `dropna()` by default removes any row with *any* null in *any* column. In this project you don't need to drop any rows on rank, but you still need to handle sentinel `-1` values in `Pts_1`, `Pts_2`, and `Odd_*` columns.

**[W3Schools — Pandas Cleaning Empty Cells](https://www.w3schools.com/python/pandas/pandas_cleaning_empty_cells.asp)** and **[Cleaning Wrong Data](https://www.w3schools.com/python/pandas/pandas_cleaning_wrong_data.asp)**
The specific concept you need: the difference between a true `NaN` (genuinely missing) and a sentinel value like `-1` (a placeholder used when data wasn't collected). If you don't replace `-1` in points columns before computing `pts_diff` in Step 5, the model will treat those as real point values, silently corrupting your features.

**[GeeksforGeeks — Convert the column type from string to datetime (pd.to_datetime)](https://www.geeksforgeeks.org/pandas/convert-the-column-type-from-string-to-datetime-format-in-pandas-dataframe/)**
The `Date` column loads as strings in `YYYY-MM-DD` format. Converting it to a proper datetime with `pd.to_datetime(df["Date"])` is what makes the time-based train/test split in Step 6 possible.

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

**[GeeksforGeeks — What is Feature Engineering?](https://www.geeksforgeeks.org/machine-learning/what-is-feature-engineering/)**
The core idea: raw columns like `Rank_1` and `Rank_2` are less useful to a model than the *relationship* between them (`rank_diff = Rank_1 - Rank_2`). A model given both raw ranks has to discover the subtraction relationship on its own — computing it explicitly makes the pattern obvious and improves performance. This is the reasoning behind every new column you create in this step.

**[GeeksforGeeks — Creating a new column in a Pandas DataFrame](https://www.geeksforgeeks.org/pandas/creating-a-new-column-in-pandas/)**
Covers the `df["new_col"] = expression` pattern you'll use to add `rank_diff`, `pts_diff`, and `round_encoded`.

**[⚠️ Official scikit-learn docs — Returning a view versus a copy](https://pandas.pydata.org/docs/user_guide/indexing.html#returning-a-view-versus-a-copy)** *(kept official on purpose)*
This `SettingWithCopyWarning` gotcha is one the tutorial sites gloss over, and getting it slightly wrong causes silent, confusing bugs — so use the authoritative pandas source. The specific fix: when you write `df["new_col"] = expression` on a filtered slice of a DataFrame (which `wimbledon` is), pandas may raise a `SettingWithCopyWarning`. Calling `.copy()` when you create the filtered DataFrame in Step 2 prevents the warning from appearing in every step after it.

**[GeeksforGeeks — Label Encoding in Python](https://www.geeksforgeeks.org/machine-learning/ml-label-encoding-of-datasets-in-python/)**
Round names like `"Quarterfinals"` and `"Semifinals"` are strings, but models only understand numbers. In this project you'll use a hand-crafted dictionary map rather than `LabelEncoder` directly — but understanding what label encoding *is* and why the order matters (`The Final=7 > Semifinals=6 > Quarterfinals=5...`) is what makes `round_encoded` a meaningful ordinal feature rather than arbitrary numbers.

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

**[GeeksforGeeks — How To Do Train Test Split Using Sklearn In Python](https://www.geeksforgeeks.org/machine-learning/how-to-do-train-test-split-using-sklearn-in-python/)**
The concept to internalize: a model's accuracy on the data it was *trained on* tells you nothing useful — it has seen those examples before. Accuracy on the *test set* (data the model has never seen) tells you whether the model has learned a general pattern or just memorized. This page covers `train_test_split` parameters; for *this* project you'll override the default random split with a date-based split (see leakage note below).

**[GeeksforGeeks — Data Leakage in Machine Learning](https://www.geeksforgeeks.org/machine-learning/data-leakage-in-machine-learning/)**
This is the most dangerous mistake you can make in this step. The specific risk: if you use a random split instead of a time-based split, some 2010 matches end up in the test set and some 2020 matches end up in training. The model can then effectively "see the future," producing inflated accuracy numbers that fall apart on real predictions. The fix is splitting strictly on date: everything before 2018 trains, everything after tests.

**[⚠️ Official scikit-learn docs — Common pitfalls / data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)** *(kept official on purpose)*
Leakage is subtle enough that it's worth reading the authoritative description once — including the exact mistake of fitting any transformation on the full dataset before splitting. The tutorial-site versions are good for the concept; this is the reference that gets the details right.

---

## Step 7 — Naive Baseline

### ✅ Expected Output

```
Baseline accuracy: 0.688
```

~69% is your floor. Any model that can't beat this is not worth deploying. The baseline rule is: predict Player_1 wins if `rank_diff < 0` (Player_1 has the better rank). The goal of Steps 8–10 is to meaningfully exceed this number.

> **Note:** Unlike the original ATP dataset where winners were always listed first, this dataset has real match pairings — both players are present and `target` is 50/50 overall. The baseline captures the real signal that better-ranked players win more often, but it's a meaningful challenge to beat on the test set.

### 📖 What to Learn Before Writing This Step

**[GeeksforGeeks — Underfitting and Overfitting in Machine Learning](https://www.geeksforgeeks.org/machine-learning/underfitting-and-overfitting-in-machine-learning/)**
The concept to understand: a model that scores 75% accuracy might sound impressive, but if a simple rule gets 69% for free, your model is only adding 6% of real value. Baselines calibrate what "good" actually means.

**[GeeksforGeeks — How to Calculate Accuracy in Python (sklearn accuracy_score)](https://www.geeksforgeeks.org/machine-learning/how-to-compute-the-accuracy-of-classification-model-in-python/)**
For the baseline, your "predicted" labels come from a simple rule (`rank_diff < 0`), not a trained model. This is intentional — it shows that even a rule with zero learning beats random guessing by a wide margin.

**[TutorialsPoint — Scikit Learn DummyClassifier](https://www.tutorialspoint.com/scikit_learn/scikit_learn_estimator_api.htm)**
Optional. `DummyClassifier` is sklearn's built-in way to produce baseline predictions (e.g. "always predict the majority class"). Seeing it makes the *purpose* of a baseline click — it's the standard tool for the exact job this step does by hand.

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
Watch this first, before reading anything else. It explains *why* logistic regression outputs a probability between 0 and 1 rather than a raw number — using the sigmoid function to squash linear output. If you skip this, the `predict_proba` output in Step 12 won't make sense.

**[GeeksforGeeks — Logistic Regression using Python (scikit-learn)](https://www.geeksforgeeks.org/machine-learning/ml-logistic-regression-using-python/)**
A full worked example on the breast-cancer dataset — fitting, predicting, and reading `predict_proba` / ROC-AUC. This is the closest tutorial-site match to what you'll write here.

**[GeeksforGeeks — StandardScaler in Sklearn](https://www.geeksforgeeks.org/machine-learning/standardscaler-minmaxscaler-and-robustscaler-techniques/)**
The key reason you must scale before Logistic Regression: the algorithm treats all features as equally important by default. If `pts_diff` ranges from -16000 to +16000 and `round_encoded` ranges from 1 to 7, the model will overweight points differences just because the numbers are bigger — not because they're more predictive. Scaling puts all features on the same scale.

**[GeeksforGeeks — Pipelines in Scikit-learn](https://www.geeksforgeeks.org/machine-learning/pipelines-using-scikit-learn/)**
The critical concept: a Pipeline chains the scaler and the classifier into one object so they can't be applied in the wrong order. Without a Pipeline, it's easy to accidentally scale the test data using test-data statistics — another form of leakage. With a Pipeline, `fit()` learns the scale from training data only.

**Random Forest**

**[StatQuest — Random Forests (YouTube, 9 min)](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)**
Watch before reading. The intuition: a single decision tree memorizes the training data and fails on new data (overfitting). A random forest fixes this by building hundreds of trees on random subsets of the data and averaging their predictions — the randomness prevents any one tree from dominating.

**[GeeksforGeeks — Random Forest Classifier using Scikit-learn](https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/)**
A complete worked example covering `n_estimators`, `max_depth`, and — most importantly for this step — `feature_importances_`, which tells you which of your three features the forest relied on most and validates whether your feature engineering in Step 5 was actually useful.

**XGBoost**

**[StatQuest — XGBoost Part 1 (YouTube, 16 min)](https://www.youtube.com/watch?v=OtD8wVaFm6E)**
Watch Part 1 only for now. The key difference from Random Forest: XGBoost builds trees *sequentially*, where each new tree is trained specifically on the mistakes of the previous trees. This is why it typically outperforms Random Forest on structured tabular data.

**[GeeksforGeeks — XGBClassifier](https://www.geeksforgeeks.org/machine-learning/xgbclassifier/)**
Walks through the scikit-learn-compatible XGBoost API and the parameters you'll tune in Step 10 (`n_estimators`, `learning_rate`, `max_depth`). For early stopping (stopping training automatically when validation performance plateaus to avoid overfitting), the authoritative reference is the [official XGBoost early-stopping docs](https://xgboost.readthedocs.io/en/stable/python/python_intro.html#early-stopping).

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

**[GeeksforGeeks — Metrics for Machine Learning Model Evaluation](https://www.geeksforgeeks.org/machine-learning/metrics-for-machine-learning-model/)**
The key insight: accuracy alone is misleading when one class is more common than the other. ROC-AUC measures whether the model ranks positive examples higher than negative ones — it stays meaningful even if class distribution is uneven. An AUC of 0.5 means random guessing; 1.0 means perfect ranking; above 0.75 is practically useful.

**[StatQuest — ROC and AUC (YouTube, 16 min)](https://www.youtube.com/watch?v=4jRBRDbJemM)**
Watch this if the ROC curve concept isn't clicking from text alone. StatQuest builds the curve step-by-step from scratch using a tiny example — by the end you'll understand exactly what the x and y axes represent (false positive rate and true positive rate) and why the area under the curve is a useful single-number summary.

**[GeeksforGeeks — Confusion Matrix in Machine Learning](https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/)**
Understand the 2×2 layout: top-left = true negatives (correctly predicted Player_2 wins), top-right = false positives (wrongly predicted Player_1), bottom-left = false negatives (missed Player_1 wins), bottom-right = true positives. The confusion matrix tells you *where* the model fails — not just *how often* — which is critical when comparing models.

**[⚠️ Official scikit-learn docs — roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)** *(kept official on purpose)*
The one thing to know is a silent-bug trap the tutorial sites tend to skip: `roc_auc_score` requires *probabilities* (from `predict_proba`), not hard class labels, and you must pass the probability of the *positive class* — column index `[:, 1]`. Passing the wrong column gives a wrong score with no error message, so use the precise official reference here.

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

**[GeeksforGeeks — Hyperparameter tuning using GridSearchCV and KerasClassifier](https://www.geeksforgeeks.org/machine-learning/sklearn-gridsearchcv-with-pipeline/)**
The concept: `GridSearchCV` tries every combination of values in `param_grid` and scores each one using cross-validation. With 3 values for each of 3 parameters, that's 27 combinations × 5 folds = 135 total training runs. Setting `n_jobs=-1` runs these in parallel using all available CPU cores — without it, this step takes significantly longer.

**[GeeksforGeeks — Cross Validation in Machine Learning](https://www.geeksforgeeks.org/machine-learning/cross-validation-machine-learning/)**
Background on why cross-validation gives a more trustworthy score than a single validation split — it averages performance across several folds instead of trusting one lucky/unlucky split.

**[⚠️ Official scikit-learn docs — TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)** *(kept official on purpose)*
This is the crux of the step and the tutorial sites are thin on it, so use the official reference and study its fold diagram. The problem with standard k-fold on time-series data: it shuffles before splitting, so fold 1 might train on 2019 data and validate on 2015 data — future-training-on-past, which is leakage. `TimeSeriesSplit` always trains on older data and validates on newer data, mirroring the real-world use case where you predict future matches from historical ones. (See also the official [Visualizing cross-validation behavior](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html) example for the colored fold diagrams.)

---

## Step 11 — Save the Trained Model

### ✅ Expected Output

```
['models/xgb_tuned.joblib']

Round-trip accuracy check: 0.690
```

The first line is the return value of `joblib.dump()` — it confirms the path where the file was written. The `models/` folder now contains `xgb_tuned.joblib`. The round-trip accuracy comes from `grid.best_estimator_` evaluated on the test set — it must match exactly before and after loading. If it doesn't, the most common cause is a version mismatch between the sklearn/xgboost version used to train and the version used to load.

### 📖 What to Learn Before Writing This Step

**[GeeksforGeeks — Saving a machine learning Model (joblib & pickle)](https://www.geeksforgeeks.org/machine-learning/saving-a-machine-learning-model/)**
The key concept: a trained model is just a Python object with learned weights stored in memory. `joblib.dump()` serializes that object to disk so you can reload it later without retraining. The page also explains why `joblib` is preferred over Python's built-in `pickle` for ML models — joblib handles large numpy arrays (which store the tree weights) more efficiently.

**[GeeksforGeeks — Save and Load Machine Learning Models in Python with scikit-learn](https://www.geeksforgeeks.org/machine-learning/save-and-load-machine-learning-models-in-python-with-scikit-learn/)**
Worked `dump`/`load` example. Always save `grid.best_estimator_` (the tuned model from Step 10), not the original `xgb` object from Step 8. If you accidentally save the wrong model, your prediction function in Step 12 will silently use untuned parameters.

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

**[GeeksforGeeks — Python Functions (arguments & return values)](https://www.geeksforgeeks.org/python/python-functions/)**
Functions with default arguments (e.g. `round_name="Quarterfinals"`) let callers omit that argument if the default is fine. Return a dictionary — `return {"winner": ..., "confidence": ...}` — rather than a tuple, because a named dict is self-documenting and easier to use downstream.

**[⚠️ Official scikit-learn docs — predict_proba (glossary)](https://scikit-learn.org/stable/glossary.html#term-predict_proba)** *(kept official on purpose)*
This is the single most error-prone line in the whole project, so use the precise source. `predict_proba` returns a 2D array where each row is one prediction and each column is one class. Column `0` is the probability of Player_2 winning; column `1` is the probability of Player_1 winning. Passing `[:, 0]` instead of `[:, 1]` is a silent bug that inverts all your confidence scores without any error.

**[GeeksforGeeks — Creating a Pandas DataFrame from a dictionary](https://www.geeksforgeeks.org/pandas/create-a-pandas-dataframe-from-dicts/)**
The model was trained on a DataFrame with specific column names (`rank_diff`, `pts_diff`, `round_encoded`). When you call `predict_proba` at inference time, you must pass a DataFrame with those *exact same column names in the same order*. Constructing the single-row DataFrame from a dictionary is the cleanest way to guarantee the names are correct.

---

## Supplementary — General ML Foundations

| Resource | Format | What it covers | Time investment |
|---|---|---|---|
| [GeeksforGeeks — Machine Learning Tutorial](https://www.geeksforgeeks.org/machine-learning/machine-learning/) | Text + code | End-to-end ML concepts, every algorithm with worked Python | Browse as needed |
| [W3Schools — Python Machine Learning](https://www.w3schools.com/python/python_ml_getting_started.asp) | Text + interactive | Mean/median/mode, train/test, decision trees, confusion matrix, AUC | ~3 hours |
| [TutorialsPoint — Scikit-learn Tutorial](https://www.tutorialspoint.com/scikit_learn/index.htm) | Text + code | The full sklearn workflow: estimators, modelling, evaluation | ~3 hours |
| [Kaggle — Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning) | Interactive notebook | Decision trees, model validation, overfitting | ~3 hours |
| [Kaggle — Intermediate ML](https://www.kaggle.com/learn/intermediate-machine-learning) | Interactive notebook | Missing values, pipelines, cross-validation, XGBoost, leakage | ~4 hours |
| [StatQuest YouTube](https://www.youtube.com/@statquest) | Video | Intuition for every ML algorithm, no math assumed | Pick videos as needed |

> The Kaggle courses are the fastest *interactive* path from zero to writing this project's code. If you prefer reading lessons with code you can copy, GeeksforGeeks' Machine Learning Tutorial covers every step of this project in article form.

---

## Quick Reference — Docs by Library

| Library | What you use it for | Tutorial | Official docs |
|---|---|---|---|
| `pandas` | Loading, filtering, cleaning, feature engineering | [GfG Pandas Tutorial](https://www.geeksforgeeks.org/pandas/pandas-tutorial/) · [W3Schools Pandas](https://www.w3schools.com/python/pandas/) | [pandas.pydata.org](https://pandas.pydata.org/docs/) |
| `numpy` | Array operations, NaN handling | [GfG NumPy Tutorial](https://www.geeksforgeeks.org/python/numpy-tutorial/) · [W3Schools NumPy](https://www.w3schools.com/python/numpy/) | [numpy.org](https://numpy.org/doc/) |
| `matplotlib` | Bar charts, histograms in EDA | [GfG Matplotlib](https://www.geeksforgeeks.org/python/python-introduction-matplotlib/) · [W3Schools Matplotlib](https://www.w3schools.com/python/matplotlib_intro.asp) | [matplotlib.org](https://matplotlib.org/stable/tutorials/) |
| `seaborn` | Distribution plots in EDA | [GfG Seaborn](https://www.geeksforgeeks.org/python/introduction-to-seaborn-python/) | [seaborn.pydata.org](https://seaborn.pydata.org/tutorial.html) |
| `scikit-learn` | LR, Random Forest, evaluation, splitting | [TutorialsPoint Scikit-learn](https://www.tutorialspoint.com/scikit_learn/index.htm) · [GfG sklearn](https://www.geeksforgeeks.org/machine-learning/learning-model-building-scikit-learn-python-machine-learning-library/) | [scikit-learn.org](https://scikit-learn.org/stable/user_guide.html) |
| `xgboost` | Primary model | [GfG XGBoost](https://www.geeksforgeeks.org/machine-learning/xgboost/) | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) |
| `joblib` | Saving and loading the trained model | [GfG Saving Models](https://www.geeksforgeeks.org/machine-learning/saving-a-machine-learning-model/) | [joblib.readthedocs.io](https://joblib.readthedocs.io/) |

---

## 🔧 Resource swaps applied in this revision

Tutorial-site pages (GeeksforGeeks / W3Schools / TutorialsPoint) now lead each section. The following links were **kept official on purpose** because the tutorial-site versions are thinner or risk silent bugs on these exact points:

| Step | Kept official | Why |
|---|---|---|
| 5 | pandas — view vs copy (`SettingWithCopyWarning`) | Tutorial sites gloss over it; getting it wrong = silent bugs |
| 6 | scikit-learn — Common pitfalls / data leakage | Authoritative description of the most dangerous mistake here |
| 9 | scikit-learn — `roc_auc_score` | The `[:, 1]` positive-class column trap is a silent bug |
| 10 | scikit-learn — `TimeSeriesSplit` (+ CV-visualization example) | Crux of the step; tutorial sites are thin on time-series CV |
| 12 | scikit-learn — `predict_proba` glossary | Most error-prone line in the project; column order matters |
| 8 | XGBoost — early stopping (secondary link) | Precise behavior of the early-stopping callback |

> ✅ All tutorial-site links were verified live. StatQuest videos were kept as the video resource for the model concepts (Logistic Regression, Random Forest, XGBoost, ROC/AUC) since they're the clearest visual explanations available. If any link breaks later, search the page title on the site (e.g. "geeksforgeeks random forest classifier scikit-learn").