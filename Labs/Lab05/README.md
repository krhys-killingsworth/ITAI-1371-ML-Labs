# Lab 05 — Data Preparation

**Course:** ITAI 1371 — Applied Machine Learning
**Student:** Krhystopher Killingsworth
**Notebook:** [`L05_KrhystopherKillingsworth_ITAI1371.ipynb`](L05_KrhystopherKillingsworth_ITAI1371.ipynb)
**Dataset:** Titanic passenger data

---

## What this lab covered

Preprocessing raw data into something a model can actually consume: imputing missing values, encoding categorical text into numbers, and scaling numeric features onto comparable ranges.

## What I did

- Identified missing values across the dataset, concentrated in `Age` and `Cabin`.
- Imputed missing `Age` values using the median rather than the mean.
- Applied One-Hot Encoding to categorical columns so text labels became usable features.
- Applied feature scaling to the numeric columns and compared the before/after distributions.

## What I learned

**Median beats mean for imputation when the distribution is skewed.** Titanic `Age` and especially `Fare` both have long right tails — a handful of first-class passengers paid many times the typical fare — which drags the mean upward and away from anything representative. The median sits at the middle of the actual values and is far more robust to those outliers.

**One-Hot Encoding is an interpreter.** It converts a categorical column of text labels into separate binary columns, one per category, keeping them independent of each other. Models speak math, not text, and this is the translation layer.

**Scaling is model-dependent, not universal.** A Decision Tree does not need feature scaling, because it splits on the values of one feature at a time. Multiply a feature by 100 and the tree just picks a threshold that is 100× larger — the split order is unchanged. Distance-based and gradient-based models care enormously; trees don't.

## Challenges

The scaling question was the one I got wrong at first. My instinct was that scaling is always good practice, so it took working through *how* a tree actually makes a split to see why it's simply irrelevant there. Understanding the mechanism, rather than memorizing a preprocessing checklist, is what let me answer it.

## Tools and libraries

pandas, NumPy, scikit-learn (SimpleImputer, OneHotEncoder, StandardScaler)

---

[← Back to all labs](../) · [← Back to portfolio home](../../)
