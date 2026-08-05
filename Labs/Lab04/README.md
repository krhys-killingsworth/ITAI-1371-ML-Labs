# Lab 04 — Exploratory Data Analysis

**Course:** ITAI 1371 — Applied Machine Learning
**Student:** Krhystopher Killingsworth
**Notebook:** [`L04_KrhystopherKillingsworth_ITAI1371.ipynb`](L04_KrhystopherKillingsworth_ITAI1371.ipynb)
**Dataset:** Titanic passenger data

---

## What this lab covered

Using summary statistics and visualization to understand a dataset *before* modeling it — looking for patterns, relationships, outliers, and data quality problems so that later modeling decisions rest on evidence instead of assumption.

## What I did

- Generated descriptive statistics with `.describe()` and audited the dataset for missing values.
- Built visualizations of survival against passenger class, sex, and age.
- Used a correlation heatmap to examine relationships between numeric variables.
- **Experiment 1:** a countplot of survival by port of embarkation (Cherbourg, Queenstown, Southampton).
- **Experiment 2:** a boxplot comparing the `Fare` distribution for survivors against non-survivors.

## What I learned

The survival profile that emerges from the plots is stark: first-class, female, and young passengers survived at dramatically higher rates, with fare acting as a proxy for the same underlying advantage.

The methodological lesson mattered more than the historical one. Summary statistics compress an entire distribution into a single number, and that compression hides things. A mean age of 29 tells you nothing about whether the survivors included young children or the elderly. A boxplot shows the spread and the outliers that the mean silently averages away.

The way I'd put it: **a statistic tells you a number; a plot shows you why.**

## Challenges

The temptation with a fully-coded lab is to run every cell and move on. Forcing myself to write down what each visualization was actually *for* — before looking at the output — was the part that made it stick. Reading the `Fare` boxplot also took a second pass, because the heavy right skew from a handful of very expensive tickets initially made the survivor and non-survivor distributions look more similar than they are.

## Tools and libraries

pandas, Matplotlib, Seaborn

---

[← Back to all labs](../) · [← Back to portfolio home](../../)
