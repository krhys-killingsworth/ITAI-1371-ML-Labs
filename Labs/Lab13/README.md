# Lab 13 — Building ML Pipelines

**Course:** ITAI 1371 — Applied Machine Learning
**Student:** Krhystopher Killingsworth
**Notebook:** [`L13_KrhystopherKillingsworth_ITAI1371.ipynb`](L13_KrhystopherKillingsworth_ITAI1371.ipynb)
**Dataset:** Titanic passenger data

---

## What this lab covered

Refactoring a sprawling manual preprocessing workflow into a single scikit-learn `Pipeline` object — the step that turns lab code into something another team could actually deploy.

## What I did

- Rebuilt the manual Titanic preprocessing chain (impute → scale → encode → train) as a `ColumnTransformer` plus `Pipeline`.
- Compared the pipeline's accuracy against the manual approach to confirm equivalence.
- Worked through where a dimensionality-reduction step would slot into the pipeline.

## What I learned

Both approaches reached similar accuracy, but the pipeline gets there with less code and three advantages that matter more than the score: **leakage protection, reproducibility, and maintainability.**

**The leakage point is the important one.** `fit_transform()` *learns* statistics — mean, standard deviation — from the data it's given, while `transform()` only applies values already learned. Calling `fit_transform()` on the test set lets test statistics influence the scaling, inflates the accuracy estimate, and leaks information backward. A `Pipeline` makes this structurally impossible: `.fit()` fits every step on training data only, and `.predict()` calls `transform()` on everything downstream.

**On adding PCA:** it goes in the middle, between the preprocessor and the `RandomForestClassifier`. PCA needs numeric input, so scaling and one-hot encoding have to finish first, and the classifier then trains on the reduced components.

**On handing a model to a deployment team:** with five separate objects, they have to reconstruct the exact transformation order and column selections by hand. One wrong step or one forgotten `.transform()` corrupts predictions silently in production. A single `final_pipeline` is one artifact that takes a raw DataFrame and returns predictions, so training and serving logic cannot drift apart.

## Challenges

`fit_transform()` versus `transform()` looks like trivia until you understand that getting it backwards inflates your score rather than throwing an error. That's what makes it dangerous — a leaked test set produces a model that looks *better*, so nothing prompts you to check. Refactoring the manual code made me realize how easily I could have made that mistake in an earlier lab without ever knowing.

## Tools and libraries

pandas, NumPy, scikit-learn (Pipeline, ColumnTransformer, RandomForestClassifier)

---

[← Back to all labs](../) · [← Back to portfolio home](../../)
