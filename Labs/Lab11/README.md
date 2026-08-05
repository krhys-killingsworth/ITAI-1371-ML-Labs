# Lab 11 — Hyperparameter Tuning and AutoML

**Course:** ITAI 1371 — Applied Machine Learning
**Student:** Krhystopher Killingsworth
**Notebook:** [`L11_KrhystopherKillingsworth_ITAI1371.ipynb`](L11_KrhystopherKillingsworth_ITAI1371.ipynb)
**Dataset:** Iris (`sklearn.datasets`)

---

## What this lab covered

Optimizing a model by systematically searching its hyperparameter space, then handing the same problem to an AutoML framework and comparing the outcome.

## What I did

- Ran `GridSearchCV` over an exhaustive hyperparameter grid.
- Ran `RandomizedSearchCV` over the same space for comparison.
- Trained an AutoGluon `TabularPredictor` on the identical task and read the resulting leaderboard.

## What I learned

**Parameters versus hyperparameters:** parameters are internal values the model learns automatically during training. Hyperparameters are configurations you set *before* training begins, and the model never adjusts them on its own.

**Grid Search versus Random Search** is a budget decision. Grid Search is exhaustive, so it's the right choice when the search space is small and you can afford to cover it completely. Random Search samples instead, which makes it the better option when the space is large or compute time is constrained — it finds a strong configuration without enumerating every combination.

**On the AutoGluon leaderboard, XGBoost came out on top** with 1.00 test accuracy and a 0.952 validation score. Several models tied at 1.00 on the test set but scored lower on validation — a reminder that a test-set tie is not a tie, and validation performance is the better discriminator.

AutoML is powerful because it automates the whole pipeline rather than one step of it. AutoGluon handles its own preprocessing and cross-validation, trains and tunes many model families in parallel, then stacks the best performers into a weighted ensemble.

## Challenges

AutoGluon's leaderboard is genuinely disorienting the first time, because reading it correctly means ignoring the column your eye goes to. Multiple models sitting at a perfect test score forces you to look at validation to break the tie. It also raised a question I still find worth sitting with: if AutoML beats manual tuning this consistently, the skill worth developing is problem framing and metric selection — the parts it can't do for you.

## Tools and libraries

pandas, NumPy, scikit-learn (GridSearchCV, RandomizedSearchCV), AutoGluon

---

[← Back to all labs](../) · [← Back to portfolio home](../../)
