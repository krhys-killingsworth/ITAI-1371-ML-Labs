# Lab 06 — Regression and Classification Models

**Course:** ITAI 1371 — Applied Machine Learning
**Student:** Krhystopher Killingsworth
**Notebook:** [`L06_KrhystopherKillingsworth_ITAI1371.ipynb`](L06_KrhystopherKillingsworth_ITAI1371.ipynb)
**Dataset:** Titanic passenger data

---

## What this lab covered

The fundamental split in supervised learning — predicting a continuous number versus predicting a discrete category — and building the first linear model for each.

## What I did

- **Regression:** trained a `LinearRegression` model to predict `Fare` from `Age` and `Pclass`, evaluated with Mean Squared Error.
- **Classification:** encoded `Sex` numerically, then trained a `LogisticRegression` model to predict `Survived` from `Age`, `Pclass`, and `Sex`, evaluated with accuracy.
- Inspected the regression model's `.coef_` attribute to read the learned feature weights.

## What I learned

Regression predicts a continuous numeric output; classification assigns a discrete category or label. That difference propagates all the way through — it dictates the model, the loss function, and the evaluation metric.

`lr_model.coef_` returns one coefficient per feature (here, one for `Age` and one for `Pclass`). Each is a learned weight describing how much that feature moves the prediction.

The metrics can't be swapped. MSE measures how far predictions land from the true values and penalizes large errors disproportionately, which is what you want when optimizing a continuous prediction. Accuracy is meaningless for fare prediction, because predicting a fare of \$32.14 when the true value is \$32.15 is not "wrong" in any useful sense — but under accuracy it counts as a total miss.

## Challenges

Keeping the two pipelines mentally separate was the tricky part, since both run on the same Titanic dataframe with overlapping features. Encoding `Sex` in place partway through the notebook also means the cells have to be executed in order — a small preview of why Lab 13's `Pipeline` objects exist.

## Tools and libraries

pandas, NumPy, scikit-learn (LinearRegression, LogisticRegression, mean_squared_error, accuracy_score)

---

[← Back to all labs](../) · [← Back to portfolio home](../../)
