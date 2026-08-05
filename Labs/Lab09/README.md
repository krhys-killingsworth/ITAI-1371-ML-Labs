# Lab 09 — Ensemble Methods

**Course:** ITAI 1371 — Applied Machine Learning
**Student:** Krhystopher Killingsworth
**Notebook:** [`L09_KrhystopherKillingsworth_ITAI1371.ipynb`](L09_KrhystopherKillingsworth_ITAI1371.ipynb)
**Dataset:** Iris (`sklearn.datasets`)

---

## What this lab covered

Combining multiple models into a committee whose collective prediction is more robust than any individual member — bagging (Random Forest) versus boosting, and reading feature importances.

## What I did

- Trained a single Decision Tree classifier on the Iris dataset as the baseline.
- Trained a Random Forest and compared the two on test accuracy.
- Extracted and plotted the Random Forest's feature importances.

## What I learned

The core idea is the wisdom of the crowd. Rather than trusting one algorithm, you combine predictions from multiple diverse models, which reduces the risk that any single model's overfitting drives the final answer.

**The interesting result here was that both models scored 100%, which is a finding about the dataset rather than about the models.** Iris is small, clean, and close to linearly separable, so even a basic model can find decision boundaries that perfectly classify the test set depending on the random split. A perfect score is a signal to be suspicious of the benchmark, not to celebrate the model — this dataset simply cannot distinguish between a good classifier and a great one.

The feature importance plot ranked the petal measurements above the sepal measurements, which matches intuition. When you identify flower species by eye, petal size and shape are the distinguishing features.

## Challenges

Getting a perfect score from both models was initially confusing, because the lab is structured to demonstrate that the ensemble wins. Working out *why* the comparison was uninformative — the dataset is too easy to separate the two — was more valuable than the comparison would have been. It's the reason my final project used a majority-class baseline as an explicit floor, so no score could be mistaken for skill.

## Tools and libraries

pandas, NumPy, Matplotlib, scikit-learn (DecisionTreeClassifier, RandomForestClassifier)

---

[← Back to all labs](../) · [← Back to portfolio home](../../)
