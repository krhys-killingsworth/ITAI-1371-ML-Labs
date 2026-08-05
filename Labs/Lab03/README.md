# Lab 03 — The ML Workflow and Types of Learning

**Course:** ITAI 1371 — Applied Machine Learning
**Student:** Krhystopher Killingsworth
**Notebook:** [`L03_KrhystopherKillingsworth_ITAI1371.ipynb`](L03_KrhystopherKillingsworth_ITAI1371.ipynb)
**Dataset:** Wine (`sklearn.datasets`) — 3 cultivars, 13 chemical features

---

## What this lab covered

The distinction between supervised, unsupervised, and reinforcement learning, and the full end-to-end ML workflow: problem definition → data exploration → preprocessing → model training → evaluation → insight. This was the first lab where I built and compared real classification models.

## What I did

- Explored the Wine dataset and ran EDA on the chemical features across the three cultivars.
- Split the data into training and test sets.
- Trained and compared a Logistic Regression model against a Decision Tree classifier.
- Evaluated both with accuracy, a confusion matrix, and a classification report.
- Ran a feature-selection experiment, retraining the same model on different feature subsets.
- Wrote a full reflective journal on the workflow (included in the notebook).

## What I learned

The Decision Tree outperformed Logistic Regression here, and the reason is structural rather than incidental: a tree can carve out non-linear decision boundaries between the three cultivars, while Logistic Regression is constrained to linear separation.

The result I did not expect came from the feature experiment. Using `alcohol`, `color_intensity`, and `proline` gave **83.3% accuracy**. Swapping to `flavanoids`, `proline`, and `od280/od315_of_diluted_wines` — with no other change, same model, same split — gave **94.4%**. An eleven-point swing from feature choice alone.

That reframed the whole course for me. I came in assuming the math would be the hard part. It turned out that *deciding which features to feed the model* is the hard part.

## Challenges

Understanding overfitting properly took some work. It clicked once I stopped thinking of it as "bad accuracy" and started thinking of it as memorization: a model that scores perfectly on training data but collapses on the test set has not learned anything generalizable. That's also what makes the train/test split in step 2 and the evaluation in step 4 two halves of the same idea rather than two separate steps.

If I revisited this lab, I would apply a `StandardScaler` before the Logistic Regression — the Wine features vary wildly in scale, which handicaps a linear model — and use cross-validation instead of a single split for a more reliable estimate.

## Tools and libraries

pandas, NumPy, Matplotlib, Seaborn, scikit-learn (LogisticRegression, DecisionTreeClassifier)

---

[← Back to all labs](../) · [← Back to portfolio home](../../)
