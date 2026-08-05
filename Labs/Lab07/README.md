# Lab 07 — Better Model Evaluation

**Course:** ITAI 1371 — Applied Machine Learning
**Student:** Krhystopher Killingsworth
**Notebook:** [`L07_KrhystopherKillingsworth_ITAI1371.ipynb`](L07_KrhystopherKillingsworth_ITAI1371.ipynb)
**Dataset:** Titanic passenger data

---

## What this lab covered

Moving past raw accuracy into evaluation that actually reflects what a model gets wrong: the confusion matrix, precision, recall, F1, and cross-validation.

## What I did

- Trained a survival classifier on the Titanic dataset as the subject of evaluation.
- Generated and visualized a confusion matrix as a heatmap.
- Produced a full `classification_report` with per-class precision, recall, and F1.
- Ran cross-validation and compared the result against the single train/test split score.

## What I learned

**Precision matters when a false positive is the expensive error.** A spam filter is the clean example: flagging a legitimate email as spam can make someone miss something important at work. A spam message slipping into the inbox is mildly annoying but recoverable.

**Recall matters when a false negative is the expensive error.** Medical screening inverts the calculation entirely. Missing a patient who actually has the disease delays treatment and can be irreversible. A false alarm costs a follow-up test.

**Cross-validation is more trustworthy than a single split** because a single split evaluates the model against exactly one arbitrary subset of the data. CV rotates through multiple folds so every data point is used for testing at some point, and the variance across folds tells you how sensitive the model is to which rows it happened to be handed.

## Challenges

Precision and recall are easy to define and genuinely hard to keep straight under pressure. What fixed it for me was dropping the formulas and anchoring each one to a scenario where the cost of the error is obvious — spam for precision, cancer screening for recall. That framing came back directly in my final project, where choosing a single metric without a guard nearly led me to ship the wrong model.

## Tools and libraries

pandas, Matplotlib, Seaborn, scikit-learn (confusion_matrix, classification_report, cross_val_score)

---

[← Back to all labs](../) · [← Back to portfolio home](../../)
