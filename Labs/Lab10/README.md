# Lab 10 — Unsupervised Learning

**Course:** ITAI 1371 — Applied Machine Learning
**Student:** Krhystopher Killingsworth
**Notebook:** [`L10_KrhystopherKillingsworth_ITAI1371.ipynb`](L10_KrhystopherKillingsworth_ITAI1371.ipynb)
**Dataset:** Iris (`sklearn.datasets`)

---

## What this lab covered

Learning without labels: K-Means clustering to discover groups in data, and Principal Component Analysis to reduce dimensionality while preserving as much variance as possible.

## What I did

- Applied K-Means clustering to the Iris features with the labels withheld.
- Built an elbow plot of inertia across values of `k` to select the cluster count.
- Reduced the 4-dimensional feature space to 2 principal components with PCA.
- Plotted the 2-component projection and checked the explained variance ratio.

## What I learned

Supervised learning trains on labeled data to predict a known outcome; unsupervised learning searches unlabeled data for structure that nobody specified in advance.

**On the elbow plot:** you can't just maximize `k`. Push it far enough and every data point becomes its own cluster, driving inertia to zero while producing groups that mean nothing. The elbow is where additional clusters stop buying meaningful separation — it's a judgment call about usefulness, not an optimization target.

**On PCA:** the first two components explained over 95% of the variance in the 4-feature dataset. That says the original features are heavily correlated with one another — the dataset's real structure is close to two-dimensional, and half the columns are largely redundant. Compressing it loses almost nothing.

## Challenges

The conceptual shift was the hard part. Every previous lab had a right answer sitting in a `y` column to check against, and clustering removes that. Evaluating a clustering result means asking whether the groups are *useful*, not whether they're correct, and that's a much less comfortable question.

## Tools and libraries

pandas, Matplotlib, Seaborn, scikit-learn (KMeans, PCA)

---

[← Back to all labs](../) · [← Back to portfolio home](../../)
