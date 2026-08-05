# Lab 08 — The Bias-Variance Tradeoff

**Course:** ITAI 1371 — Applied Machine Learning
**Student:** Krhystopher Killingsworth
**Notebook:** [`L08_KrhystopherKillingsworth_ITAI1371.ipynb`](L08_KrhystopherKillingsworth_ITAI1371.ipynb)
**Dataset:** Synthetic polynomial data

---

## What this lab covered

Underfitting, overfitting, and the tradeoff between them — visualized directly by fitting models of deliberately different complexity to the same data and plotting learning curves for each.

## What I did

- Fit polynomial regression models of degree 1, degree 4, and degree 15 to the same dataset.
- Plotted all three fits against the underlying data to see complexity behave visually.
- Generated learning curves for the underfitting and overfitting cases, tracking training score against cross-validation score as the training set grows.

## What I learned

**Degree 1 underfits.** The fit is a straight line and simply cannot bend to the shape of the data — high bias.

**Degree 15 overfits.** The curve is erratic, swinging aggressively to chase individual noisy points instead of following any smooth trend — high variance.

**Degree 4 is the good fit.** It flows smoothly through the data without violent bends, capturing the real pattern while ignoring the noise.

The learning curves make the diagnosis mechanical rather than visual, which is the actual point of the lab:

- **Underfitting:** training and cross-validation scores are both low and converge quickly to the same poor value. Adding data will not help — the model is too simple to represent the pattern at all.
- **Overfitting:** a large, persistent gap between a high training score and a much lower cross-validation score. The model has memorized the training set rather than learning from it.

## Challenges

I could see overfitting in the degree-15 plot immediately, but reading it off a learning curve was less intuitive. The thing that made it usable was realizing the *gap* is the signal, not the absolute scores: converged-but-low means bias, wide-and-persistent means variance. That gives you a diagnostic you can apply to a model you can't plot.

## Tools and libraries

NumPy, Matplotlib, scikit-learn (PolynomialFeatures, LinearRegression, learning_curve)

---

[← Back to all labs](../) · [← Back to portfolio home](../../)
