# Lab 12 — Ethics, Fairness, and Bias in ML

**Course:** ITAI 1371 — Applied Machine Learning
**Student:** Krhystopher Killingsworth
**Notebook:** [`L12_KrhystopherKillingsworth_ITAI1371.ipynb`](L12_KrhystopherKillingsworth_ITAI1371.ipynb)
**Dataset:** Adult Census Income

---

## What this lab covered

How models inherit and amplify bias present in their training data, how to measure that bias with group-wise error rates, and how to reason about deployment when the errors are unevenly distributed.

## What I did

- Trained an income classifier on the Adult Census dataset.
- Computed False Positive Rate and False Negative Rate separately for each demographic group.
- Audited the disparity between groups and wrote a deployment recommendation grounded in the numbers.

## What I learned

The audit produced a clear disparate impact. The model over-predicts high income for men — **10.26% FPR versus 2.81%** — but the far more consequential gap is on the other side. The female **FNR is 47.84%**: nearly half of genuinely high-earning women are misclassified as low-income, against **37.8%** for men.

**My recommendation was not to approve this model for hiring screening.** In that context the false negative is the harmful error. A false positive advances someone to an interview, where a human can still filter them out. A false negative eliminates a qualified candidate silently, with no appeal and no visibility. A ten-point gap in that error rate between groups is a direct violation of equal opportunity.

**Dropping the sensitive column does not fix it.** Remove `sex` and the model reconstructs the same boundary from `relationship`, `marital-status`, `occupation`, and `hours-per-week`, all of which correlate with it. The bias survives; only the ability to *audit* it is destroyed. Removing a protected attribute makes a model less fair to inspect, not more fair to deploy.

## Challenges

The hardest part was resisting the framing the accuracy score invites. The model's aggregate accuracy looks respectable, and nothing in that number hints that half of qualified women are being rejected. The disparity only appears when you disaggregate the errors by group — which means fairness auditing has to be a deliberate step, because no default metric will surface it for you.

This is the lab that changed how I think about deploying anything.

## Tools and libraries

pandas, scikit-learn

---

[← Back to all labs](../) · [← Back to portfolio home](../../)
