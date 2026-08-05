# Lab 02 — Tools Used in Machine Learning

**Course:** ITAI 1371 — Applied Machine Learning
**Student:** Krhystopher Killingsworth
**Notebook:** [`L02_KrhystopherKillingsworth_ITAI1371.ipynb`](L02_KrhystopherKillingsworth_ITAI1371.ipynb)
**Dataset:** Iris (`sklearn.datasets`)

---

## What this lab covered

Setting up the working environment for the rest of the course. This lab was about the toolchain rather than the modeling: getting Jupyter, Colab, and VS Code running, confirming the core Python data stack imports correctly, writing professional documentation in Markdown, and initializing a GitHub repository for version control.

## What I did

- Verified the environment by importing and version-checking NumPy, pandas, Matplotlib, and scikit-learn.
- Loaded the Iris dataset and inspected its shape, feature names, and target classes.
- Confirmed the class balance across the three species and visualized it with a bar chart.
- Practiced Markdown formatting for headers, lists, code blocks, and emphasis.
- Initialized the GitHub repository that became this portfolio.

## What I learned

Iris has 150 samples, 4 features, and 3 classes, with exactly 50 samples per class — a perfectly balanced dataset. That balance matters more than it first appears: because no class dominates, accuracy is a meaningful metric here, which is *not* true of the imbalanced datasets I hit later in the course.

The bigger takeaway was about workflow, not data. Jupyter makes it easy to verify and validate data as you go, and GitHub's version history means the work is tracked and documented rather than living in a folder of untitled files. Those two habits carried through every lab after this one.

## Challenges

The setup itself was straightforward. The harder part was resisting the urge to jump straight to modeling — the questions I wrote down at the end (which feature has the most separating power, and how much the features overlap between species) were exactly the questions the EDA lab in Module 4 taught me how to actually answer.

## Tools and libraries

Jupyter Notebook, Google Colab, VS Code, NumPy, pandas, Matplotlib, scikit-learn, Git/GitHub, Markdown

---

[← Back to all labs](../) · [← Back to portfolio home](../../)
