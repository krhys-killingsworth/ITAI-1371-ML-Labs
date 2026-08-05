# Krhystopher Killingsworth — Machine Learning Portfolio

**ITAI 1371 — Applied Machine Learning**
Houston City College · A.A.S. Artificial Intelligence & Robotics

---

## About Me

I'm a first-year AI and Robotics student at Houston City College, working toward an A.A.S. in Artificial Intelligence. I came into this program from a background in customer-facing, administrative, and skilled trade work, with no prior programming experience.

My interest is in AI and robotics: systems that act in the world rather than just report a number. That makes it matter a great deal whether the number is real. The labs I found most interesting here were the ones where the model produced a number that looked good and was, on inspection, meaningless: a Random Forest scoring 100% on a dataset too easy to be informative, a classifier with respectable aggregate accuracy that quietly rejected half of all qualified women, a metric I had chosen for defensible business reasons that a model then gamed by doing nothing useful. Building models turns out to be the accessible part. Knowing when to distrust one is the actual skill.

Alongside this coursework I build AI applications under the K.A.I. (Killingsworth Artificial Intelligence) name — computer vision, agent architectures, and full-stack ML deployment. I placed 2nd in the beginner track at HackHCC Hackathon 2026 and I am currently an AWS AI & ML Scholar on the Agent Developer track.

Interests: AI Agents · Robotics · Algorithms · Video Games · Travel · Animals

---

## Featured Project

### [Airline Tweet Sentiment Analysis →](airline-sentiment-analysis/)

Three-class sentiment classification on 14,427 airline customer tweets, framed as a complaint-triage system. Compares Logistic Regression, Multinomial Naive Bayes, and Linear SVM against a majority-class baseline, with full error analysis and a deployment recommendation.

The headline finding: the model that won my chosen business metric was the worst model for the job. Naive Bayes posted the best complaint recall in the study (96.6%) by predicting "negative" at almost everything — its neutral recall collapsed to 30%. Only a paired guard metric caught it.

> Built as the final project for **ITAI 1371** and included here as a portfolio showcase of the techniques developed across this course. Complete with source module, methodology docs, nine figures, and a results table.

---

## Labs

Every lab below is a completed Jupyter notebook with saved outputs, paired with a README covering what I did, what I learned, and where I got stuck.

| # | Lab | Topic | Dataset |
|---|---|---|---|
| 02 | [Tools Used in Machine Learning](Labs/Lab02/) | Environment setup, Python data stack, Markdown, Git | Iris |
| 03 | [The ML Workflow and Types of Learning](Labs/Lab03/) | Supervised vs. unsupervised vs. reinforcement; end-to-end workflow | Wine |
| 04 | [Exploratory Data Analysis](Labs/Lab04/) | Descriptive statistics, distribution and relationship plots | Titanic |
| 05 | [Data Preparation](Labs/Lab05/) | Imputation, One-Hot Encoding, feature scaling | Titanic |
| 06 | [Regression and Classification Models](Labs/Lab06/) | Linear Regression, Logistic Regression, metric selection | Titanic |
| 07 | [Better Model Evaluation](Labs/Lab07/) | Confusion matrix, precision/recall/F1, cross-validation | Titanic |
| 08 | [The Bias-Variance Tradeoff](Labs/Lab08/) | Underfitting, overfitting, learning curves | Synthetic |
| 09 | [Ensemble Methods](Labs/Lab09/) | Decision Tree vs. Random Forest, feature importance | Iris |
| 10 | [Unsupervised Learning](Labs/Lab10/) | K-Means clustering, elbow method, PCA | Iris |
| 11 | [Hyperparameter Tuning and AutoML](Labs/Lab11/) | Grid Search, Random Search, AutoGluon | Iris |
| 12 | [Ethics, Fairness, and Bias in ML](Labs/Lab12/) | Group-wise FPR/FNR auditing, disparate impact | Adult Census Income |
| 13 | [Building ML Pipelines](Labs/Lab13/) | `Pipeline`, `ColumnTransformer`, leakage prevention | Titanic |

---

## Three things this course actually taught me

**Feature selection outweighs model selection more often than I expected.** In Lab 03, swapping which three Wine features I fed the model — same algorithm, same split, nothing else changed — moved accuracy from 83.3% to 94.4%. I came in assuming the math would be the hard part.

**A good score is a claim that needs auditing, not a result.** Lab 09's Random Forest hit 100% because Iris is too clean to distinguish a good model from a great one. Lab 12's income classifier looked fine in aggregate while producing a 47.8% false negative rate for women. Neither problem is visible in the headline metric.

**The gap between a working notebook and a deployable model is mostly plumbing discipline.** Lab 13's `Pipeline` refactor was the moment that landed — `fit_transform()` on a test set doesn't raise an error, it just quietly inflates your score, which is exactly what makes it dangerous.

---

## Repository structure

```
Krhystopher-Killingsworth-ML-Course/
├── README.md                      ← you are here
├── Labs/
│   ├── Lab02/  README.md + notebook
│   ├── Lab03/  README.md + notebook
│   └── ...     through Lab13
└── airline-sentiment-analysis/    ← featured project
    ├── README.md
    ├── notebooks/
    ├── src/
    ├── figures/
    ├── results/
    └── docs/
```

---

## Tools and libraries

`Python 3` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn` · `AutoGluon` · `Jupyter` / `Google Colab` · `Git` / `GitHub`

---

## Running the notebooks

Every notebook is committed with its outputs saved, so it can be read directly on GitHub without executing anything.

To run one yourself:

```bash
git clone https://github.com/krhys-killingsworth/Krhystopher-Killingsworth-ML-Course.git
cd Krhystopher-Killingsworth-ML-Course
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

Lab 11 additionally requires `autogluon`. The featured project has its own `requirements.txt`.

---

## Contact

**Krhystopher Killingsworth**
GitHub: [@krhys-killingsworth](https://github.com/krhys-killingsworth)
