# Methodology

Supporting detail for the decisions summarized in the [README](../README.md). The full narrative analysis lives in [the notebook](../notebooks/FP_AirlineTweets_KrhystopherKillingsworth.ipynb); this document is the condensed reference for *why* each choice was made.

---

## 1. Dataset

**Twitter US Airline Sentiment**, collected by CrowdFlower (now Appen), distributed via Kaggle, loaded here from a public GitHub mirror.

| Property | Value |
|---|---|
| Raw rows | 14,640 |
| After de-duplication | 14,427 |
| After dropping empty post-cleaning rows | 14,365 |
| Collection window | 17–24 February 2015 |
| Airlines | United, US Airways, American, Southwest, Delta, Virgin America |
| Labels | `negative` (63%), `neutral` (21%), `positive` (16%) |

**Why this dataset over IMDb or Amazon reviews:**

- **Three-class, not binary.** Neutral is where a real triage system fails. A binary dataset would have hidden the hardest part of the problem, and in fact the neutral class produced the project's most interesting finding.
- **Genuine in-the-wild text.** Carries the noise a deployed model actually faces: mentions, hashtags, URLs, emoji, typos, inconsistent casing.
- **Rich metadata.** Airline name, annotator confidence, and a `negativereason` field for negative tweets, which turns a modeling exercise into a business analysis.
- **Small enough to iterate.** ~3 MB trains in seconds on CPU, leaving budget for tuning and error analysis rather than waiting on fits.

**Data quality notes:**

- `negativereason` is null for all non-negative rows. Expected by design rather than a defect, because the annotation task only asked for a reason when sentiment was negative.
- 213 duplicate texts (retweets, boilerplate replies) removed **before splitting**, so near-identical rows cannot straddle the train/test boundary and inflate test scores.
- 62 tweets reduce to empty strings after cleaning (they consisted only of a handle and a link) and are dropped.

---

## 2. The metric decision

This was made **before** any modeling, and it drove everything downstream.

The two error types are not equally expensive:

| Error | Consequence | Cost |
|---|---|---|
| Missing a complaint (negative → neutral/positive) | Angry customer silently dropped from the support queue | **Expensive** |
| Over-flagging (neutral → negative) | Agent spends thirty seconds reading a harmless tweet | Cheap |

So: **recall on the negative class** as the headline metric, **macro F1** as a guard, plain accuracy reported for completeness only.

**Why accuracy is nearly useless here.** The majority-class baseline scores **63.1% accuracy** by always predicting negative. Any accuracy figure must be read against that floor. This is why a formal `DummyClassifier` is fitted in the notebook rather than assumed.

**Why the guard metric was load-bearing.** Recall on a single class is trivially gameable, since a model that predicts that class always scores 100%. Macro F1 weights all three classes equally and cannot be gamed the same way. This is not a theoretical concern: Multinomial Naive Bayes gamed the headline metric on this exact dataset (96.6% negative recall, 30% neutral recall, worst macro F1 of any real model). Without the guard, it would have been selected.

---

## 3. Preprocessing

| Step | Rationale |
|---|---|
| Decode HTML entities | The scrape left raw `&amp;`; unfixed it becomes a junk token `amp` |
| **Strip `@mentions` including airline handles** | See §3.1, the most consequential decision here |
| Remove URLs | No sentiment content; every unique URL would be its own useless feature |
| Lowercase | Collapses casing variants, roughly halves vocabulary. Cost: loses ALL-CAPS anger |
| Strip digits | Flight numbers and times are high-cardinality noise |
| Strip punctuation, **keep hashtag words** | `#neveragain` is sentiment-dense; the `#` symbol is not |
| Collapse elongations (`sooooo` → `soo`) | Normalizes a very common Twitter pattern onto a shared token |
| **Negation-preserving stopword removal** | See §3.2 |
| Drop tokens ≤ 2 characters | Removes residual fragments |

Net effect: ~56% token reduction while retaining sentiment-bearing content.

### 3.1 Removing airline handles

Nearly every tweet opens with the airline's handle, and sentiment correlates strongly with carrier:

| Airline | % negative | Tweet volume |
|---|---|---|
| US Airways | 77.8% | 2,913 |
| American | 71.9% | 2,759 |
| United | 69.2% | 3,822 |
| Southwest | 49.2% | 2,420 |
| Delta | 43.2% | 2,222 |
| Virgin America | 36.0% | 504 |

Leaving `@usairways` in the text lets the model learn *"this mentions US Airways, therefore probably negative."* That is a real statistical pattern that would have **raised** test scores. It is also not sentiment analysis. It would fail on a compliment to US Airways and be useless for a new carrier with no prior distribution.

Verification: §5 coefficient inspection confirms no airline name appears among the learned features.

*(Separate caveat on the table above: Virgin America has 504 tweets versus United's 3,822, and was a smaller carrier with a younger, more brand-loyal customer base. The comparison confounds service quality with fleet size and demographics. It supports a claim about public sentiment volume on Twitter, not about which airline is objectively better.)*

### 3.2 Preserving negation

Standard stopword lists strip `not`, `no`, `never`, `cannot`, `don't`. For sentiment analysis this is destructive: *"not a good flight"* and *"a good flight"* reduce to identical bags of words.

Implementation: subtract a 29-word negation/intensity set from scikit-learn's 318-word list before applying it (18 of the 29 were actually present in the base list), then add the 11 airline tokens. Final list: 311 words.

Combined with `ngram_range=(1,2)`, this lets the model learn `not good` as a feature distinct from `good`.

**Verification and its limit:** `not` does end up as one of the strongest negative predictors, so the decision measurably paid off. But error analysis showed the model still confidently reads *"no thanks"* as positive, because `thanks` is such a strong positive feature that the rare `no thanks` bigram cannot outweigh it. **Preserving negation was necessary but not sufficient.** A proper fix requires a model that represents word order and scope, not more cleaning.

---

## 4. Feature extraction and splitting

**Split before vectorizing.** `TfidfVectorizer` learns vocabulary *and* IDF weights from whatever it is fit on. Fitting on the full dataset leaks test-set vocabulary distribution into training and produces optimistic scores. The vectorizer is fit on the training fold only; the test fold is merely transformed.

**Stratified 80/20 split**, seed 42. With a minority class at 16%, an unstratified split could meaningfully skew the test set. Verified: both folds preserve 63.1 / 21.0 / 15.9.

**TF-IDF parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `max_features` | 5000 | Memory control, and acts as regularization by discarding the long tail |
| `ngram_range` | (1, 2) | The only mechanism by which bag-of-words can represent negation |
| `min_df` | 2 | Drops typos and one-off strings that cannot generalize |
| `max_df` | 0.90 | Drops terms appearing in >90% of documents as uninformative |
| `sublinear_tf` | True | `1 + log(tf)`; on tweets, a word appearing 3× is not 3× as important |
| `stop_words` | **not set** | Deliberately omitted, since it is already handled by the negation-preserving list. Setting it here would undo §3.2 |

Resulting matrix: 11,492 × 5,000 at 0.17% density, or 0.8 MB sparse versus ~460 MB dense. Sparse representation is not optional for text.

**Why TF-IDF over raw counts:** term frequency alone over-weights words appearing everywhere. IDF discounts by document frequency, pushing "flight" (in most tweets regardless of sentiment) down and "cancelled" / "wonderful" up. Sentiment lives in the rare emotional words, not the common logistical ones.

---

## 5. Modeling

**Four models**, chosen so the comparison is informative rather than arbitrary:

| Model | Assumption | Why included |
|---|---|---|
| `DummyClassifier` | Always predict majority class | Establishes the 63.1% accuracy floor |
| `MultinomialNB` | Features conditionally independent given class | Classic text baseline; **no `class_weight` support** |
| `LogisticRegression` | Classes linearly separable in TF-IDF space | Calibrated probabilities, interpretable coefficients |
| `LinearSVC` | Maximum-margin hyperplane separates classes | Usually strongest linear model on sparse text |

**Imbalance handling:** `class_weight='balanced'` on LR and SVM scales each class's loss contribution inversely to frequency. `MultinomialNB` has no equivalent, so it enters at a structural disadvantage. That is kept deliberately, because what it costs is the central result.

Alternatives considered and rejected: SMOTE and random resampling (synthetic text is dubious, and undersampling would discard 60% of the data for no benefit at this scale).

**Random Forest was considered and rejected** in favor of Linear SVM as the third model. With 5,000 sparse features, tree ensembles must repeatedly split on individual near-zero-variance terms, which is slow and weak compared to a linear boundary. Maximum-margin classifiers are the standard strong baseline for sparse high-dimensional text.

**Selection protocol:** 5-fold stratified CV on the training set, scored on macro F1, *before* touching the test set. Then grid search on `C` for the two strongest candidates.

**Tuning changed the answer.** At default settings Logistic Regression edged out Linear SVM (0.711 vs 0.698 CV macro F1). After tuning, the SVM overtook it (0.723 vs 0.711), because its default `C=1.0` was badly over-fit for this feature space, with the optimum at `C=0.1`. An untuned comparison would have pointed at the wrong model.

Logistic Regression's sensitivity to `C` also confirmed that regularization matters here: macro F1 degrades monotonically from `C=1.0` (0.711) to `C=10.0` (0.688).

---

## 6. Evaluation and findings

Final held-out results are in [`results/metrics.csv`](../results/metrics.csv) and the README table.

### The tradeoff curve

The three real models sit at three points on the *same* curve rather than being ranked:

| Model | Complaints missed | Neutral over-flagged |
|---|---|---|
| Naive Bayes | 3.4% | 62.8% |
| Linear SVM | 12.5% | 34.2% |
| Logistic Regression | 22.0% | 22.1% |

No model is strictly better. The business cost structure picks the setting, and since missed complaints are the expensive error, the right choice is on the recall side of balanced.

### Coefficient inspection caught what metrics could not

Reading the learned weights surfaced two things invisible in any score table:

1. **Confirmation the handle-stripping worked.** No airline name appears among the top features for any class.
2. **A spurious-correlation problem in the neutral class.** Its top features are city names, campaign hashtags, and topical fragments rather than sentiment words. The model isn't detecting absence of sentiment; it's detecting topic as a proxy. This means real-world neutral performance is likely worse than the test score suggests, since a February 2015 promo hashtag has no predictive value in any other month.

### Confidence as a routing signal

`LinearSVC` provides a decision margin (gap between top-two class scores) rather than probabilities. That margin separates correct from incorrect predictions:

- Mean margin when correct: 0.993
- Mean margin when wrong: 0.466

| Band | Accuracy |
|---|---|
| Top 10% | 96.2% |
| Top 25% | 95.8% |
| Top 50% | 91.6% |
| Bottom 25% | 57.0% |

Operational implication: auto-route confident predictions, send the low-margin tail to human review, feed those decisions back as training data. This also partly mitigates sarcasm. The model can't detect irony, but it often detects that it's unsure, and the sarcastic test case lands in the low-margin band.

### Documented failure modes

- **Short "thanks" replies labeled neutral, predicted positive,** including "no thanks" and "no thank you." The dominant confident-error pattern.
- **Sarcasm.** *"Great, another delay. Just what I wanted today."* Lexically positive, semantically furious.
- **Mixed sentiment.** The single-label task forces a choice the text doesn't support.
- **Emoji-carried meaning.** One confident error hinges on a winking emoji stripped as non-alphabetic. A preprocessing decision with a real cost.
- **Context-free fragments.** A bare case ID is genuinely unclassifiable from text alone.
- **Label ambiguity.** Annotator confidence is lowest on neutral (0.823 vs 0.933 for negative), so some measured error is inherited noise rather than model failure. Low-confidence rows were deliberately **not** filtered: dropping them would measure performance on an artificially easy subset that doesn't resemble the production stream.

---

## 7. Reproducibility

- Seed fixed at **42** throughout: `train_test_split`, all CV folds, every model initialization.
- Vectorizer fit exclusively on the training fold.
- Duplicates removed before splitting.
- No filtering on annotator confidence.
- Dataset loads from a public URL; no manual download.
- `src/sentiment_pipeline.py` reproduces the notebook's held-out numbers exactly.
