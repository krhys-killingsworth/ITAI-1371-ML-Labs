"""
Airline tweet sentiment classification pipeline.

A standalone, importable implementation of the pipeline developed in
notebooks/FP_AirlineTweets_KrhystopherKillingsworth.ipynb.

The notebook is the analysis and the argument. This module is the reusable
artifact: it exposes the same preprocessing and model configuration as
functions you can import, test, or drop into a service.

Usage
-----
Train and evaluate all models, writing a metrics table to disk:

    python src/sentiment_pipeline.py --report results/metrics.csv

Classify text from the command line:

    python src/sentiment_pipeline.py --predict "3 hour delay and no one will help"

Import into your own code:

    from sentiment_pipeline import SentimentPipeline
    pipe = SentimentPipeline().fit()
    pipe.predict(["my bag never arrived"])          # -> ['negative']
    pipe.predict_with_confidence(["thanks!"])       # -> [('positive', 2.31)]

Author: Krhystopher Killingsworth
License: MIT
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

RANDOM_STATE = 42
SENT_ORDER = ["negative", "neutral", "positive"]

DATA_URL = (
    "https://raw.githubusercontent.com/satyajeetkrjha/"
    "kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
)

# Negation and intensity words rescued from the stopword list.
#
# This is the single most important preprocessing decision in the project.
# A default stopword list strips "not", which collapses "not a good flight"
# and "a good flight" into the identical bag of words. Combined with bigrams
# in the vectorizer, keeping these lets the model learn "not good" as a
# feature distinct from "good".
NEGATION_KEEP = {
    "no", "not", "nor", "never", "none", "cannot", "cant", "wont", "dont",
    "didnt", "doesnt", "isnt", "wasnt", "arent", "werent", "hasnt", "havent",
    "wouldnt", "shouldnt", "couldnt", "without", "against", "very", "too",
    "again", "still", "but", "only", "off",
}

# Airline handles ADDED to the stopword list.
#
# Sentiment correlates strongly with carrier in this dataset (36% negative for
# Virgin America vs 78% for US Airways) and nearly every tweet opens with the
# airline's handle. Leaving them in lets the model learn "mentions US Airways,
# therefore probably negative" is a shortcut that raises test scores while
# failing to generalize to a new carrier. Removing them costs accuracy and
# buys a model that actually classifies language.
AIRLINE_TOKENS = {
    "united", "usairways", "americanair", "southwestair", "jetblue",
    "virginamerica", "delta", "american", "usairway", "airline", "airlines",
}

STOP_WORDS = (set(ENGLISH_STOP_WORDS) - NEGATION_KEEP) | AIRLINE_TOKENS

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_NONALPHA_RE = re.compile(r"[^a-z\s]")
_ELONG_RE = re.compile(r"(.)\1{2,}")
_SPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Normalize a raw tweet into a cleaned, stopword-filtered string.

    Strips HTML entities, URLs, @mentions (including airline handles),
    digits, punctuation, and emoji; lowercases; collapses character
    elongations; and removes stopwords while preserving negation.

    Note that hashtag *words* survive, since only the '#' symbol is removed.
    Text like "#neveragain" is highly sentiment-bearing.

    Parameters
    ----------
    text : str
        Raw tweet text.

    Returns
    -------
    str
        Cleaned text, possibly empty if the tweet was only a handle and a link.
    """
    t = html.unescape(str(text))
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = t.lower()
    t = _NONALPHA_RE.sub(" ", t)
    t = _ELONG_RE.sub(r"\1\1", t)
    tokens = [w for w in t.split() if w not in STOP_WORDS and len(w) > 2]
    return _SPACE_RE.sub(" ", " ".join(tokens)).strip()


def load_data(source: str | Path = DATA_URL) -> pd.DataFrame:
    """Load the airline tweets dataset and apply cleaning.

    Removes duplicate texts *before* any split so that near-identical rows
    cannot straddle the train/test boundary and inflate test scores.

    Parameters
    ----------
    source : str or Path
        URL or local path to Tweets.csv.

    Returns
    -------
    pandas.DataFrame
        Columns: airline_sentiment, airline, text, cleaned_text.
    """
    df = pd.read_csv(source)
    df = df[["airline_sentiment", "airline", "text"]].copy()
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["cleaned_text"] = df["text"].apply(preprocess_text)
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def build_models() -> dict:
    """Return the four models compared in the analysis.

    Hyperparameters are the tuned values from the notebook's grid search
    (macro F1, 5-fold stratified CV on the training split).

    MultinomialNB has no class_weight parameter, so it cannot compensate for
    the 63/21/16 class imbalance. This is deliberate and load-bearing: it is
    why NB posts the highest negative-class recall in the study while having
    the worst macro F1. See the README's "headline result".
    """
    return {
        "Baseline (majority class)": DummyClassifier(
            strategy="most_frequent", random_state=RANDOM_STATE
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Linear SVM": LinearSVC(
            C=0.1, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


@dataclass
class SentimentPipeline:
    """End-to-end TF-IDF + Linear SVM sentiment classifier.

    Defaults to the model recommended in the analysis: a tuned LinearSVC with
    balanced class weights over 5,000 TF-IDF unigram/bigram features.

    Attributes
    ----------
    model_name : str
        Key from build_models(). Defaults to "Linear SVM".
    test_size : float
        Held-out fraction for evaluation.
    """

    model_name: str = "Linear SVM"
    test_size: float = 0.20
    random_state: int = RANDOM_STATE

    vectorizer: TfidfVectorizer = field(init=False, default=None)
    model: object = field(init=False, default=None)
    _fitted: bool = field(init=False, default=False)

    def _new_vectorizer(self) -> TfidfVectorizer:
        # No stop_words parameter here on purpose. Stopwords are already
        # handled by preprocess_text() with the negation-preserving list.
        # Setting it here would re-strip the negation words we rescued.
        return TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.90,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

    def fit(self, df: pd.DataFrame | None = None) -> "SentimentPipeline":
        """Fit the vectorizer and model on a stratified training split."""
        if df is None:
            df = load_data()

        X_train, X_test, y_train, y_test = train_test_split(
            df["cleaned_text"], df["airline_sentiment"],
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df["airline_sentiment"],
        )

        self.vectorizer = self._new_vectorizer()
        # Fit on training data ONLY; the test fold is merely transformed.
        X_train_vec = self.vectorizer.fit_transform(X_train)

        self.model = build_models()[self.model_name]
        self.model.fit(X_train_vec, y_train)

        self._X_test, self._y_test = X_test, y_test
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call .fit() before predicting.")

    def predict(self, texts) -> np.ndarray:
        """Predict sentiment labels for raw (uncleaned) texts."""
        self._check_fitted()
        if isinstance(texts, str):
            texts = [texts]
        vec = self.vectorizer.transform([preprocess_text(t) for t in texts])
        return self.model.predict(vec)

    def predict_with_confidence(self, texts) -> list[tuple[str, float]]:
        """Predict labels alongside a confidence score.

        For LinearSVC, confidence is the *decision margin*: the gap between
        the top two class scores. Large margin means the model placed the text
        far from the boundary between its two best candidates.

        The margin is a usable routing signal. In the notebook's held-out
        evaluation, the most confident quartile scores about 96% accuracy while
        the least confident quartile falls to 58%. Route the confident
        predictions automatically and send the tail to human review.
        """
        self._check_fitted()
        if isinstance(texts, str):
            texts = [texts]
        vec = self.vectorizer.transform([preprocess_text(t) for t in texts])
        preds = self.model.predict(vec)

        if hasattr(self.model, "decision_function"):
            scores = np.atleast_2d(self.model.decision_function(vec))
            ordered = np.sort(scores, axis=1)
            conf = ordered[:, -1] - ordered[:, -2]
        else:
            conf = self.model.predict_proba(vec).max(axis=1)

        return list(zip(preds, conf))

    def evaluate(self, verbose: bool = True) -> dict:
        """Score the fitted model on its held-out test split."""
        self._check_fitted()
        X_vec = self.vectorizer.transform(self._X_test)
        y_pred = self.model.predict(X_vec)

        metrics = {
            "model": self.model_name,
            "accuracy": accuracy_score(self._y_test, y_pred),
            "f1_macro": f1_score(self._y_test, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(self._y_test, y_pred, average="weighted", zero_division=0),
            "recall_negative": recall_score(
                self._y_test, y_pred, labels=["negative"],
                average="macro", zero_division=0,
            ),
        }

        cm = confusion_matrix(self._y_test, y_pred, labels=SENT_ORDER)
        metrics["complaints_missed"] = (cm[0, 1] + cm[0, 2]) / cm[0].sum()
        metrics["neutral_overflagged"] = cm[1, 0] / cm[1].sum()

        if verbose:
            print(f"\n{self.model_name}")
            print("=" * 60)
            print(classification_report(self._y_test, y_pred, digits=3, zero_division=0))
            print(f"Complaints missed   : {metrics['complaints_missed']:.1%}  (expensive error)")
            print(f"Neutral over-flagged: {metrics['neutral_overflagged']:.1%}  (cheap error)")

        return metrics


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def compare_all(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Train every model and return a comparison table."""
    if df is None:
        print("Loading data...")
        df = load_data()
        print(f"Loaded {len(df):,} tweets after cleaning.\n")

    rows = []
    for name in build_models():
        pipe = SentimentPipeline(model_name=name).fit(df)
        rows.append(pipe.evaluate(verbose=False))

    return pd.DataFrame(rows).set_index("model").round(4)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Airline tweet sentiment classification pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predict", metavar="TEXT", nargs="+",
        help="Classify one or more texts and exit.",
    )
    parser.add_argument(
        "--report", metavar="PATH", nargs="?", const="results/metrics.csv",
        help="Train all models and write a metrics table (default: results/metrics.csv).",
    )
    parser.add_argument(
        "--data", metavar="PATH", default=DATA_URL,
        help="Path or URL to Tweets.csv (defaults to the public mirror).",
    )
    args = parser.parse_args(argv)

    if args.predict:
        pipe = SentimentPipeline().fit(load_data(args.data))
        print()
        for text, (label, conf) in zip(args.predict, pipe.predict_with_confidence(args.predict)):
            route = "  -> PRIORITY SUPPORT QUEUE" if label == "negative" else ""
            print(f"[{label.upper():<8} margin {conf:4.2f}] {text}{route}")
        return 0

    df = load_data(args.data)
    print(f"Loaded {len(df):,} tweets after cleaning.")
    results = compare_all(df)

    print("\n" + "=" * 78)
    print("MODEL COMPARISON".center(78))
    print("=" * 78)
    print(results.to_string())

    print("\nNote: Naive Bayes has the best recall_negative but the worst f1_macro.")
    print("It earns that recall by predicting 'negative' at almost everything.")
    print("See the README for why this is the central result of the project.")

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out)
        print(f"\nWrote metrics to {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
