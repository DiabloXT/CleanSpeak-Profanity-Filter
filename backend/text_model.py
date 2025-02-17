import re
import joblib
import string
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk

from nltk.corpus import wordnet
from nltk.corpus import stopwords

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


class TrainTextModel:
    def __init__(self):
        nltk.download("all")

    def preprocess_text(self, input_text):
        text = str(input_text).lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))

        cleaned_text = re.sub(r"http\S+", "", text)
        cleaned_text = re.sub(r"[^A-Za-z0-9]+", " ", cleaned_text)
        cleaned_text = re.sub(r"<[^>]*>", "", cleaned_text)
        cleaned_text = re.sub(r"[^\w\s]", " ", cleaned_text)
        cleaned_text = re.sub(r"\d+", "", cleaned_text)

        cleaned_text = cleaned_text.encode("ascii", "ignore").decode()

        tokens = word_tokenize(cleaned_text)
        filtered_tokens = [
            word for word in tokens if word not in set(stopwords.words("english"))
        ]

        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

        return " ".join(lemmatized_tokens).strip()

    def train_text_profanity_filter(self):
        with zipfile.ZipFile("data/text_profanity_data.zip", "r") as zip_ref:
            zip_ref.extractall("data")

        data = pd.read_csv(
            "data/train.csv",
            usecols=[
                "comment_text",
                "toxic",
                "severe_toxic",
                "obscene",
                "threat",
                "insult",
                "identity_hate",
            ],
        )

        count_duplicates = data.duplicated().sum()

        if count_duplicates > 0:
            data.drop_duplicates(inplace=True)

        text_prof = TrainTextModel()
        data["cleaned_comment_text"] = data["comment_text"].apply(text_prof.preprocess_text)

        X = data["comment_text"]
        y = data[
            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        ]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=50,
        )

        vectorizer = TfidfVectorizer(stop_words="english")

        X_train_tfidf = vectorizer.fit_transform(X_train)
        joblib.dump(vectorizer, "model/tfidf_embeddings.joblib")

        X_test_tfidf = vectorizer.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, verbose=3
        )
        clf.fit(X_train_tfidf, y_train)

        model_filename = "model/text_profanity_filter_rfc_model.joblib"
        joblib.dump(clf, model_filename)
