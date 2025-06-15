# Copyright 2025 Scientific Knowledge Organization (SciKnowOrg) Research Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script provides an implementation of label mapping using different machine learning approaches.
It defines a base `LabelMapper` class and two specific subclasses:
- `TFIDFLabelMapper`: Uses a TfidfVectorizer and a classifier for label prediction.
- `SetFitShallowLabelMapper`: Uses a pretrained SetFit model for label prediction.
"""
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class LabelMapper:
    """
    Base class for label mapping, providing common functionality for derived classes.
    """
    def __init__(self, label_dict: Dict[str, List[str]] = None, iterator_no: int = 10):
        """
        Initializes the label mapper with training data and labels.

        Parameters:
            label_dict (Dict[str, List[str]]): Dictionary mapping each label to a list of candidate phrases.
            iterator_no (int): Number of iterations to replicate training data for better generalization.
        """
        if label_dict is None:
            label_dict = {
                "yes": ["yes", "correct", "true"],
                "no": ["no", "incorrect", "false"]
            }
        self.labels = [label.lower() for label in list(label_dict.keys())]
        self.x_train, self.y_train = [], []
        for label, candidates in label_dict.items():
            self.x_train += [label] + candidates
            self.y_train += [label] * (len(candidates) + 1)
        self.x_train = iterator_no * self.x_train
        self.y_train = iterator_no * self.y_train
        assert len(self.x_train) == len(self.y_train)

    def fit(self):
        """Placeholder for model fitting logic, implemented by subclasses."""
        pass

    def validate_predicts(self, preds: List[str]):
        """
        Validates if predictions are among valid labels.

        Parameters:
            preds (List[str]): List of predicted labels.
        """
        for pred in preds:
            if pred.lower() not in self.labels:
                raise AssertionError(f"{pred} in prediction is not a valid label!")

    def predict(self, X: List[str]) -> List[str]:
        """
        Predicts labels for the given input.

        Parameters:
            X (List[str]): List of input texts to classify.

        Returns:
            List[str]: Predicted labels.
        """
        predictions = list(self._predict(X))
        self.validate_predicts(predictions)
        return predictions

    def _predict(self, X: List[str]) -> List[str]:
        """Placeholder for the prediction logic, implemented by subclasses."""
        pass


class TFIDFLabelMapper(LabelMapper):
    """
    LabelMapper subclass using a TF-IDF vectorizer and a classifier for label prediction.
    """
    def __init__(self, classifier: Any, ngram_range: Tuple, label_dict: Dict[str, List[str]]=None,
                 analyzer: str = 'word', iterator_no: int = 10):
        """
        Initializes the TFIDFLabelMapper with a specified classifier and TF-IDF configuration.

        Parameters:
            classifier (Any): Classifier object (e.g., LogisticRegression, SVC).
            ngram_range (Tuple): Range of n-grams for the TF-IDF vectorizer.
            label_dict (Dict[str, List[str]]): Dictionary mapping each label to a list of candidate phrases.
            analyzer (str): Specifies whether to analyze at the 'word' or 'char' level.
            iterator_no (int): Number of iterations to replicate training data.
        """
        super().__init__(label_dict, iterator_no)
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)),
            ('classifier', classifier)
        ])

    def fit(self):
        """Fits the TF-IDF pipeline on the training data."""
        self.model.fit(self.x_train, self.y_train)

    def _predict(self, X: List[str]) -> List[str]:
        """
        Predicts labels for the given input using the TF-IDF pipeline.

        Parameters:
            X (List[str]): List of input texts to classify.

        Returns:
            List[str]: Predicted labels.
        """
        return self.model.predict(X)


class SBERTLabelMapper(LabelMapper):
    """
    LabelMapper subclass using SentenceTransformer embeddings and a classifier for label prediction.

    Example usage:
    >>> label_dict = {
    >>>        "yes":["yes", "correct", "true"],
    >>>        "no":["no", "incorrect", "false"]
    >>>    }
    >>> mapper = SBERTLabelMapper("all-MiniLM-L12-v2", label_dict)
    >>> mapper.fit()
    >>> mapper.predict(["yes", "correct", "false", "nice", "too bad", "very good"])
    ['yes', 'yes', 'no', 'yes', 'no', 'yes']
    """
    def __init__(self, model_id: str, label_dict: Dict[str, List[str]], classifier=None, iterator_no: int = 10):
        """
        Initializes the SBERTLabelMapper.

        Parameters:
            model_id (str): Name of the pretrained SentenceTransformer model.
            label_dict (Dict[str, List[str]]): Dictionary mapping each label to a list of candidate phrases.
            iterator_no (int): Number of iterations to replicate training data.
        """
        super().__init__(label_dict, iterator_no)
        self.embedder = SentenceTransformer(model_id)
        self.classifier = classifier or LogisticRegression()
        self.label_encoder = LabelEncoder()

    def fit(self):
        """Fits the classifier on the sentence embeddings."""
        embeddings = self.embedder.encode(self.x_train, convert_to_numpy=True)
        y_encoded = self.label_encoder.fit_transform(self.y_train)
        self.classifier.fit(embeddings, y_encoded)

    def _predict(self, X: List[str]) -> List[str]:
        """
        Predicts labels using the sentence transformer + classifier pipeline

        Parameters:
            X (List[str]): List of input texts to classify.

        Returns:
            List[str]: Predicted labels.
        """
        embeddings = self.embedder.encode(X, convert_to_numpy=True)
        y_pred_encoded = self.classifier.predict(embeddings)
        return [str(pred) for pred in self.label_encoder.inverse_transform(y_pred_encoded)]
