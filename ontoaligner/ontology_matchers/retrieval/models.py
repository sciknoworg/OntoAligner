# -*- coding: utf-8 -*-
"""
This script defines various retrieval models used for information retrieval tasks.
It includes both traditional methods (such as TF-IDF and BM25) as well as more modern
approaches using bi-encoder architectures and pre-trained models. The models are designed
to compute similarity scores between a query and candidate documents.

Classes:
    - BERTRetrieval: A retrieval class extending BiEncoderRetrieval using BERT-based encoding.
    - FlanT5Retrieval: A retrieval class extending BiEncoderRetrieval using Flan-T5 model encoding.
    - TFIDFRetrieval: A retrieval class using TF-IDF vectorization for document similarity estimation.
    - BM25Retrieval: A retrieval class using BM25 (Okapi BM25) model for document similarity estimation.
    - SVMBERTRetrieval: A retrieval class extending MLRetrieval using SVM-based BERT retrieval.
    - AdaRetrieval: A retrieval class using embeddings loaded from pre-trained OpenAI models.
"""

import os
from typing import Any
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .retrieval import BiEncoderRetrieval, MLRetrieval, Retrieval
from ...utils import io


class BERTRetrieval(BiEncoderRetrieval):
    """
    BERTRetrieval is a subclass of BiEncoderRetrieval that uses a BERT-based encoder
    for retrieval tasks. This class implements a method for returning the string representation
    of the retrieval model, appending the specific model's name.
    """

    def __str__(self):
        """
        Returns the string representation of the BERTRetrieval model.

        Returns:
            str: The string "+BERTRetrieval" appended to the superclass string.
        """
        return super().__str__() + "+BERTRetrieval"


class FlanT5Retrieval(BiEncoderRetrieval):
    """
    FlanT5Retrieval is a subclass of BiEncoderRetrieval that uses the Flan-T5 encoder
    for retrieval tasks. This class implements a method for returning the string representation
    of the retrieval model, appending the specific model's name.
    """

    def __str__(self):
        """
        Returns the string representation of the FlanT5Retrieval model.

        Returns:
            str: The string "+FlanT5XLRetrieval" appended to the superclass string.
        """
        return super().__str__() + "FlanT5XLRetrieval"


class TFIDFRetrieval(Retrieval):
    """
    TFIDFRetrieval implements the TF-IDF vectorization method for document retrieval.
    It allows for fitting a TF-IDF model to input data, transforming input data into feature vectors,
    and estimating the similarity between query and candidate documents using cosine similarity.
    """

    def load(self, path: str = None):
        """
        Loads the TF-IDF vectorizer model.

        Args:
            path (str, optional): The path to load the model from (default is None).
        """
        self.model = TfidfVectorizer()

    def fit(self, inputs: Any) -> Any:
        """
        Fits the TF-IDF model on the input data and transforms it into feature vectors.

        Args:
            inputs (Any): The input data to fit the model on.

        Returns:
            Any: Transformed feature vectors based on the input data.
        """
        self.model.fit(inputs)
        return self.transform(inputs=inputs)

    def transform(self, inputs: Any) -> Any:
        """
        Transforms the input data into TF-IDF feature vectors.

        Args:
            inputs (Any): The input data to transform.

        Returns:
            Any: Transformed TF-IDF feature vectors.
        """
        return self.model.transform(inputs)

    def estimate_similarity(self, query_embed: Any, candidate_embeds: Any) -> Any:
        """
        Estimates the cosine similarity between the query and candidate embeddings.

        Args:
            query_embed (Any): The query embedding.
            candidate_embeds (Any): The candidate embeddings.

        Returns:
            Any: Cosine similarity scores between the query and candidate embeddings.
        """
        return cosine_similarity(query_embed, candidate_embeds).reshape((-1,))

    def __str__(self):
        """
        Returns the string representation of the TFIDFRetrieval model.

        Returns:
            str: The string "+TFIDFRetrieval" appended to the superclass string.
        """
        return super().__str__() + "+TFIDFRetrieval"


class BM25Retrieval(Retrieval):
    """
    BM25Retrieval implements the BM25 retrieval model (Okapi BM25), a probabilistic information retrieval method.
    This model is used to estimate document relevance based on term frequency and inverse document frequency.
    http://ethen8181.github.io/machine-learning/search/bm25_intro.html
    """

    def load(self, path: str = None):
        """
        Loads the BM25 model. In this implementation, no additional loading is needed.

        Args:
            path (str, optional): Path to load model from (default is None).
        """
        pass

    def fit(self, inputs: Any) -> Any:
        """
        Tokenizes the input documents and fits the BM25 model.

        Args:
            inputs (Any): The input data (documents) to fit the model on.

        Returns:
            None
        """
        tokenized_inputs = [input.split(" ") for input in inputs]
        self.model = BM25Okapi(tokenized_inputs)
        return None

    def transform(self, inputs: Any) -> Any:
        """
        Tokenizes the input data.

        Args:
            inputs (Any): The input data to tokenize.

        Returns:
            Any: Tokenized input data.
        """
        return [input.split(" ") for input in inputs]

    def estimate_similarity(self, query_embed: Any, candidate_embeds: Any) -> Any:
        """
        Estimates similarity scores between the query and candidate documents using BM25.

        Args:
            query_embed (Any): The query embedding or tokens.
            candidate_embeds (Any): The candidate document embeddings or tokens.

        Returns:
            Any: BM25 similarity scores between the query and candidate documents.
        """
        docs_scores = self.model.get_scores(query_embed)
        return docs_scores

    def __str__(self):
        """
        Returns the string representation of the BM25Retrieval model.

        Returns:
            str: The string "+BM25Retrieval" appended to the superclass string.
        """
        return super().__str__() + "+BM25Retrieval"


class SVMBERTRetrieval(MLRetrieval):
    """
    SVMBERTRetrieval is a subclass of MLRetrieval that uses a Support Vector Machine (SVM)
    combined with BERT-based embeddings for retrieval tasks.
    """

    def __str__(self):
        """
        Returns the string representation of the SVMBERTRetrieval model.

        Returns:
            str: The string "+SVMBERTRetrieval" appended to the superclass string.
        """
        return super().__str__() + "+SVMBERTRetrieval"


class AdaRetrieval(BiEncoderRetrieval):
    """
    AdaRetrieval is a subclass of BiEncoderRetrieval that uses pre-trained embeddings from OpenAI.
    It is designed to load embeddings from files, fit them, and transform input data into corresponding embeddings.
    """

    def __str__(self):
        """
        Returns the string representation of the AdaRetrieval model.

        Returns:
            str: The string "+AdaRetrieval" appended to the superclass string.
        """
        return super().__str__() + "+AdaRetrieval"

    def load(self, path: str):
        """
        Loads the pre-trained OpenAI embeddings and label-to-index mappings from files.

        Args:
            path (str): The directory path where the embeddings and labels are stored.
        """
        self.model = np.load(os.path.join(self.path, "openai_embeddings.npy"))
        self.labels2index = io.read_json(os.path.join(self.path, "labels2index.json"))

    def fit(self, inputs: Any) -> Any:
        """
        Fits the model by transforming the input data into corresponding embeddings.

        Args:
            inputs (Any): The input data to fit the model on.

        Returns:
            Any: Transformed embeddings based on the input data.
        """
        return self.transform(inputs=inputs)

    def transform(self, inputs: Any) -> Any:
        """
        Transforms input data into embeddings based on pre-trained OpenAI model.

        Args:
            inputs (Any): The input data (strings) to transform into embeddings.

        Returns:
            np.array: An array of embeddings for the input data.
        """
        embeddings = []
        for input_text in tqdm(inputs):
            index = self.labels2index.get(input_text, 0)
            if index == 0:
                print("NO match for the string:", input_text)
            embeddings.append(self.model[index])
        return np.array(embeddings)
