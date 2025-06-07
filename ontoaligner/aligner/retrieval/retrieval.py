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
This script defines different retrieval models for matching source and target ontologies.
It provides several classes that implement different retrieval techniques, such as Bi-Encoder Retrieval,
ML Retrieval, and the base Retrieval model. These models are used for tasks that require estimating the
similarity between queries and candidate documents, returning the top-k most relevant results.

Classes:
    - Retrieval: A base class for general retrieval tasks.
    - BiEncoderRetrieval: A retrieval model using bi-encoder architectures and SentenceTransformers.
    - MLRetrieval: A retrieval model using SVM and pre-trained SentenceTransformers.
"""

from typing import Any, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ...base import BaseOMModel


class Retrieval(BaseOMModel):
    """
    The base class for retrieval models. This class defines the common interface for loading, fitting,
    transforming, and generating retrieval results. It is intended to be subclassed and extended by more
    specific retrieval models.
    """
    path: str = ""
    model: Any = None

    def __init__(self, device: str='cpu', top_k: int=5, openai_key: str="None", **kwargs) -> None:
        """
        Initializes the Retrieval model.

        Args:
            **kwargs: Additional keyword arguments passed to the superclass constructor.
        """
        super().__init__(device=device, top_k=top_k, openai_key=openai_key, **kwargs)

    def load(self, path: str) -> Any:
        """
        Loads the retrieval model from the specified path.

        Args:
            path (str): Path to the model to be loaded.

        Returns:
            Any: The loaded model.
        """
        pass

    def __str__(self):
        """
        Returns the string representation of the retrieval model.

        Returns:
            str: "Retrieval" as the string representation.
        """
        return "Retrieval"

    def fit(self, inputs: Any) -> Any:
        """
        Fits the model to the provided input data.

        Args:
            inputs (Any): The input data for model training.

        Returns:
            Any: Transformed data after fitting the model.
        """
        pass

    def transform(self, inputs: Any) -> Any:
        """
        Transforms the input data for retrieval.

        Args:
            inputs (Any): The input data to transform.

        Returns:
            Any: Transformed input data.
        """
        pass

    def estimate_similarity(self, query_embed: Any, candidate_embeds: Any) -> Any:
        """
        Estimates the similarity between query and candidate embeddings.

        Args:
            query_embed (Any): The query embedding.
            candidate_embeds (Any): The candidate document embeddings.

        Returns:
            Any: Similarity scores between the query and candidate embeddings.
        """
        pass

    def get_top_k(self, query_embed: Any, candidate_embeds: Any) -> [List, List]:
        """
        Returns the top-k most similar items based on the query and candidate embeddings.

        Args:
            query_embed (Any): The query embedding.
            candidate_embeds (Any): The candidate document embeddings.

        Returns:
            [List, List]: A tuple containing the top-k indexes and corresponding similarity scores.
        """
        results = self.estimate_similarity(query_embed=query_embed, candidate_embeds=candidate_embeds)
        values = [(score, index) for index, score in enumerate(results)]
        dtype = [("score", float), ("index", int)]
        results = np.array(values, dtype=dtype)
        try:
            top_k_items = np.sort(results, order="score")[-self.kwargs["top_k"]:][::-1]
        except IndexError:
            top_k_items = np.sort(results, order="score")[::-1]
        top_k_indexes, top_k_scores = [], []
        for top_k in top_k_items:
            top_k_scores.append(top_k[0])
            top_k_indexes.append(top_k[1])
        return top_k_indexes, top_k_scores

    def generate(self, input_data: List) -> List:
        """
        Generates predictions based on source and target ontologies.

        Args:
            input_data (List): A list containing the source and target ontologies for matching.

        Returns:
            List: A list of predictions with source-IRI, target-candidates, and their scores.
        """
        source_ontology = input_data[0]
        target_ontology = input_data[1]
        predictions = []

        candidates_embedding = self.fit(inputs=[target["text"] for target in target_ontology])
        queries_embedding = self.transform(inputs=[source["text"] for source in source_ontology])

        for source_id, query_embed in tqdm(enumerate(queries_embedding)):
            ids, scores = self.get_top_k(query_embed=query_embed, candidate_embeds=candidates_embedding)
            candidates_iris, candidates_scores = [], []
            for candidate_id, candidate_score in zip(ids, scores):
                candidates_iris.append(target_ontology[candidate_id]["iri"])
                candidates_scores.append(candidate_score)
            if len(candidates_iris) != 0:
                predictions.append(
                    {
                        "source": source_ontology[source_id]["iri"],
                        "target-cands": candidates_iris,
                        "score-cands": candidates_scores,
                    }
                )
        return predictions


class BiEncoderRetrieval(Retrieval):
    """
    A retrieval model using bi-encoder architecture based on SentenceTransformers. This model generates embeddings
    for both queries and candidates and calculates similarity using cosine similarity.
    """
    def load(self, path: str):
        """
        Loads the bi-encoder model (SentenceTransformer) from the specified path.

        Args:
            path (str): Path to the bi-encoder model.

        Returns:
            None
        """
        self.model = SentenceTransformer(path, device=self.kwargs["device"], trust_remote_code=True)

    def fit(self, inputs: Any) -> Any:
        """
        Fits the bi-encoder model on the input data, generating embeddings.

        Args:
            inputs (Any): The input data (texts) for generating embeddings.

        Returns:
            Any: Generated embeddings for the input data.
        """
        return self.model.encode(inputs, show_progress_bar=True, batch_size=16)

    def transform(self, inputs: Any) -> Any:
        """
        Transforms the input data into embeddings using the bi-encoder model.

        Args:
            inputs (Any): The input data (texts) for generating embeddings.

        Returns:
            Any: Generated embeddings for the input data.
        """
        return self.model.encode(inputs, show_progress_bar=True, batch_size=16)

    def generate(self, input_data: List) -> List:
        """
        Generates predictions based on source and target ontologies using bi-encoder retrieval.

        Args:
            input_data (List): A list containing source and target ontologies.

        Returns:
            List: A list of predictions with source-IRI, target-candidates, and their similarity scores.
        """
        source_ontology = input_data[0]
        target_ontology = input_data[1]
        predictions = []

        candidates_embedding = self.fit(inputs=[target["text"] for target in target_ontology])
        queries_embedding = self.transform(inputs=[source["text"] for source in source_ontology])

        estimated_similarity = cosine_similarity(queries_embedding, candidates_embedding)

        for source_id, similarities in tqdm(enumerate(estimated_similarity)):
            values, indexes = torch.topk(torch.Tensor(similarities), k=self.kwargs["top_k"], axis=-1)
            scores = [float(value) for value in values]
            ids = [int(index) for index in indexes]
            candidates_iris, candidates_scores = [], []
            for candidate_id, candidate_score in zip(ids, scores):
                candidates_iris.append(target_ontology[candidate_id]["iri"])
                candidates_scores.append(candidate_score)
            if len(candidates_iris) != 0:
                predictions.append(
                    {
                        "source": source_ontology[source_id]["iri"],
                        "target-cands": candidates_iris,
                        "score-cands": candidates_scores,
                    }
                )
        return predictions


class MLRetrieval(Retrieval):
    """
    A retrieval model using SVM-based classification for matching queries with candidates.
    This model computes similarity scores using SVM and sentence embeddings.
    This retriever is the slowest model. So it should be used for labels based retrieval
    """
    def load(self, path: str):
        """
        Loads the SVM-based retrieval model from the specified path.

        Args:
            path (str): Path to the SVM model.

        Returns:
            None
        """
        self.model = SentenceTransformer(self.path, device=self.kwargs["device"])

    def fit(self, inputs: Any) -> Any:
        """
        Fits the model to the input data, generating sentence embeddings.

        Args:
            inputs (Any): The input data for fitting the model.

        Returns:
            Any: Generated embeddings for the input data.
        """
        return self.model.encode(inputs, show_progress_bar=True, batch_size=16)

    def transform(self, inputs: Any) -> Any:
        """
        Transforms the input data into embeddings using the SVM model.

        Args:
            inputs (Any): The input data (texts) to be transformed.

        Returns:
            Any: Generated embeddings for the input data.
        """
        return self.model.encode(inputs, show_progress_bar=True, batch_size=16)

    def generate(self, input_data: List) -> List:
        """
        Generates predictions based on source and target ontologies using SVM retrieval.

        Args:
            input_data (List): A list containing source and target ontologies.

        Returns:
            List: A list of predictions with source-IRI, target-candidates, and their similarity scores.
        """
        source_ontology = input_data[0]
        target_ontology = input_data[1]
        predictions = []

        candidates_embedding = self.fit(inputs=[target["text"] for target in target_ontology])
        candidates_embedding = candidates_embedding / np.sqrt((candidates_embedding**2).sum(1, keepdims=True))

        queries_embedding = self.transform(inputs=[source["text"] for source in source_ontology])
        queries_embedding = queries_embedding / np.sqrt((queries_embedding**2).sum(1, keepdims=True))

        # q_len = len(source_ontology)
        c_len = len(target_ontology)
        # dim = queries_embedding.shape[1]
        # y = np.zeros(q_len+c_len)
        # y[0:q_len] = 1
        # x = np.concatenate([queries_embedding, candidates_embedding])
        # clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
        # clf.fit(x, y)
        # estimated_similarity = cosine_similarity(queries_embedding, candidates_embedding)

        for source_id, query_embed in tqdm(enumerate(queries_embedding)):
            x = np.concatenate([[query_embed], candidates_embedding])
            y = np.zeros(c_len + 1)
            y[0] = 1
            clf = svm.LinearSVC(
                class_weight="balanced",
                verbose=False,
                max_iter=1000,
                tol=1e-6,
                C=0.1,
                dual="auto",
            )
            clf.fit(x, y)
            similarities = clf.decision_function(x)[1:]
            values, indexes = torch.topk(torch.Tensor(similarities), k=self.kwargs["top_k"], axis=-1)
            scores = [float(value) for value in values]
            ids = [int(index) for index in indexes]
            candidates_iris, candidates_scores = [], []
            for candidate_id, candidate_score in zip(ids, scores):
                candidates_iris.append(target_ontology[candidate_id]["iri"])
                candidates_scores.append(candidate_score)
            if len(candidates_iris) != 0:
                predictions.append(
                    {
                        "source": source_ontology[source_id]["iri"],
                        "target-cands": candidates_iris,
                        "score-cands": candidates_scores,
                    }
                )
        return predictions
