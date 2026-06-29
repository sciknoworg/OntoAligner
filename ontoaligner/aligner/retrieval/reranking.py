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
This script defines reranking models for refining retrieved ontology alignment candidates.
The reranking models operate after candidate generation and preserve the retrieval output
format expected by the existing postprocessors.

Classes:
    - Reranking: A base class for reranking retrieved ontology alignment candidates.
    - CohereReranking: A reranking class using the Cohere Rerank API.
    - CrossEncoderReranking: A reranking class using SentenceTransformers CrossEncoder models.
"""

from typing import Any, List

import cohere
import numpy as np
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from ...base import BaseOMModel


class Reranking(BaseOMModel):
    """
    The base class for reranking models. This class defines the common interface for loading,
    scoring, normalizing, and generating reranked retrieval results.
    """
    path: str = ""
    model: Any = None

    def __init__(self, device: str='cpu', top_k: int=5, normalize_score: str="none", **kwargs) -> None:
        """
        Initializes the Reranking model.

        Args:
            **kwargs: Additional keyword arguments passed to the superclass constructor.
        """
        super().__init__(device=device, top_k=top_k, normalize_score=normalize_score, **kwargs)

    def load(self, path: str) -> Any:
        """
        Loads the reranking model from the specified path.

        Args:
            path (str): Path to the model to be loaded.

        Returns:
            Any: The loaded model.
        """
        pass

    def __str__(self):
        """
        Returns the string representation of the reranking model.

        Returns:
            str: "Reranking" as the string representation.
        """
        return "Reranking"

    def rerank_candidates(self, query: str, documents: List[str]) -> List:
        """
        Reranks candidate documents against the provided query.

        Args:
            query (str): The source concept text.
            documents (List[str]): List of candidate target concept texts.

        Returns:
            List: A list of reranking scores aligned with the candidate documents.
        """
        pass

    def normalize(self, scores: List[float]) -> List[float]:
        """
        Normalizes reranking scores using the configured normalization method.

        Args:
            scores (List[float]): The scores returned by the reranking model.

        Returns:
            List[float]: The normalized scores.
        """
        if len(scores) == 0:
            return []

        scores = np.array(scores, dtype=float)
        normalize_score = self.kwargs.get("normalize_score", "none")

        if normalize_score == "none":
            return scores.tolist()

        if normalize_score == "sigmoid":
            return (1 / (1 + np.exp(-scores))).tolist()

        if normalize_score == "softmax":
            exp_scores = np.exp(scores - np.max(scores))
            return (exp_scores / exp_scores.sum()).tolist()

        if normalize_score == "minmax":
            min_score = np.min(scores)
            max_score = np.max(scores)
            if max_score == min_score:
                return [1.0 for _ in scores]
            return ((scores - min_score) / (max_score - min_score)).tolist()

        raise ValueError(f"Unknown score normalization method: {normalize_score}")

    def get_top_k(self, scores: List[float]) -> [List, List]:
        """
        Returns the top-k most relevant items based on reranking scores.

        Args:
            scores (List[float]): The reranking scores.

        Returns:
            [List, List]: A tuple containing the top-k indexes and corresponding reranking scores.
        """
        if len(scores) == 0:
            return [], []

        values = [(score, index) for index, score in enumerate(scores)]
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
        Generates reranked predictions based on source ontology, target ontology, and retrieval outputs.

        Args:
            input_data (List): A list containing source ontology, target ontology, and retrieval outputs.

        Returns:
            List: A list of predictions with source-IRI, reranked target-candidates, and their scores.
        """
        source_ontology = input_data[0]
        target_ontology = input_data[1]
        retrieval_outputs = input_data[2]
        predictions = []

        source_iri2text = {source["iri"]: source.get("text", " ") for source in source_ontology}
        target_iri2text = {target["iri"]: target.get("text", " ") for target in target_ontology}

        for retrieval_output in tqdm(retrieval_outputs):
            source_iri = retrieval_output["source"]
            target_cands = retrieval_output["target-cands"]
            score_cands = retrieval_output.get("score-cands", [])

            query = source_iri2text.get(source_iri, " ") or " "
            documents = [target_iri2text.get(target, " ") or " " for target in target_cands]

            if len(documents) == 0:
                continue

            reranking_scores = self.rerank_candidates(query=query, documents=documents)
            reranking_scores = self.normalize(scores=reranking_scores)

            ids, scores = self.get_top_k(scores=reranking_scores)
            candidates_iris, candidates_scores, retriever_scores = [], [], []

            for candidate_id, candidate_score in zip(ids, scores):
                candidates_iris.append(target_cands[candidate_id])
                candidates_scores.append(float(candidate_score))
                if len(score_cands) > candidate_id:
                    retriever_scores.append(score_cands[candidate_id])

            if len(candidates_iris) != 0:
                prediction = {
                    "source": source_iri,
                    "target-cands": candidates_iris,
                    "score-cands": candidates_scores,
                }
                if len(retriever_scores) == len(candidates_iris):
                    prediction["retriever-score-cands"] = retriever_scores
                predictions.append(prediction)

        return predictions


class CohereReranking(Reranking):
    """
    CohereReranking is a subclass of Reranking that uses the Cohere Rerank API
    for reranking retrieved ontology alignment candidates.
    """
    def __init__(self, device: str='cpu', top_k: int=5, normalize_score: str="none", cohere_key: str="None", **kwargs) -> None:
        """
        Initializes the CohereReranking model.

        Args:
            **kwargs: Additional keyword arguments passed to the superclass constructor.
        """
        super().__init__(
            device=device,
            top_k=top_k,
            normalize_score=normalize_score,
            cohere_key=cohere_key,
            **kwargs,
        )

    def load(self, path: str = "rerank-v3.5"):
        """
        Loads the Cohere reranking client.

        Args:
            path (str): Cohere rerank model name.

        Returns:
            None
        """
        self.path = path
        self.model = cohere.ClientV2(api_key=self.kwargs.get("cohere_key", "None"))

    def rerank_candidates(self, query: str, documents: List[str]) -> List:
        """
        Reranks candidate documents using the Cohere Rerank API.

        Args:
            query (str): The source concept text.
            documents (List[str]): List of candidate target concept texts.

        Returns:
            List: A list of reranking scores aligned with the candidate documents.
        """
        if len(documents) == 0:
            return []

        response = self.model.rerank(
            model=self.path,
            query=query,
            documents=documents,
            top_n=len(documents),
        )

        scores = [0.0 for _ in documents]
        for result in response.results:
            scores[result.index] = float(result.relevance_score)

        return scores

    def __str__(self):
        """
        Returns the string representation of the CohereReranking model.

        Returns:
            str: The string "+CohereReranking" appended to the superclass string.
        """
        return super().__str__() + "+CohereReranking"


class CrossEncoderReranking(Reranking):
    """
    CrossEncoderReranking is a subclass of Reranking that uses a SentenceTransformers
    CrossEncoder model for reranking retrieved ontology alignment candidates.
    """
    def __init__(self, device: str='cpu', top_k: int=5, normalize_score: str="sigmoid", **kwargs) -> None:
        """
        Initializes the CrossEncoderReranking model.

        Args:
            **kwargs: Additional keyword arguments passed to the superclass constructor.
        """
        super().__init__(device=device, top_k=top_k, normalize_score=normalize_score, **kwargs)

    def load(self, path: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        """
        Loads the CrossEncoder reranking model.

        Args:
            path (str): Path to the CrossEncoder model.

        Returns:
            None
        """
        self.path = path
        self.model = CrossEncoder(path, device=self.kwargs["device"])

    def rerank_candidates(self, query: str, documents: List[str]) -> List:
        """
        Reranks candidate documents using a CrossEncoder model.

        Args:
            query (str): The source concept text.
            documents (List[str]): List of candidate target concept texts.

        Returns:
            List: A list of reranking scores aligned with the candidate documents.
        """
        if len(documents) == 0:
            return []

        pairs = [[query, document] for document in documents]
        scores = self.model.predict(
            pairs,
            batch_size=self.kwargs.get("batch_size", 16),
        )

        return [float(score) for score in scores]

    def __str__(self):
        """
        Returns the string representation of the CrossEncoderReranking model.

        Returns:
            str: The string "+CrossEncoderReranking" appended to the superclass string.
        """
        return super().__str__() + "+CrossEncoderReranking"