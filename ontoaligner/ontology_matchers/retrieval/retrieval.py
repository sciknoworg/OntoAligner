# -*- coding: utf-8 -*-
from typing import Any, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ontoaligner.base import BaseOMModel


class Retrieval(BaseOMModel):
    path: str = ""
    model: Any = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load()

    def load(self):
        pass

    def __str__(self):
        return "Retrieval"

    def fit(self, inputs: Any) -> Any:
        pass

    def transform(self, inputs: Any) -> Any:
        pass

    def estimate_similarity(self, query_embed: Any, candidate_embeds: Any) -> Any:
        pass

    def get_top_k(self, query_embed: Any, candidate_embeds: Any) -> [List, List]:
        results = self.estimate_similarity(query_embed=query_embed, candidate_embeds=candidate_embeds)
        values = [(score, index) for index, score in enumerate(results)]
        dtype = [("score", float), ("index", int)]
        results = np.array(values, dtype=dtype)
        top_k_items = np.sort(results, order="score")[-self.kwargs["top_k"] :][::-1]
        top_k_indexes, top_k_scores = [], []
        for top_k in top_k_items:
            top_k_scores.append(top_k[0])
            top_k_indexes.append(top_k[1])
        return top_k_indexes, top_k_scores

    def generate(self, input_data: List) -> List:
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
    path: str = ""

    def load(self):
        self.model = SentenceTransformer(self.path, device=self.kwargs["device"])

    def fit(self, inputs: Any) -> Any:
        return self.model.encode(inputs, show_progress_bar=True, batch_size=16)

    def transform(self, inputs: Any) -> Any:
        return self.model.encode(inputs, show_progress_bar=True, batch_size=16)

    def generate(self, input_data: List) -> List:
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
    This retriever is the slowest model.
    So it should be used for labels based retrieval
    """

    path: str = ""

    def load(self):
        self.model = SentenceTransformer(self.path, device=self.kwargs["device"])

    def fit(self, inputs: Any) -> Any:
        return self.model.encode(inputs, show_progress_bar=True, batch_size=16)

    def transform(self, inputs: Any) -> Any:
        return self.model.encode(inputs, show_progress_bar=True, batch_size=16)

    def generate(self, input_data: List) -> List:
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
