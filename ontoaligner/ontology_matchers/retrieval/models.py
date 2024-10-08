# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Any
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .retrieval import BiEncoderRetrieval, MLRetrieval, Retrieval
from ontoaligner.utils import io


class BERTRetrieval(BiEncoderRetrieval):
    path: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    def __str__(self):
        return super().__str__() + "+BERTRetrieval"


class SpecterBERTRetrieval(BiEncoderRetrieval):
    path: str = "allenai/specter_plus_plus"

    def __str__(self):
        return super().__str__() + "SpecterBERTRetrieval"


class FlanT5XLRetrieval(BiEncoderRetrieval):
    path: str = "google/flan-t5-xl"

    def __str__(self):
        return super().__str__() + "FlanT5XLRetrieval"


class FlanT5XXLRetrieval(BiEncoderRetrieval):
    path: str = "google/flan-t5-xxl"

    def __str__(self):
        return super().__str__() + "FlanT5XXLRetrieval"


class TFIDFRetrieval(Retrieval):
    path: str = "NO MODEL LOADING IN TFIDFRetrieval MODEL"

    def load(self):
        self.model = TfidfVectorizer()

    def fit(self, inputs: Any) -> Any:
        self.model.fit(inputs)
        return self.transform(inputs=inputs)

    def transform(self, inputs: Any) -> Any:
        return self.model.transform(inputs)

    def estimate_similarity(self, query_embed: Any, candidate_embeds: Any) -> Any:
        return cosine_similarity(query_embed, candidate_embeds).reshape((-1,))

    def __str__(self):
        return super().__str__() + "+TFIDFRetrieval"


class BM25Retrieval(Retrieval):
    """
    http://ethen8181.github.io/machine-learning/search/bm25_intro.html
    """
    path: str = "NO MODEL LOADING IN BM25Retrieval MODEL"

    def fit(self, inputs: Any) -> Any:
        tokenized_inputs = [input.split(" ") for input in inputs]
        self.model = BM25Okapi(tokenized_inputs)
        return None

    def transform(self, inputs: Any) -> Any:
        return [input.split(" ") for input in inputs]

    def estimate_similarity(self, query_embed: Any, candidate_embeds: Any) -> Any:
        docs_scores = self.model.get_scores(query_embed)
        return docs_scores

    def __str__(self):
        return super().__str__() + "+BM25Retrieval"


class SVMBERTRetrieval(MLRetrieval):
    path: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    def __str__(self):
        return super().__str__() + "+SVMBERTRetrieval"


class AdaRetrieval(BiEncoderRetrieval):
    path: str = os.path.join(
        Path(__file__).parents[3], "assets", "openai-ada-embedding"
    )

    def __str__(self):
        return super().__str__() + "+AdaRetrieval"

    def load(self):
        self.model = np.load(os.path.join(self.path, "openai_embeddings.npy"))
        self.labels2index = io.read_json(os.path.join(self.path, "labels2index.json"))

    def fit(self, inputs: Any) -> Any:
        return self.transform(inputs=inputs)

    def transform(self, inputs: Any) -> Any:
        embeddings = []
        for input_text in tqdm(inputs):
            index = self.labels2index.get(input_text, 0)
            if index == 0:
                print("NO match for the string:", input_text)
            embeddings.append(self.model[index])
        return np.array(embeddings)
