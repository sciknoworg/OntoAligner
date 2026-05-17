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
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ..retrieval import BiEncoderRetrieval

class OLaLaSBERTRetrieval(BiEncoderRetrieval):
    """
    A SentenceTransformers retrieval model for OLaLa candidate generation.
    """
    def __init__(
        self,
        device: str = "cpu",
        top_k: int = 5,
        both_directions: bool = True,
        topk_per_resource: bool = True,
        **kwargs,
    ) -> None:
        """
        Initializes the OLaLa SBERT retrieval model.

        Parameters:
            device (str): The device used by SentenceTransformers.
            top_k (int): The number of candidates to retrieve per resource.
            both_directions (bool): Whether to search in both ontology directions.
            topk_per_resource (bool): Whether top-k filtering is applied per resource.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            device=device,
            top_k=top_k,
            openai_key="None",
            both_directions=both_directions,
            topk_per_resource=topk_per_resource,
            **kwargs,
        )

    def load(self, path: str = "multi-qa-mpnet-base-dot-v1"):
        """
        Loads the SentenceTransformers model.

        Parameters:
            path (str): The model path or HuggingFace model name.

        Returns:
            None
        """
        super().load(path=path)

    def get_text_examples(self, ontology: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Creates multiple text examples per ontology resource.

        Parameters:
            ontology (List[Dict[str, Any]]): The encoded ontology items.

        Returns:
            List[Dict[str, str]]: The text examples.
        """
        examples = []

        for item in ontology:
            if not item.get("keep_for_sbert", True):
                continue

            for text in item.get("texts", []):
                text = str(text).strip()
                if not text:
                    continue

                examples.append({
                    "iri": item["iri"],
                    "text": text,
                })

        return examples

    def add_prediction(
        self,
        predictions: Dict[Tuple[str, str], float],
        source_iri: str,
        target_iri: str,
        score: float,
    ) -> None:
        """
        Adds or updates a predicted correspondence.

        Parameters:
            predictions (Dict[Tuple[str, str], float]): The prediction dictionary.
            source_iri (str): The source entity IRI.
            target_iri (str): The target entity IRI.
            score (float): The confidence score.

        Returns:
            None
        """
        pair = (source_iri, target_iri)
        current_score = predictions.get(pair)

        if current_score is None or score > current_score:
            predictions[pair] = score

    def search_direction(
        self,
        query_examples: List[Dict[str, str]],
        corpus_examples: List[Dict[str, str]],
        reverse: bool = False,
    ) -> Dict[Tuple[str, str], float]:
        """
        Searches one ontology direction and returns candidate correspondences.

        Parameters:
            query_examples (List[Dict[str, str]]): The query text examples.
            corpus_examples (List[Dict[str, str]]): The corpus text examples.
            reverse (bool): Whether the search direction is target-source.

        Returns:
            Dict[Tuple[str, str], float]: Candidate pairs and confidence scores.
        """
        predictions = {}

        if not query_examples or not corpus_examples:
            return predictions

        query_embeddings = self.transform(
            inputs=[example["text"] for example in query_examples]
        )
        corpus_embeddings = self.fit(
            inputs=[example["text"] for example in corpus_examples]
        )

        similarities = cosine_similarity(query_embeddings, corpus_embeddings)
        top_k = min(self.kwargs["top_k"], len(corpus_examples))

        for query_index, scores in tqdm(
            enumerate(similarities),
            total=len(similarities),
        ):
            query_iri = query_examples[query_index]["iri"]
            top_indexes = np.argsort(scores)[::-1][:top_k]

            for corpus_index in top_indexes:
                corpus_iri = corpus_examples[corpus_index]["iri"]
                score = float(scores[corpus_index])

                if reverse:
                    self.add_prediction(
                        predictions=predictions,
                        source_iri=corpus_iri,
                        target_iri=query_iri,
                        score=score,
                    )
                else:
                    self.add_prediction(
                        predictions=predictions,
                        source_iri=query_iri,
                        target_iri=corpus_iri,
                        score=score,
                    )

        return predictions

    def filter_topk_per_resource(
        self,
        predictions: Dict[Tuple[str, str], float],
    ) -> Dict[Tuple[str, str], float]:
        """
        Filters correspondences by top-k source and target resources.

        Parameters:
            predictions (Dict[Tuple[str, str], float]): Candidate pairs and scores.

        Returns:
            Dict[Tuple[str, str], float]: The filtered candidate pairs and scores.
        """
        source_groups = {}
        target_groups = {}

        for (source_iri, target_iri), score in predictions.items():
            source_groups.setdefault(source_iri, []).append(
                (source_iri, target_iri, score)
            )
            target_groups.setdefault(target_iri, []).append(
                (source_iri, target_iri, score)
            )

        final_predictions = {}

        for group in source_groups.values():
            selected = sorted(
                group,
                key=lambda item: item[2],
                reverse=True,
            )[: self.kwargs["top_k"]]

            for source_iri, target_iri, score in selected:
                self.add_prediction(
                    predictions=final_predictions,
                    source_iri=source_iri,
                    target_iri=target_iri,
                    score=score,
                )

        for group in target_groups.values():
            selected = sorted(
                group,
                key=lambda item: item[2],
                reverse=True,
            )[: self.kwargs["top_k"]]

            for source_iri, target_iri, score in selected:
                self.add_prediction(
                    predictions=final_predictions,
                    source_iri=source_iri,
                    target_iri=target_iri,
                    score=score,
                )

        return final_predictions

    def merge_predictions(
        self,
        predictions: Dict[Tuple[str, str], float],
    ) -> List[Dict[str, Any]]:
        """
        Converts pair-level predictions to OntoAligner retrieval output.

        Parameters:
            predictions (Dict[Tuple[str, str], float]): Candidate pairs and scores.

        Returns:
            List[Dict[str, Any]]: The grouped retrieval predictions.
        """
        grouped_predictions = {}

        for (source_iri, target_iri), score in predictions.items():
            grouped_predictions.setdefault(source_iri, []).append((target_iri, score))

        outputs = []
        for source_iri, candidates in grouped_predictions.items():
            candidates = sorted(
                candidates,
                key=lambda item: item[1],
                reverse=True,
            )

            outputs.append({
                "source": source_iri,
                "target-cands": [target_iri for target_iri, _ in candidates],
                "score-cands": [score for _, score in candidates],
            })

        return outputs

    def generate(self, input_data: List) -> List:
        """
        Generates OLaLa SBERT candidate correspondences.

        Parameters:
            input_data (List): The encoded source and target ontologies.

        Returns:
            List: The generated candidate correspondences.
        """
        source_ontology = input_data[0]
        target_ontology = input_data[1]
        if len(source_ontology) == 0 or len(target_ontology) == 0:
            raise ValueError("Source or target ontologies cannot be empty.")

        source_examples = self.get_text_examples(source_ontology)
        target_examples = self.get_text_examples(target_ontology)

        predictions = self.search_direction(
            query_examples=source_examples,
            corpus_examples=target_examples,
            reverse=False,
        )

        if self.kwargs.get("both_directions", True):
            reverse_predictions = self.search_direction(
                query_examples=target_examples,
                corpus_examples=source_examples,
                reverse=True,
            )

            for (source_iri, target_iri), score in reverse_predictions.items():
                self.add_prediction(
                    predictions=predictions,
                    source_iri=source_iri,
                    target_iri=target_iri,
                    score=score,
                )

        if self.kwargs.get("topk_per_resource", True):
            predictions = self.filter_topk_per_resource(predictions)

        return self.merge_predictions(predictions)

    def __str__(self):
        """
        Returns the string representation of the retrieval model.

        Returns:
            str: The string representation of the retrieval model.
        """
        return super().__str__() + "+OLaLaSBERTRetrieval"
