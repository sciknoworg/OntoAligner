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
from typing import Any, List, Dict, Tuple
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from py_stringmatching import SoftTfIdf, JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

from ...base import BaseOMModel
from .embedding import WordEmbedding


class PropMatchAligner(BaseOMModel):
    wordembedding_model: Any = None
    sentence_transformer_model: Any = None

    def __init__(self,
                 fmt: str = "word2vec",
                 lowercase: bool = False,
                 threshold: float = 0.65,
                 steps: int = 2,
                 sim_weight: List[int] = None,
                 start_metrics: List[float] = None,
                 device: str = "cpu",
                 disable_domain_range: bool = False) -> None:
        """
        Initialize the PropMatchAligner.

        Args:
            fmt: Format for word embedding (e.g., "word2vec")
            lowercase: Whether to lowercase text
            threshold: Minimum similarity threshold for matches
            steps: Number of iterative refinement steps
            sim_weight: Which similarity components to use [0:domain, 1:label, 2:range]
            start_metrics: Additional threshold metrics for evaluation
            device: Device for computation ("cpu" or "cuda")
            disable_domain_range: If True, only uses label similarity
        """
        super().__init__(fmt=fmt,
                         lowercase=lowercase,
                         threshold=threshold,
                         steps=steps,
                         sim_weight=sim_weight,
                         start_metrics=start_metrics,
                         device=device,
                         disable_domain_range=disable_domain_range)

    def load(self, wordembedding_path: str, sentence_transformer_id: str) -> None:
        """
        Loads the pre-trained models for word-embedding and sentence transformer.

        Args:
            wordembedding_path (str): The path to the pre-trained word-embedding.
            sentence_transformer_id (str): The path to the pre-trained sentence transformer.
        """
        self.wordembedding_model = WordEmbedding(path=wordembedding_path,
                                                 fmt=self.kwargs['fmt'],
                                                 lowercase=self.kwargs['lowercase'],
                                                 device=self.kwargs['device'])
        self.sentence_transformer_model = SentenceTransformer(sentence_transformer_id,
                                                              device=self.kwargs['device'],
                                                              trust_remote_code=True)

    def __str__(self):
        return "PropMatchAligner"

    def build_tf_models(self, source_onto: List[Dict], target_onto: List[Dict]) -> Tuple:
        """
        Build the TF-IDF models for soft TF-IDF and general TF-IDF.

        Args:
            source_onto: List of source property dictionaries
            target_onto: List of target property dictionaries

        Returns:
            Tuple of (soft_metric, general_metric) models
        """
        # Build soft TF-IDF using property labels (tokenized)
        sentences_list = []
        for prop in source_onto:
            sentences_list.append(prop['label'].lower().split())
        for prop in target_onto:
            sentences_list.append(prop['label'].lower().split())

        soft_metric = SoftTfIdf(sentences_list,
                                sim_func=JaroWinkler().get_raw_score,
                                threshold=0.8) ################ 0.8

        # Build general TF-IDF using full property text (label + domain + range)
        document_list = [prop['text'] for prop in source_onto] + [prop['text'] for prop in target_onto]
        general_metric = TfidfVectorizer()
        general_metric.fit(document_list)

        return soft_metric, general_metric

    def get_core_concept(self, entity: List[str]) -> List[str]:
        """
        Get the core concept of an entity. The core concept is the first verb with length > 4
        or the first noun with its adjectives.

        Args:
            entity: List of words from property label

        Returns:
            List of core concept words
        """
        try:
            tags = nltk.pos_tag(entity)
        except (TypeError, LookupError) as er:
            raise ValueError(f"Install the `averaged_perceptron_tagger` to proceed! \n"
                             f"import nltk\n"
                             f"nltk.download('averaged_perceptron_tagger')\n"
                             f"nltk.download('averaged_perceptron_tagger_eng')\n"
                             f"\n\n\n More Information: \n{er}")

        core_concept = []
        no_name = False
        for (word, tag) in tags:
            if 'V' in tag and len(word) > 4:
                core_concept.append(word)
                break
            if ('N' in tag or 'J' in tag) and not no_name:
                if 'IN' in tag:
                    no_name = True
                else:
                    core_concept.append(word)

        return core_concept

    def filter_adjectives(self, words: List[str]) -> List[str]:
        """
        Filter adjectives from a list of words, keeping only nouns.

        Args:
            words: List of words

        Returns:
            List of words without adjectives (only nouns)
        """
        if not words:
            return []

        try:
            tags = nltk.pos_tag(words)
        except (TypeError, LookupError) as er:
            raise ValueError(f"Install the `averaged_perceptron_tagger` to proceed! \n"
                             f"import nltk\n"
                             f"nltk.download('averaged_perceptron_tagger')\n"
                             f"nltk.download('averaged_perceptron_tagger_eng')\n"
                             f"\n\n\n More Information: \n{er}")

        return list(map(lambda word: word[0], filter(lambda word: word[1][0] == 'N', tags)))

    def cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute the cosine similarity between two vectors.

        Args:
            vector1: First vector
            vector2: Second vector

        Returns:
            Cosine similarity score
        """
        if np.linalg.norm(vector1) * np.linalg.norm(vector2) == 0:
            return 0
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def get_document_similarity(self,
                                label_a_items: List[str],
                                label_b_items: List[str],
                                general_metric_model) -> Tuple[float, float]:
        """
        Compute the document similarity between two property descriptions.

        Args:
            label_a_items: List of words from property A
            label_b_items: List of words from property B
            general_metric_model: TF-IDF vectorizer model

        Returns:
            Tuple of (conf_a, conf_b) similarity scores
        """
        if len(label_a_items) <= 0 or len(label_b_items) <= 0:
            conf_a = 0
            conf_b = 0
        else:
            # A -> B
            vector = general_metric_model.transform([' '.join(label_a_items), ' '.join(label_b_items)])
            vector = vector.toarray()
            conf_a = self.cosine_similarity(vector[0], vector[1])

            # B -> A
            vector = general_metric_model.transform([' '.join(label_b_items), ' '.join(label_a_items)])
            vector = vector.toarray()
            conf_b = self.cosine_similarity(vector[0], vector[1])

        return conf_a, conf_b

    def match_property(self, source: Dict, target: Dict,
                       soft_metric_model, general_metric_model,
                       confidence_map: Dict) -> float:
        """
        Match two properties by comparing their labels, domains, and ranges.

        Args:
            source: Source property dictionary
            target: Target property dictionary
            soft_metric_model: Soft TF-IDF model for label matching
            general_metric_model: TF-IDF model for domain/range matching
            confidence_map: Map of previously aligned classes for confidence boosting

        Returns:
            Similarity confidence score
        """
        # Extract and process source property
        domain_a = source['domain_text']
        range_a = source['range_text']
        exact_label_a = source['label'].lower().split()

        # Remove range from label if it's the last word
        if len(range_a) == 1 and len(exact_label_a) > 0 and exact_label_a[-1] == range_a[0]:
            exact_label_a.pop(-1)

        string_a = self.get_core_concept(exact_label_a)
        range_a = self.filter_adjectives(range_a)

        # Extract and process target property
        domain_b = target['domain_text']
        range_b = target['range_text']
        exact_label_b = target['label'].lower().split()

        # Remove range from label if it's the last word
        if len(range_b) == 1 and len(exact_label_b) > 0 and exact_label_b[-1] == range_b[0]:
            exact_label_b.pop(-1)

        string_b = self.get_core_concept(exact_label_b)
        range_b = self.filter_adjectives(range_b)

        # Calculate label similarity
        if exact_label_a == exact_label_b:
            label_conf_a = 1
            label_conf_b = 1
        elif len(string_a) <= 0 or len(string_b) <= 0:
            label_conf_a = 0
            label_conf_b = 0
        else:
            label_conf_a = soft_metric_model.get_raw_score(string_a, string_b)
            label_conf_b = soft_metric_model.get_raw_score(string_b, string_a)

        # Calculate domain and range similarity
        domain_conf_a, domain_conf_b = self.get_document_similarity(domain_a, domain_b, general_metric_model)
        range_conf_a, range_conf_b = self.get_document_similarity(range_a, range_b, general_metric_model)

        label_confidence = (label_conf_a + label_conf_b) / 2
        domain_confidence = (domain_conf_a + domain_conf_b) / 2
        range_confidence = (range_conf_a + range_conf_b) / 2

        # Use word embedding for single-word domains when TF-IDF returns 0
        if domain_confidence == 0 and len(exact_label_a) > 0 and len(exact_label_b) > 0:
            if len(domain_a) == 1 and len(domain_b) == 1:
                domain_confidence = self.wordembedding_model.sim(domain_a[0], domain_b[0])

        # Boost confidence for previously aligned classes
        domain_key = (source.get('domain', [''])[0], target.get('domain', [''])[0])
        if domain_key in confidence_map:
            domain_confidence += confidence_map[domain_key]

        range_key = (source.get('range', [''])[0], target.get('range', [''])[0])
        if range_key in confidence_map:
            range_confidence += confidence_map[range_key]

        # Handle disable_domain_range option
        disable_dr = self.kwargs.get('disable_domain_range', False)
        sim_weights = self.kwargs.get('sim_weight')

        if disable_dr:
            domain_confidence = 0
            range_confidence = 0
            sim_weights = [1]  # Only use label

        # Fallback to sentence transformer when domain/range are very similar but label is not
        if domain_confidence > 0.95 and range_confidence > 0.95 and label_confidence < 0.1:
            if len(string_a) <= 1 and len(string_b) <= 1:
                # Encode full context: domain + label + range
                text_a = ' '.join(domain_a + exact_label_a + range_a)
                text_b = ' '.join(domain_b + exact_label_b + range_b)

                embedding_a = self.sentence_transformer_model.encode([text_a], convert_to_tensor=True)
                embedding_b = self.sentence_transformer_model.encode([text_b], convert_to_tensor=True)

                sim = nn.functional.cosine_similarity(embedding_a, embedding_b).item()
                if sim < 0.8:
                    sim = 0
                label_confidence = sim

        # Aggregate confidences based on similarity weights
        if sim_weights:
            conf = []
            if 0 in sim_weights:
                conf.append(domain_confidence)
            if 1 in sim_weights:
                conf.append(label_confidence)
            if 2 in sim_weights:
                conf.append(range_confidence)
        else:
            conf = [label_confidence, domain_confidence, range_confidence]
        # print(conf)
        # Return minimum confidence (conservative matching)
        # return min(conf)
        return sum(conf)/len(conf)

    def generate(self, input_data: List[Dict]) -> List:
        """
        Generate alignments between source and target ontology properties.

        Args:
            source: List of source property dictionaries from encoder
            target: List of target property dictionaries from encoder

        Returns:
            List of alignment dictionaries with 'source', 'target', and 'score'
        """
        # Build TF-IDF models
        source, target = input_data
        soft_metric_model, general_metric_model = self.build_tf_models(source, target)

        # Initialize alignment structures
        final_alignment = {}
        confidence_map = {}
        property_map = {}

        # Iterative refinement
        for step in range(self.kwargs['steps']):
            for source_property in tqdm(source):
                source_iri = source_property['iri']

                for target_property in target:
                    target_iri = target_property['iri']

                    # Calculate similarity
                    similarity = self.match_property(
                        source=source_property,
                        target=target_property,
                        soft_metric_model=soft_metric_model,
                        general_metric_model=general_metric_model,
                        confidence_map=confidence_map
                    )

                    # Skip if below threshold
                    if similarity <= self.kwargs['threshold']:
                        continue

                    # Handle 1-to-1 mapping constraint for source property
                    if source_iri in property_map:
                        if property_map[source_iri][1] >= similarity:
                            continue
                        elif property_map[source_iri][1] < similarity:
                            # Remove old alignment
                            old_target = property_map[source_iri][0]
                            final_alignment.pop((source_iri, old_target), None)
                            property_map.pop(old_target, None)
                            property_map.pop(source_iri, None)

                    # Handle 1-to-1 mapping constraint for target property
                    if target_iri in property_map:
                        if property_map[target_iri][1] >= similarity:
                            continue
                        elif property_map[target_iri][1] < similarity:
                            # Remove old alignment
                            old_source = property_map[target_iri][0]
                            final_alignment.pop((old_source, target_iri), None)
                            property_map.pop(old_source, None)
                            property_map.pop(target_iri, None)

                    # Add new alignment
                    final_alignment[(source_iri, target_iri)] = similarity
                    property_map[source_iri] = (target_iri, similarity)
                    property_map[target_iri] = (source_iri, similarity)

                    # Update confidence map for domain/range classes
                    if source_property.get('domain') and target_property.get('domain'):
                        domain_pair = (source_property['domain'][0], target_property['domain'][0])
                        confidence_map[domain_pair] = 0.66

                    # Handle inverse properties
                    if source_property.get('inverse_of') and target_property.get('inverse_of'):
                        inv_source = source_property['inverse_of']
                        inv_target = target_property['inverse_of']

                        # Add inverse alignment with same confidence
                        final_alignment[(inv_source, inv_target)] = similarity
                        property_map[inv_source] = (inv_target, similarity)
                        property_map[inv_target] = (inv_source, similarity)

                        # Update confidence map for inverse domain/range
                        if source_property.get('range') and target_property.get('range'):
                            inv_domain_pair = (source_property['range'][0], target_property['range'][0])
                            confidence_map[inv_domain_pair] = 0.66

        # Convert to result format
        results = [
            {
                "source": source_iri,
                "target": target_iri,
                "score": similarity
            }
            for (source_iri, target_iri), similarity in final_alignment.items()
        ]

        return results
