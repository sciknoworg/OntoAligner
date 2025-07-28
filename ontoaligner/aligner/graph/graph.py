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
from typing import Any, List, Dict
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from torch.nn.functional import normalize
import numpy as np
import torch

from ...base import BaseOMModel


class GraphEmbeddingAligner(BaseOMModel):
    """
    A model for aligning entities between two ontologies using knowledge graph embeddings.

    This class leverages PyKEEN to train a knowledge graph embedding model on input triples
    and aligns entities from a source ontology to a target ontology based on cosine similarity
    between their learned embeddings.

    Attributes:
        model (str): The name of the knowledge graph embedding model to use (must be set before training).
        graph_embedder (Any): The trained embedding model pipeline from PyKEEN.

    Example:
        >>> source_onto = {
        ...     "entity2iri": {"source1": "http://source.org/1", "source2": "http://source.org/2"},
        ...     "triplets": [("source1", "relatedTo", "source2")]
        ... }
        >>> target_onto = {
        ...     "entity2iri": {"target1": "http://target.org/1", "target2": "http://target.org/2"},
        ...     "triplets": [("target1", "relatedTo", "target2")]
        ... }
        >>> predictions = aligner.generate([source_onto, target_onto])
        >>> print(predictions)
        ... [{'source': 'http://source.org/1', 'target': 'http://target.org/2', 'score': 0.87}, ...]
    """

    model: str = ""
    graph_embedder: Any = None

    def __init__(self,
                 device: str='cpu',
                 embedding_dim: int=300,
                 num_epochs: int=50,
                 train_batch_size: int=128,
                 eval_batch_size: int=64,
                 num_negs_per_pos: int=5,
                 random_seed: int=42):
        """
        Initializes the GraphEmbeddingAligner with training configuration.

        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
            embedding_dim (int): Dimensionality of the entity embeddings.
            num_epochs (int): Number of training epochs.
            train_batch_size (int): Batch size for training.
            eval_batch_size (int): Batch size for evaluation.
            num_negs_per_pos (int): Number of negative samples per positive triple.
            random_seed (int): Random seed for reproducibility.
        """
        super().__init__(device=device,
                         embedding_dim=embedding_dim,
                         num_epochs=num_epochs,
                         train_batch_size=train_batch_size,
                         eval_batch_size=eval_batch_size,
                         num_negs_per_pos=num_negs_per_pos,
                         random_seed=random_seed)

    def fit(self, triplets: List):
        """
        Trains a knowledge graph embedding model on the given triples.

        Args:
            triplets (List): A list of triples, where each triple is a (head, relation, tail) tuple.
        """
        triples_array = np.array(triplets, dtype=str)
        train_triples, test_triples = triples_array, triples_array

        training_factory = TriplesFactory.from_labeled_triples(train_triples, create_inverse_triples=True)
        testing_factory = TriplesFactory.from_labeled_triples(test_triples,
                                                              entity_to_id=training_factory.entity_to_id,
                                                              relation_to_id=training_factory.relation_to_id,
                                                              create_inverse_triples=True)
        self.graph_embedder = pipeline(model=self.model,
                                      training=training_factory,
                                      testing=testing_factory,
                                      model_kwargs=dict(embedding_dim=self.kwargs['embedding_dim']),
                                      training_kwargs=dict(num_epochs=self.kwargs['num_epochs'],
                                                           batch_size=self.kwargs['train_batch_size']),
                                      negative_sampler_kwargs=dict(num_negs_per_pos=self.kwargs['num_negs_per_pos'],
                                                                   filtered=True),
                                      evaluation_kwargs=dict(batch_size=self.kwargs['eval_batch_size']),
                                      random_seed=self.kwargs['random_seed'],
                                      device=self.kwargs['device'])

    def _similarity_matrix(self, source_onto_tensor, target_onto_tensor):
        return torch.matmul(source_onto_tensor, target_onto_tensor.T)

    def predict(self, source_onto: Dict, target_onto: Dict):
        """
        Aligns entities from the source ontology to entities in the target ontology
        based on embedding similarity.

        Args:
            source_onto (Dict): Dictionary containing 'entity2iri' and 'triplets' for the source ontology.
            target_onto (Dict): Dictionary containing 'entity2iri' and 'triplets' for the target ontology.

        Returns:
            List[Dict]: A list of alignment mappings with source IRI, target IRI, and similarity score.
        """
        source_ent2iri, target_ent2iri = source_onto['entity2iri'], target_onto['entity2iri']

        embedding = self.graph_embedder.model.entity_representations[0](indices=None)
        embedding_entity2id = self.graph_embedder.training.entity_to_id
        source_ents, target_ents = list(source_ent2iri.keys()), list(target_ent2iri.keys())

        source_onto_tensor = torch.stack([embedding[embedding_entity2id[ent]] for ent in source_ents])
        target_onto_tensor = torch.stack([embedding[embedding_entity2id[ent]] for ent in target_ents])

        source_onto_tensor = normalize(source_onto_tensor, dim=1)  # shape: (n1, d)
        target_onto_tensor = normalize(target_onto_tensor, dim=1)  # shape: (n2, d)

        similarity_matrix = self._similarity_matrix(source_onto_tensor, target_onto_tensor) # shape: (n1, n2)

        best_scores, best_indices = similarity_matrix.max(dim=1)

        matches = [
            {
                "source": source_ent2iri[source_ents[index]],
                "target": target_ent2iri[target_ents[best_indices[index].item()]],
                "score": best_scores[index].item()
            }
            for index in range(len(source_ents))
        ]
        return matches

    def get_embeddings(self):
        """
        Returns the internal PyKEEN pipeline object containing the trained embedding model.

        Returns:
            Any: The PyKEEN pipeline object.
        """
        return self.graph_embedder

    def encode(self, input: str) -> np.ndarray:
        """
        Retrieves the embedding vector for a given entity.

        Args:
            input (str): The entity ID or label as used in training.

        Returns:
            np.ndarray: The embedding vector for the input entity.

        Raises:
            Exception: If the entity is not found in the trained model.
        """
        try:
            embedding = self.graph_embedder.model.entity_representations[0](indices=None)
            embedding_entity2id = self.graph_embedder.training.entity_to_id
            return embedding[embedding_entity2id[input]]
        except Exception as error:
            raise error

    def generate(self, input_data: List) -> List:
        """
        Full pipeline to train on combined triplets and predict alignments.

        Args:
            input_data (List): A list with two elements, each a dictionary representing a source and target ontology.

        Returns:
            List[Dict]: A list of predicted alignments with source IRI, target IRI, and similarity score.
        """
        source_onto, target_onto = input_data
        triplets = source_onto['triplets'] + target_onto['triplets']
        self.fit(triplets)
        predicts = self.predict(source_onto=source_onto, target_onto=target_onto)
        return predicts


    def __str__(self):
        """
        Returns a string representation of the model. This method must be implemented by subclasses.
        """
        return str(self.graph_embedder)
