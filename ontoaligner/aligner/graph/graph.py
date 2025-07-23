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
from sklearn.model_selection import train_test_split
from torch.nn.functional import normalize
import numpy as np
import torch

from ...base import BaseOMModel


class GraphEmbeddingAligner(BaseOMModel):

    model: str = ""
    graph_embedder: Any = None

    def __init__(self,
                 device: str='cpu',
                 embedding_dim: int=300,
                 num_epochs: int=50,
                 train_batch_size: int=128,
                 eval_batch_size: int=64,
                 num_negs_per_pos: int=5,
                 random_seed: int=42,
                 test_split: float=0.1):
        super().__init__(device=device,
                         embedding_dim=embedding_dim,
                         num_epochs=num_epochs,
                         train_batch_size=train_batch_size,
                         eval_batch_size=eval_batch_size,
                         num_negs_per_pos=num_negs_per_pos,
                         random_seed=random_seed,
                         test_split=test_split)

    def fit(self, triplets: List):
        triples_array = np.array(triplets, dtype=str)

        # Split into training and testing
        train_triples, test_triples = train_test_split(triples_array,
                                                       test_size=self.kwargs['test_split'],
                                                       random_state=self.kwargs['random_seed'],
                                                       shuffle=True)

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


    def predict(self, source_onto: Dict, target_onto: Dict):
        source_ent2iri, target_ent2iri = source_onto['entity2iri'], target_onto['entity2iri']

        embedding = self.graph_embedder.model.entity_representations[0](indices=None)
        embedding_entity2id = self.graph_embedder.training.entity_to_id
        source_ents, target_ents = list(source_ent2iri.keys()), list(target_ent2iri.keys())

        source_onto_tensor = torch.stack([embedding[embedding_entity2id[ent]] for ent in source_ents])
        target_onto_tensor = torch.stack([embedding[embedding_entity2id[ent]] for ent in source_ents])

        source_onto_tensor = normalize(source_onto_tensor, dim=1)  # shape: (n1, d)
        target_onto_tensor = normalize(target_onto_tensor, dim=1)  # shape: (n2, d)

        similarity_matrix = torch.matmul(source_onto_tensor, target_onto_tensor.T)  # shape: (n1, n2)

        best_scores, best_indices = similarity_matrix.max(dim=1)

        matches = [
            {
                "source": source_ent2iri[source_ents[index]],
                "target": target_ents[target_ents[best_indices[index].item()]],
                "score": best_scores[index].item()
            }
            for index in range(len(source_ents))
        ]
        return matches
        #
        # topk_scores, topk_indices = similarity_matrix.topk(k=5, dim=1)
        #
        # # For each O1 entity, get top-5 aligned O2 entities
        # topk_matches = [
        #     (O1_entities[i], [(O2_entities[topk_indices[i][j]], topk_scores[i][j].item()) for j in range(5)])
        #     for i in range(len(O1_entities))
        # ]

    def generate(self, input_data: List) -> List:
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
