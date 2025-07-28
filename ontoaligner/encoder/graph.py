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
from typing import Any
from ..base import BaseEncoder

class GraphTripleEncoder(BaseEncoder):

    def encode_ontology(self, ontology):
        ontology_dict = {'triplets': [], 'entity2iri': {}}
        for item in ontology:
            sub_iri, sub = item['subject']
            pred_iri, pred = item['predicate']
            obj_iri, obj = item['object']
            sub = self.preprocess(sub)
            pred = self.preprocess(pred)
            obj = self.preprocess(obj)
            ontology_dict['triplets'].append((sub, pred, obj))
            if item['subject_is_class']:
                ontology_dict['entity2iri'][sub] = sub_iri
            if item['object_is_class']:
                ontology_dict['entity2iri'][obj] = obj_iri
        return ontology_dict

    def parse(self, **kwargs) -> Any:
        """
        Parses the source and target ontologies, applying preprocessing.

        This method extracts ontology items (IRI and label) from the source and target ontologies,
        applies text preprocessing to the labels, and returns the encoded data.

        Parameters:
            **kwargs: Contains the source and target ontologies as keyword arguments.

        Returns:
            list: A list containing two elements, the processed source and target ontologies.
        """
        source_onto, target_onto = kwargs["source"], kwargs["target"]
        source_ontos = self.encode_ontology(source_onto)
        target_ontos = self.encode_ontology(target_onto)
        return [source_ontos, target_ontos]

    def __str__(self):
        """
        Returns a string representation of the encoder.
        """
        return {"GraphTripleEncoder": "(Subject, Object, Predicate) representation for Graph Learning"}

    def get_encoder_info(self):
        """
        Provides information about the encoder.
        """
        return "INPUT CONSIST OF GRAPH TRIPLETS FOR TRAINING GRAPH‌ EMBEDDINGS. THE ENCOER, PREPROCESS, RESTRUCTURE‌ THE INPUTS."
