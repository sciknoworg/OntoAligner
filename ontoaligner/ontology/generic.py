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
from rdflib import Graph, URIRef, RDFS, SKOS, BNode
from rdflib.namespace import OWL, RDF
from tqdm import tqdm
from typing import Any, Dict, List

from ..base import BaseOntologyParser, OMDataset

track = "Generic"

class GenericOntology(BaseOntologyParser):
    """
    An abstract base class for parsing OWL ontologies. This class defines methods to extract data such as
    names, labels, IRIs, children, parents, synonyms, and comments for ontology classes.
    """

    def __init__(self, language: str = 'en'):
        self.language = language

    def is_valid_label(self, label: str) -> Any:
        invalids = ['root', 'thing']
        if label.lower() in invalids:
            return None
        return label

    def get_label(self, owl_class: str) -> Any:
        """
        Extracts the label for a given URI in the specified language from the RDF graph.
        If no valid label is found, returns None.

        :param uri: URI of the entity to retrieve the label for.
        :return: The label in the specified language, or None if no label found.
        """
        entity = URIRef(owl_class)
        labels = list(self.graph.objects(subject=entity, predicate=RDFS.label))
        for label in labels:
            if hasattr(label, 'language') and label.language == self.language:
                return self.is_valid_label(str(label))
        if labels:
            first_label = str(labels[0])
            if not first_label.startswith("http"):
                return self.is_valid_label(first_label)
        if "#" in owl_class:
            local_name = owl_class.split("#")[-1]
        elif "/" in owl_class:
            local_name = owl_class.split("/")[-1]
        else:
            local_name = owl_class
        label = self.is_valid_label(local_name)
        if not label:
            return None
        return label

    def load_ontology(self, input_file_path: str) -> Any:
        """
        Loads an ontology from the specified file path using rdflib.
        """
        graph = Graph()
        graph.parse(input_file_path, format="xml")
        return graph

    def is_class(self, owl_class: URIRef) -> bool:
        """
        Checks if the given entity is an actual class (not a Restriction or property).
        """
        return (self.graph.value(owl_class, RDFS.subClassOf) is not None)

    def get_synonyms(self, owl_class: URIRef) -> List[Dict[str, str]]:
        """
        Retrieves synonyms for the given ontology class.

        Tries different properties (like `rdfs:label`, `skos:altLabel`).
        """
        synonyms = []
        # Search for alternative labels (SKOS altLabel)
        for alt_label in self.graph.objects(owl_class, SKOS.altLabel):
            synonyms.append({
                "iri": str(owl_class),
                "label": str(alt_label),
                "name": str(alt_label),
            })
        return synonyms

    def get_name(self, owl_class: URIRef) -> str:
        """
        Retrieves the name (IRI) of the class.
        """
        return self.get_label(owl_class)
        # return str(label) if label else None

    def get_iri(self, owl_class: URIRef) -> str:
        """
        Returns the IRI of the class.
        """
        return str(owl_class)

    def get_comments(self, owl_class: URIRef) -> List[str]:
        """
        Retrieves comments for the given class.
        """
        comments = []
        for comment in self.graph.objects(owl_class, RDFS.comment):
            comments.append(str(comment))
        return comments

    def get_parents(self, owl_class: URIRef) -> List[Dict[str, str]]:
        """
        Retrieves parent classes for the given ontology class.
        """
        parents = []
        for parent in self.graph.objects(owl_class, RDFS.subClassOf):
            if self.is_class(parent):
                name = self.get_name(parent)
                label = self.get_label(parent)
                iri = self.get_iri(parent)
                if label and iri:
                    parents.append({"iri": iri, "label": label, "name": name})
        return parents

    def get_childrens(self, owl_class: URIRef) -> List[Dict[str, str]]:
        """
        Retrieves child classes for the given ontology class.
        """
        childrens = []
        for child in self.graph.subjects(RDFS.subClassOf, owl_class):
            if self.is_class(child):
                name = self.get_name(child)
                label = self.get_label(child)
                iri = self.get_iri(child)
                if label and iri:
                    childrens.append({"iri": iri, "label": label, "name": name})
        return childrens

    def get_class_info(self, owl_class: URIRef) -> Any:
        """
        Collects all relevant information for a given ontology class.
        """
        label = self.get_label(owl_class)
        name = self.get_name(owl_class)
        iri = self.get_iri(owl_class)

        if not label or not iri:
            return None

        class_info = {
            "name": name,
            "iri": iri,
            "label": label,
            "childrens": self.get_childrens(owl_class),
            "parents": self.get_parents(owl_class),
            "synonyms": self.get_synonyms(owl_class),
            "comment": self.get_comments(owl_class)
        }
        return class_info

    def extract_data(self, graph: Any) -> List[Dict[str, Any]]:
        """
        Extracts and processes data from all classes in the ontology.
        """
        self.graph = graph
        parsed_ontology = []

        # Iterate over all classes explicitly defined as owl:Class
        for owl_class in tqdm(self.graph.subjects(RDF.type, OWL.Class)):
            if isinstance(owl_class, BNode):  # Skip blank nodes
                continue

            class_info = self.get_class_info(owl_class)
            if class_info:
                parsed_ontology.append(class_info)
        return parsed_ontology


class GenericOMDataset(OMDataset):
    """
    A dataset class for working with the Source-Target ontology.
    """
    track = track
    ontology_name = "Source-Target"

    source_ontology = GenericOntology()
    target_ontology = GenericOntology()
