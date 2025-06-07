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
This script provides functionality for parsing ontologies and alignment files.
It includes methods for extracting data from OWL ontologies, such as names, labels, and relationships,
as well as parsing alignment data in RDF format to extract relationships between entities and their corresponding data.

Classes:
    - BaseOntologyParser: A base class for parsing OWL ontologies, extracting information such as
                          names, labels, parents, children, synonyms, and comments.
    - BaseAlignmentsParser: A base class for parsing alignment data, extracting relationships between
                            entities and their corresponding RDF data.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


from owlready2 import World
from rdflib import Namespace, URIRef
from tqdm import tqdm

class BaseOntologyParser(ABC):
    """
    An abstract base class for parsing OWL ontologies. This class defines methods to extract data such as
    names, labels, IRIs, children, parents, synonyms, and comments for ontology classes.
    """

    def is_contain_label(self, owl_class: Any) -> bool:
        """
        Checks if the given ontology class contains a label.

        Parameters:
            owl_class (Any): An ontology class.

        Returns:
            bool: True if the class contains a label, False otherwise.
        """
        try:
            if len(owl_class.label) == 0:
                return False
            return True
        except Exception:
            return False

    def get_name(self, owl_class: Any) -> str:
        """
        Retrieves the name of the given ontology class.

        Parameters:
            owl_class (Any): An ontology class.

        Returns:
            str: The name of the ontology class.
        """
        return owl_class.name

    def get_label(self, owl_class: Any) -> str:
        """
        Retrieves the label of the given ontology class.

        Parameters:
            owl_class (Any): An ontology class.

        Returns:
            str: The label of the ontology class.
        """
        return owl_class.label.first()

    def get_iri(self, owl_class: Any) -> str:
        """
        Retrieves the IRI of the given ontology class.

        Parameters:
            owl_class (Any): An ontology class.

        Returns:
            str: The IRI of the ontology class.
        """
        return owl_class.iri

    def get_childrens(self, owl_class: Any) -> List:
        """
        Retrieves the subclasses (children) of the given ontology class.

        Parameters:
            owl_class (Any): An ontology class.

        Returns:
            List: A list of subclasses (children) of the ontology class.
        """
        return self.get_owl_items(owl_class.subclasses())  # include_self = False

    def get_parents(self, owl_class: Any) -> List:
        """
        Retrieves the superclasses (parents) of the given ontology class.

        Parameters:
            owl_class (Any): An ontology class.

        Returns:
            List: A list of superclasses (parents) of the ontology class.
        """
        ans = self.get_owl_items(owl_class.is_a)  # include_self = False, ancestors()
        return ans

    def get_synonyms(self, owl_class: Any) -> List:
        """
        Retrieves the synonyms of the given ontology class.

        Parameters:
            owl_class (Any): An ontology class.

        Returns:
            List: A list of synonyms of the ontology class.
        """
        return self.get_owl_items(owl_class.hasRelatedSynonym)

    @abstractmethod
    def get_comments(self, owl_class: Any) -> List:
        """
        Abstract method to retrieve comments for the given ontology class.

        Parameters:
            owl_class (Any): An ontology class.

        Returns:
            List: A list of comments associated with the ontology class.
        """
        pass

    def get_owl_items(self, owl_class: Any) -> List:
        """
        Extracts relevant items from the given ontology class, including IRI, name, and label.

        Parameters:
            owl_class (Any): An ontology class.

        Returns:
            List: A list of dictionaries containing IRI, name, and label of relevant ontology items.
        """
        owl_items = []
        for item in owl_class:
            if self.is_contain_label(item):
                owl_items.append(
                    {
                        "iri": self.get_iri(item),
                        "name": self.get_name(item),
                        "label": self.get_label(item),
                    }
                )
        return owl_items

    def get_owl_classes(self, ontology: Any) -> Any:
        """
        Retrieves all classes from the given ontology.

        Parameters:
            ontology (Any): An ontology.

        Returns:
            Any: A collection of all classes in the ontology.
        """
        return ontology.classes()

    def duplicate_removals(self, owl_class_info: Dict) -> Dict:
        """
        Removes duplicate ontology class information based on IRI.

        Parameters:
            owl_class_info (Dict): A dictionary containing information about an ontology class.

        Returns:
            Dict: A dictionary with duplicates removed from the class information.
        """
        def ignore_duplicates(iri: str, duplicated_list: List[Dict]) -> List:
            new_list = []
            for item in duplicated_list:
                if iri != item["iri"]:
                    new_list.append(item)
            return new_list

        new_owl_class_info = {
            "name": owl_class_info["name"],
            "iri": owl_class_info["iri"],
            "label": owl_class_info["label"],
            "childrens": ignore_duplicates(
                iri=owl_class_info["iri"], duplicated_list=owl_class_info["childrens"]
            ),
            "parents": ignore_duplicates(
                iri=owl_class_info["iri"], duplicated_list=owl_class_info["parents"]
            ),
            "synonyms": owl_class_info["synonyms"],
            "comment": owl_class_info["comment"],
        }
        return new_owl_class_info

    def extract_data(self, ontology: Any) -> List[Dict]:
        """
        Extracts and processes data from the given ontology, including children, parents, synonyms, and comments.

        Parameters:
            ontology (Any): An ontology.

        Returns:
            List: A list of dictionaries containing extracted ontology class data.
        """
        parsed_ontology = []
        for owl_class in tqdm(self.get_owl_classes(ontology)):
            if not self.is_contain_label(owl_class):
                continue
            owl_class_info = {
                "name": self.get_name(owl_class),
                "iri": self.get_iri(owl_class),
                "label": self.get_label(owl_class),
                "childrens": self.get_childrens(owl_class),
                "parents": self.get_parents(owl_class),
                "synonyms": self.get_synonyms(owl_class),
                "comment": self.get_comments(owl_class),
            }
            owl_class_info = self.duplicate_removals(owl_class_info=owl_class_info)
            parsed_ontology.append(owl_class_info)
        return parsed_ontology

    def load_ontology(self, input_file_path: str) -> Any:
        """
        Loads an ontology from the specified file path.

        Parameters:
            input_file_path (str): The file path of the ontology.

        Returns:
            Any: The loaded ontology.
        """
        ontology = World()
        ontology.get_ontology(input_file_path).load()
        return ontology

    def parse(self, input_file_path: str) -> List[Dict[str, Any]]:
        """
        Loads and processes the ontology, extracting relevant data.

        Parameters:
            input_file_path (str): The file path of the ontology.

        Returns:
            List: A list of extracted ontology data.
        """
        ontology = self.load_ontology(input_file_path=input_file_path)
        return self.extract_data(ontology)


class BaseAlignmentsParser(ABC):
    """
    An abstract base class for parsing RDF alignment data. This class provides methods for extracting
    relationships between entities in the alignment data, including entities and their relations.
    """

    namespace: Namespace = Namespace(
        "http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
    )
    entity_1: URIRef = URIRef(namespace + "entity1")
    entity_2: URIRef = URIRef(namespace + "entity2")
    relation: URIRef = URIRef(namespace + "relation")

    def extract_data(self, reference: Any) -> List[Dict]:
        """
        Extracts alignment data from an RDF graph, processing relationships between entities.

        Parameters:
            reference (Any): RDF reference containing the alignment data.

        Returns:
            List: A list of dictionaries representing entity relationships in the alignment data.
        """
        parsed_references = []
        graph = reference.as_rdflib_graph()
        for source, predicate, target in tqdm(graph):
            if predicate == self.relation:
                entity_1 = [
                    str(o) for s, p, o in graph.triples((source, self.entity_1, None))
                ][0]
                entity_2 = [
                    str(o) for s, p, o in graph.triples((source, self.entity_2, None))
                ][0]
                parsed_references.append(
                    {"source": entity_1, "target": entity_2, "relation": str(target)}
                )
        return parsed_references

    def load_ontology(self, input_file_path: str) -> Any:
        """
        Loads an RDF alignment file from the specified file path.

        Parameters:
            input_file_path (str): The file path of the RDF alignment file.

        Returns:
            Any: The loaded RDF alignment data.
        """
        ontology = World()
        ontology.get_ontology(input_file_path).load()
        return ontology

    def parse(self, input_file_path: str="") -> List:
        """
        Loads and processes the RDF alignment file, extracting relevant data.

        Parameters:
            input_file_path (str): The file path of the RDF alignment file.

        Returns:
            List: A list of extracted alignment data.
        """
        try:
            reference = self.load_ontology(input_file_path=input_file_path)
            return self.extract_data(reference)
        except Exception:
            return []
