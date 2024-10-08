# -*- coding: utf-8 -*-
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from owlready2 import World
from rdflib import Namespace, URIRef
from tqdm import tqdm


class BaseOntologyParser(ABC):
    def is_contain_label(self, owl_class: Any) -> bool:
        try:
            if len(owl_class.label) == 0:
                return False
            return True
        except Exception as e:
            print(f"Exception: {e}")
            return False

    def get_name(self, owl_class: Any) -> str:
        return owl_class.name

    def get_label(self, owl_class: Any) -> str:
        return owl_class.label.first()

    def get_iri(self, owl_class: Any) -> str:
        return owl_class.iri

    def get_childrens(self, owl_class: Any) -> List:
        return self.get_owl_items(owl_class.subclasses())  # include_self = False

    def get_parents(self, owl_class: Any) -> List:
        ans = self.get_owl_items(owl_class.is_a)  # include_self = False, ancestors()
        return ans

    def get_synonyms(self, owl_class: Any) -> List:
        return self.get_owl_items(owl_class.hasRelatedSynonym)

    @abstractmethod
    def get_comments(self, owl_class: Any) -> List:
        pass

    def get_owl_items(self, owl_class: Any) -> List:
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
        return ontology.classes()

    def duplicate_removals(self, owl_class_info: Dict) -> Dict:
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
        ontology = World()
        ontology.get_ontology(input_file_path).load()
        return ontology

    def parse(self, root_dir: str, ontology_file_name: str) -> List:
        input_file_path = os.path.join(root_dir, ontology_file_name)
        print(f"\t\tworking on {input_file_path}")
        ontology = self.load_ontology(input_file_path=input_file_path)
        return self.extract_data(ontology)


class BaseAlignmentsParser(ABC):
    namespace: Namespace = Namespace(
        "http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
    )
    entity_1: URIRef = URIRef(namespace + "entity1")
    entity_2: URIRef = URIRef(namespace + "entity2")
    relation: URIRef = URIRef(namespace + "relation")

    def extract_data(self, reference: Any) -> List[Dict]:
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
        ontology = World()
        ontology.get_ontology(input_file_path).load()
        return ontology

    def parse(self, root_dir: str, reference_file_name: str) -> List:
        input_file_path = os.path.join(root_dir, reference_file_name)
        print(f"\t\tworking on reference: {input_file_path}")
        reference = self.load_ontology(input_file_path=input_file_path)
        return self.extract_data(reference)
