# -*- coding: utf-8 -*-
import re
from typing import Any, List

import ontospy

from ..base import BaseOntologyParser, OMDataset

track = "mse"


def split_string(input_str):
    # Define a regular expression pattern to capture the desired components
    pattern = r"([A-Z]+)(\d+)([A-Z][a-z]+)?([A-Z][a-z]+)?"
    # Use re.findall to find all matching patterns in the input string
    matches = re.findall(pattern, input_str)
    if matches:
        # The first element of each match is the whole match, so we need to slice from the second element
        result = [match[0:] for match in matches[0]]
        # Filter out empty strings from the result
        result = [component for component in result if component]
    else:
        result = re.findall("[A-Z][^A-Z]*", input_str)
    result = " ".join(result)
    return result


class EMMOOntology(BaseOntologyParser):
    def is_contain_label(self, owl_class: Any) -> bool:
        try:
            if str(owl_class) == "owl.Thing":
                return False
            if len(owl_class.prefLabel) == 0:
                return False
            return True
        except Exception as e:
            print(f"Exception: {e}")
            return False

    def get_comments(self, owl_class: Any) -> List:
        return owl_class.comment.en

    def get_label(self, owl_class: Any) -> str:
        return split_string(owl_class.prefLabel.en.first())

    def get_ancestors(self, owl_class: Any) -> List:
        return self.get_owl_items(list(owl_class.ancestors()))

    def get_synonyms(self, owl_class: Any) -> List:
        return []


class MaterialInformationOntoOntology(BaseOntologyParser):
    def is_contain_label(self, owl_class: Any) -> bool:
        return True

    def get_name(self, owl_class: Any) -> str:
        return str(owl_class.uri).split("#")[1]

    def get_label(self, owl_class: Any) -> str:
        preprocessed_str = (
            self.get_iri(owl_class).split("#")[1].replace("_", " ").replace("-", "")
        )
        return split_string(preprocessed_str)

    def get_iri(self, owl_class: Any) -> str:
        return str(owl_class.uri)

    def get_childrens(self, owl_class: Any) -> List:
        return self.get_owl_items(owl_class.children())

    def get_parents(self, owl_class: Any) -> List:
        return self.get_owl_items(owl_class.parents())

    def get_synonyms(self, owl_class: Any) -> List:
        return []

    def get_comments(self, owl_class: Any) -> List:
        return []

    def get_owl_classes(self, ontology: Any) -> Any:
        return ontology.all_classes

    def load_ontology(self, input_file_path: str) -> Any:
        ontology = ontospy.Ontospy(input_file_path, verbose=True)
        return ontology



class MatOntoOntology(BaseOntologyParser):
    def get_comments(self, owl_class: Any) -> List:
        return owl_class.comment.en

    def get_synonyms(self, owl_class: Any) -> List:
        return owl_class.synonym


class MaterialInformationEMMOOMDataset(OMDataset):
    track = track
    ontology_name = "MaterialInformation-EMMO"

    source_ontology = MaterialInformationOntoOntology()
    target_ontology = EMMOOntology()


class MaterialInformationMatOntoMDataset(OMDataset):
    track = track
    ontology_name = "MaterialInformation-MatOnto"

    source_ontology = MaterialInformationOntoOntology()
    target_ontology = MatOntoOntology()


class MaterialInformationMatOntoReducedMDataset(OMDataset):
    track = track
    ontology_name = "MaterialInformationReduced-MatOnto"

    source_ontology = MaterialInformationOntoOntology()
    target_ontology = MatOntoOntology()
