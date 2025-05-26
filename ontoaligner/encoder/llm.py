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
from typing import Any, Dict

from .encoders import LLMEncoder

class ConceptLLMEncoder(LLMEncoder):
    """
    Encodes OWL items that represent concepts.

    This class inherits from the `LightweightEncoder` class and is designed to encode OWL items that consist of
    concepts. The `get_owl_items` method retrieves the IRI and label of the concept.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, a Concept.
    """
    items_in_owl: str = """(Concept)"""

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Extracts the IRI and label of a concept from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL item, expected to contain 'iri' and 'label' keys.

        Returns:
            Dict: A dictionary containing the IRI and label of the concept.
        """
        return {"iri": owl["iri"], "concept": owl["label"]}


class ConceptChildrenLLMEncoder(LLMEncoder):
    """
    Encodes OWL items that represent concepts and their children.

    This class inherits from the `LightweightEncoder` class and is designed to encode OWL items that consist of
    concepts and their children. The `get_owl_items` method retrieves the IRI, label of the concept, and the labels of its children.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, a Concept with Children.
    """
    items_in_owl: str = "(Concept, Children)"

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Extracts the IRI and label of a concept, along with the labels of its children, from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL item, expected to contain 'iri', 'label',
                        and 'childrens' keys where 'childrens' is a list of children with 'label' attributes.

        Returns:
            Dict: A dictionary containing the IRI, label of the concept, and the concatenated labels of its children.
        """
        childrens = ", ".join([children["label"] for children in owl["childrens"]])
        return {"iri": owl["iri"], "concept": owl["label"], "childrens": str(childrens)}


class ConceptParentLLMEncoder(LLMEncoder):
    """
    Encodes OWL items that represent concepts and their parents.

    This class inherits from the `LightweightEncoder` class and is designed to encode OWL items that consist of
    concepts and their parents. The `get_owl_items` method retrieves the IRI, label of the concept, and the labels of its parents.

    Attributes:
        items_in_owl (str): Specifies the type of OWL items being encoded, in this case, a Concept with Parent.
    """
    items_in_owl: str = "(Concept, Parent)"

    def get_owl_items(self, owl: Dict) -> Any:
        """
        Extracts the IRI and label of a concept, along with the labels of its parents, from the given OWL item.

        Parameters:
            owl (Dict): A dictionary representing an OWL item, expected to contain 'iri', 'label',
                        and 'parents' keys where 'parents' is a list of parents with 'label' attributes.

        Returns:
            Dict: A dictionary containing the IRI, label of the concept, and the concatenated labels of its parents.
        """
        parents = ", ".join([parent["label"] for parent in owl["parents"]])
        return {"iri": owl["iri"], "concept": owl["label"], "parents": str(parents)}
