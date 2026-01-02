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
PropMatch Encoder for property-based ontology matching.

This module implements encoders that reformat parsed property data from ontology graphs
into structures suitable for the PropertyMatcher alignment algorithm.
"""
from typing import Any, Dict

from ..base import BaseEncoder


class PropertyEncoder(BaseEncoder):
    """
    Base encoder for property-based ontology matching.

    This encoder transforms parsed property data from RDF graphs into a format
    suitable for the PropertyMatcher alignment algorithm. It extracts property
    labels, domains, and ranges, and formats them for comparison.

    The encoder processes properties that have been parsed by PropMatchOntology
    and prepares them for matching based on label similarity.

    Attributes:
        items_in_owl (str): Description of OWL items being encoded
        prompt_template (str): Template describing the encoding task
    """
    items_in_owl: str = "(Property)"
    prompt_template: str = "Encode property with its label for matching"

    def parse(self, **kwargs) -> Any:
        """
        Parses the source and target ontology graphs and extracts property information.

        This method processes the RDF graphs loaded by the parser, extracts properties
        (entities with rdfs:domain and rdfs:range), and formats them into a structure
        suitable for the PropertyMatcher algorithm.

        Parameters:
            **kwargs: Contains 'source' and 'target' RDF graphs

        Returns:
            list: A list containing [source_properties, target_properties]
                  where each is a list of formatted property dictionaries

        """
        source_onto = kwargs["source"]
        target_onto = kwargs["target"]

        # Extract and encode source properties
        source_properties = []
        for property in source_onto:
            encoded_prop = self.get_owl_items(property=property)
            encoded_prop["text"] = self.preprocess(encoded_prop["text"])
            source_properties.append(encoded_prop)

        # Extract and encode target properties
        target_properties = []
        for property in target_onto:
            encoded_prop = self.get_owl_items(property=property)
            encoded_prop["text"] = self.preprocess(encoded_prop["text"])
            target_properties.append(encoded_prop)

        return [source_properties, target_properties]


    def get_owl_items(self, property) -> Dict:
        """
        Extracts the IRI and label of a property from the RDF graph.
        >>> result = encoder.get_owl_items(property=property)
        >>> print(result)
        >>> {'iri': 'http://example.org#hasAuthor', 'text': 'has author'}
        """
        return {"iri": property['iri'], "text": property['label']}

    def get_encoder_info(self) -> str:
        """
        Provides information about the encoder.

        Returns:
            str: A description of the encoder's function
        """
        return "Encodes properties (label only) for PropertyMatcher alignment"

    def __str__(self):
        """
        Returns a string representation of the encoder.

        Returns:
            dict: A dictionary with encoder name and items
        """
        return {"PropertyEncoder": self.items_in_owl}


class PropMatchEncoder(PropertyEncoder):
    """
    Encoder for properties that includes inverse property information.

    This encoder includes information about inverse properties when available,
    which can improve matching accuracy for bidirectional relationships.

    Attributes:
        items_in_owl (str): Description of OWL items (Property with Inverse)
    """

    items_in_owl: str = "(Property, Domain, Range, Inverse)"
    prompt_template: str = "Encode property with domain, range, and inverse for matching"

    def get_owl_items(self, property) -> Dict:
        """
        Extracts property information including domain, range, and inverse.

        Parameters:
            owl (URIRef): The property URI
            graph: The RDF graph containing the property

        Returns:
            Dict: Dictionary with 'iri' and 'text' (label + domain + range + inverse)
        """
        label = property['label']
        domain_text = " ".join(property['domain_text']) if len(property['domain_text']) > 0 else ""
        range_text = " ".join(property['range_text']) if len(property['range_text']) > 0 else ""
        inverse_text = ""
        if property['inverse_of']:
            inverse_text = " ".join(property['inverse_label']) if len(property['inverse_label']) > 0 else ""

        combined_text = label
        if domain_text:
            combined_text += "  " + domain_text
        if range_text:
            combined_text += "  " + range_text
        if inverse_text:
            combined_text += "  inverse: " + inverse_text
        return {**property, "text": combined_text}

    def get_encoder_info(self) -> str:
        """
        Provides information about the encoder.

        Returns:
            str: A description of the encoder's function
        """
        return "Encodes properties with label, domain, range, and inverse for PropertyMatcher alignment"

    def __str__(self):
        """Returns a string representation of the encoder."""
        return {"PropMatchEncoder": self.items_in_owl}
