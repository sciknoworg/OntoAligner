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
This module provides functionality to generate XML alignment files compliant with the Alignment API.
It is useful for representing ontology matching results in a standardized XML format.
"""
from xml.etree.ElementTree import Element, SubElement, tostring
from typing import List, Dict, Any
import xml.dom.minidom

def xml_alignment_generator(matchings: List[Dict], return_rdf: bool = False, relation: str = "=", digits: int = -2) -> Any:
    """
    Generates an XML file representing ontology matching results in RDF format.

    Parameters:
        matchings (List[dict]): A list of dictionaries representing matching pairs, where each dictionary contains:
            - 'source' (str): URI of the source entity.
            - 'target' (str): URI of the target entity.
            - 'score' (float): Confidence score of the mapping.
        relation (str): The default relation to be used between source and target if not provided in the input (default is "=").
        digits (int): The number of decimal places to round the confidence score. A value of -2 rounds to two decimal places.

    Returns:
        Any: A prettified XML string representing the ontology matchings, or an RDF element if `return_rdf` is True.
    """
    # Create the root RDF element
    rdf = Element("rdf:RDF", {
        "xmlns": "http://knowledgeweb.semanticweb.org/heterogeneity/alignment",
        "xmlns:rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "xmlns:xsd": "http://www.w3.org/2001/XMLSchema#"
    })

    alignment = SubElement(rdf, "Alignment") # Create the Alignment element

    # Add metadata to the Alignment element
    SubElement(alignment, "xml").text = "yes"
    SubElement(alignment, "level").text = "0"
    SubElement(alignment, "type").text = "??"  # Replace "??" with the specific alignment type if known

    for matching in matchings:
        entity1, entity2 = matching['source'], matching['target']
        try:
            confidence = matching['score']
        except KeyError:
            confidence = None
        try:
            matching_relation = matching['relation']
        except KeyError:
            matching_relation = relation
        map_element = SubElement(alignment, "map")
        cell = SubElement(map_element, "Cell")
        SubElement(cell, "entity1", {"rdf:resource": entity1}) # Add the source entity
        SubElement(cell, "entity2", {"rdf:resource": entity2}) # Add the source entity
        SubElement(cell, "relation").text = matching_relation  # Add the relation
        if confidence is not None:
            formatted_confidence = str(confidence)[:-1 if digits < 0 else digits + 2]
            SubElement(cell, "measure", {"rdf:datatype": "xsd:float"}).text =  formatted_confidence # Add the confidence measure
    if return_rdf:
        return rdf
    else:
        xml_str = xml.dom.minidom.parseString(tostring(rdf)).toprettyxml(indent="  ")  # Beautify the XML
        return xml_str
