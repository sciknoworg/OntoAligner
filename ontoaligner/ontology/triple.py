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
from rdflib import URIRef, BNode, Graph, RDFS, RDF, OWL

from ..base import  OMDataset
from ..base import BaseOntologyParser

track = "GraphTriple"

class GraphTripleOntology(BaseOntologyParser):
    """
    A parser that extracts subject-predicate-object triples from an RDF/OWL ontology graph.

    This class filters out blank nodes and non-informative constructs, while associating
    human-readable labels with each IRI. It also marks which entities are declared as OWL or RDFS classes.

    Attributes:
        language (str): Language code used for label extraction (currently not applied directly, reserved for future use).

    Example:
        >>> parser = GraphTripleOntology(language='en')
        >>> graph = parser.load_ontology("ontology.owl")
        >>> triples = parser.extract_data(graph)
        >>> print(triples[0])
        ... {
        ...    'subject': ('http://example.org#Gene', 'Gene'),
        ...    'predicate': ('http://www.w3.org/2000/01/rdf-schema#subClassOf', 'subClassOf'),
        ...    'object': ('http://example.org#Entity', 'Entity'),
        ...    'subject_is_class': True,
        ...    'object_is_class': True
        ...}
    """
    def __init__(self, language: str = 'en'):
        """Initialize the ontology parser with the specified language."""
        self.language = language

    def load_ontology(self, input_file_path: str):
        """
        Loads the RDF/OWL ontology from the given file path.
        """
        self.graph = Graph()
        self.graph.parse(input_file_path)
        return self.graph

    def extract_data(self, graph):
        """
        Extracts labeled triples from the RDF graph, filtering blank nodes
        and annotating class membership.

        Args:
            graph (rdflib.Graph): The RDFLib graph to extract triples from.

        Returns:
            List[Dict]: A list of dictionaries, each representing a triple with labels and class info.
        """
        iri2label, triples = {}, []

        # Step 1: Collect all rdfs:label values
        for s, _, o in graph.triples((None, RDFS.label, None)):
            if isinstance(s, URIRef):
                iri2label[s] = str(o)

        # Step 2: Identify all class IRIs (rdf:type owl:Class or rdfs:Class)
        class_iris = set()
        for s in graph.subjects(RDF.type, OWL.Class):
            class_iris.add(s)
        for s in graph.subjects(RDF.type, RDFS.Class):
            class_iris.add(s)

        # Step 3: Label fallback function
        def get_label(iri):
            return iri2label.get(iri, iri.split("#")[-1] if "#" in iri else iri.split("/")[-1])

        # Step 4: Extract and annotate triples
        for s, p, o in graph:
            if any(isinstance(x, BNode) for x in (s, p, o)):
                continue
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                subj_label = get_label(s)
                pred_label = get_label(p)
                obj_label = get_label(o)

                triples.append({
                    "subject": (str(s), subj_label),
                    "predicate": (str(p), pred_label),
                    "object": (str(o), obj_label),
                    "subject_is_class": s in class_iris,
                    "object_is_class": o in class_iris
                })

        return triples


class GraphTripleOMDataset(OMDataset):
    """
    A dataset class representing a source-target ontology mapping task,
    using subject-predicate-object triples extracted from RDF graphs.

    Attributes:
        track (Any): The benchmark track for evaluation.
        ontology_name (str): The name of the dataset/ontology pair.
        source_ontology (GraphTripleOntology): Parsed source ontology with labeled triples.
        target_ontology (GraphTripleOntology): Parsed target ontology with labeled triples.
    """
    track = track
    ontology_name = "Source-Target"
    source_ontology = GraphTripleOntology()
    target_ontology = GraphTripleOntology()
