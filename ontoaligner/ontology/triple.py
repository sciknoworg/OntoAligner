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
from rdflib import Graph, URIRef, RDFS, BNode

from ..base import  OMDataset
from ..base import BaseOntologyParser



track = "GraphTriple"

class GraphTripleOntology(BaseOntologyParser):
    """
    A parser that extracts subject-predicate-object triples from an ontology RDF graph,
    filtering out blank nodes and non-informative concepts.
    """
    def __init__(self, language: str = 'en'):
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
        Extracts labeled triples (subject, predicate, object) from the RDF graph.
        """
        iri2label, triples = {}, []
        for s, _, o in graph.triples((None, RDFS.label, None)):
            if isinstance(s, URIRef):
                iri2label[s] = str(o)
        for s, p, o in graph:
            if isinstance(s, BNode) or isinstance(o, BNode) or isinstance(p, BNode):
                continue
            subj = iri2label.get(s, s.split("#")[-1] if "#" in s else str(s))
            pred = iri2label.get(p, p.split("#")[-1] if "#" in p else str(p))
            obj =  iri2label.get(o, o.split("#")[-1] if "#" in o else str(o))
            if subj is None or pred is None or obj is None:
                continue
            if obj not in ['Class', 'Thing']:
                triples.append({
                    "subject": (str(s), subj),
                    "predicate": (str(p), pred),
                    "object": (str(o), obj)
                })
        return triples


class GraphTripleOMDataset(OMDataset):
    """
    A dataset class for working with the Source-Target ontology.
    """
    track = track
    ontology_name = "Source-Target"
    source_ontology = GraphTripleOntology()
    target_ontology = GraphTripleOntology()
