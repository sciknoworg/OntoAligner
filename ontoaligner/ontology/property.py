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

from rdflib.namespace import RDFS, OWL, DCTERMS, SKOS
from rdflib import Graph, URIRef, BNode, RDF
from enum import Enum

from typing import Literal
from collections import Counter


from ..base import OMDataset, BaseOntologyParser

track = "PropertyMatching"


def get_n(e, g):
    type_name = type(e).__name__
    if type_name in {'BNode'}:
        return str(e)
    elif type_name in {'URIRef'}:
        v = e.n3(g.namespace_manager)
        if ':' in v:
            return v.split(':')[1]
        elif v.startswith('<') and v.endswith('>'):
            return v[1:-1]
        else:
            return v
    elif type_name in {'Literal'}:
        return str(e)
    else:
        raise Exception(f'Type {type_name} not found.')

def get_lc(c):
    if c.islower():
        return 0
    if c.isupper():
        return 1
    return 2

def get_char_class(c):
    if len(c) <= 0:
        return -1
    if c.isalpha():
        return 0
    if c.isnumeric():
        return 1
    if c.isspace():
        return 2
    if not c.isalnum():
        return 3

def split_w(w):
    split = []
    lc = -1
    sb = ''
    for c in list(w):
        cc = get_lc(c)
        if cc == 1 and lc != 1 and lc != -1:
            split.append(sb)
            sb = ''
        sb += c
        lc = cc
    split.append(sb)
    return split

def split_sent(e):
    if len(e) <= 1:
        return [(e, get_char_class(e))]
    split = []
    lc = get_char_class(e[0])
    sb = e[0]
    for c in list(e)[1:]:
        cc = get_char_class(c)
        if cc == lc:
            sb += c
        else:
            split.append((sb, lc))
            sb = c
        lc = cc

    split.append((sb, lc))
    return split

def tokenize(e):
    s = split_sent(e)
    split = []
    for w, t in s:
        if t == 0:
            split += split_w(w)
        elif t == 1:
            split.append(w)
    return split

def is_restriction(entity, graph):
    """
    Check if an entity is a restriction in a graph.
    :param entity:
    :param graph:
    :return:
    """
    return graph.value(entity, RDF.type) == OWL.Restriction

def is_joinable(entity, graph):
    """
    Check if an entity is joinable in a graph. An entity is joinable if it has only one predicate and that predicate is
    a unionOf predicate.
    :param entity:
    :param graph:
    :return:
    """
    preds = set(graph.predicates(entity)).difference({RDF.type})
    return len(preds) == 1 and OWL.unionOf in preds

def flat_rdf_list_chain(entity, graph):
    """
    Convert and RDF first rest tree to a list.
    :param entity:
    :param graph:
    :return:
    """
    if graph.value(entity, RDF.rest) == RDF.nil:
        return [graph.value(entity, RDF.first)]
    else:
        return [graph.value(entity, RDF.first)] + flat_rdf_list_chain(graph.value(entity, RDF.rest), graph)

def join_nodes(nodes, graph):
    """
    Join a list of nodes in a graph.
    :param nodes:
    :param graph:
    :return:
    """
    return '_'.join(list(map(lambda x: get_n(x, graph), nodes)))

def get_concatenated_predicate_entities(entity, graph):
    """
    Get the concatenation of the predicates of an entity in a graph.
    :param entity:
    :param graph:
    :return:
    """
    not_type_predicates = list(set(graph.predicates(entity)).difference({RDF.type}))
    tmp = not_type_predicates + flat_rdf_list_chain(graph.value(entity, not_type_predicates[0]), graph)
    objs = list(map(lambda x: get_n(x, graph), tmp))
    return '_'.join(objs), len(objs)

def flat_restriction(entity, graph):
    """
    Flatten a restriction in a graph.
    :param entity:
    :param graph:
    :return:
    """
    nodes = []
    for s, p, o in graph.triples((entity, None, None)):
        if p == RDF.type:
            continue
        if type(o) is BNode:
            nodes.extend(flat_restriction(o, graph))
        else:
            nodes.extend([p, o])

    return nodes


def get_property_sentence(entity, graph, predicate):
    """
    Get the property sentence of an entity in a graph.
    :param entity:
    :param graph:
    :param predicate:
    :return:
    """
    sentence = []
    objects = list(graph.objects(entity, predicate))
    object_count = len(objects)
    for obj in objects:
        if type(obj) is BNode:
            if is_joinable(obj, graph):
                name, oc = get_concatenated_predicate_entities(obj, graph)
                object_count += oc - 1
            elif is_restriction(obj, graph):
                name = join_nodes(flat_restriction(obj, graph), graph)
            else:
                name = get_n(obj, graph)
        else:
            name = get_n(obj, graph)
        sentence.extend(map(str.lower, tokenize(name)))

    return sentence, object_count


def get_type(entity, graph):
    """Get the type of an entity."""
    if type(entity) is Literal:
        return entity.datatype

    parent = graph.value(entity, DCTERMS.subject)
    if parent is None:
        return entity

    return parent


def join_entities(entity_counts, separator='_'):
    """
    Join entities with their counts into a single URI string.

    Args:
        entity_counts: List of (entity, count) tuples
        separator: String to join entities

    Returns:
        Joined URI string
    """
    return separator.join([str(entity) for entity, count in entity_counts])


def get_type_hierarchy(entity, graph, max_depth=1):
    """Get the type hierarchy of an entity up to max_depth."""
    if type(entity) is Literal:
        return [entity.datatype]

    entity_parent = graph.value(entity, DCTERMS.subject)
    if entity_parent is None:
        return [entity]

    hierarchy = [entity_parent]

    for _ in range(max_depth):
        if graph.value(entity_parent, SKOS.broader) is None:
            break
        entity_parent = graph.value(entity_parent, SKOS.broader)
        hierarchy.append(entity_parent)

    return hierarchy

def most_common_pair(graph: Graph) -> Graph:
    """
    Infer domain and range for properties by finding the most common
    (subject_type, object_type) pairs in actual property usage.

    This strategy looks at how properties are actually used in the ontology
    and assigns domain/range based on the most frequent type pair.

    Args:
        graph: Input RDF graph

    Returns:
        New graph with inferred domain/range declarations
    """
    # Find all properties
    props = set()
    for s, p, o in graph.triples((None, RDF.type, RDF.Property)):
        props.add(s)

    new_graph = Graph()
    new_graph.namespace_manager = graph.namespace_manager

    # Copy all original triples
    for s, p, o in graph:
        new_graph.add((s, p, o))

    pair_counter = {}

    # Analyze actual property usage
    for prop in props:
        for s, p, o in graph.triples((None, prop, None)):
            subject_type = get_type(s, graph)
            object_type = get_type(o, graph)

            if p not in pair_counter:
                pair_counter[p] = Counter()

            pair_counter[p][(subject_type, object_type)] += 1

    # Add inferred domain/range based on most common pair
    for prop, counter in pair_counter.items():
        if counter:
            most_common = counter.most_common(1)[0]
            domain, rng = most_common[0]

            # Only add if not already present
            if not list(graph.objects(prop, RDFS.domain)):
                new_graph.add((prop, RDFS.domain, domain))
            if not list(graph.objects(prop, RDFS.range)):
                new_graph.add((prop, RDFS.range, rng))

    return new_graph

def most_common_domain_range_pair(graph: Graph, max_depth: int = 1,
                                  most_common_count: int = 5) -> Graph:
    """
    Infer domain and range by analyzing type hierarchies and finding the most
    common types at different depths.

    This more sophisticated strategy considers type hierarchies and can identify
    multiple common domains/ranges, joining them into a composite description.

    Args:
        graph: Input RDF graph
        max_depth: How far up the type hierarchy to look
        most_common_count: How many top types to consider

    Returns:
        New graph with inferred domain/range declarations
    """
    # Find all properties
    properties = set()
    for s, p, o in graph.triples((None, RDF.type, RDF.Property)):
        properties.add(s)

    new_graph = Graph()
    new_graph.namespace_manager = graph.namespace_manager

    # Copy all original triples
    for s, p, o in graph:
        new_graph.add((s, p, o))

    pair_counter = {}

    # Analyze property usage with type hierarchies
    for prop in properties:
        for s, p, o in graph.triples((None, prop, None)):
            subject_types = get_type_hierarchy(s, graph, max_depth=max_depth)
            object_types = get_type_hierarchy(o, graph, max_depth=max_depth)

            if p not in pair_counter:
                pair_counter[p] = {'domain': Counter(), 'range': Counter()}

            # Count all types in hierarchy
            for st in subject_types:
                pair_counter[p]['domain'][st] += 1

            for ot in object_types:
                pair_counter[p]['range'][ot] += 1

    # Add inferred domain/range based on most common types
    for prop, counts in pair_counter.items():
        domain_common = counts['domain'].most_common(most_common_count)
        range_common = counts['range'].most_common(most_common_count)

        if domain_common:
            joined_domain = join_entities(domain_common)
            # Only add if not already present
            if not list(graph.objects(prop, RDFS.domain)):
                new_graph.add((prop, RDFS.domain, URIRef(joined_domain)))

        if range_common:
            joined_range = join_entities(range_common)
            # Only add if not already present
            if not list(graph.objects(prop, RDFS.range)):
                new_graph.add((prop, RDFS.range, URIRef(joined_range)))

    return new_graph


class ProcessingStrategy(str, Enum):
    """
    Preprocessing strategies for ontology graphs before property matching.

    These strategies infer domain/range information from actual property usage
    in the ontology instances when explicit declarations are missing.
    """
    NONE = "none"
    MOST_COMMON_PAIRS = "most_common_pairs"
    MOST_COMMON_DOMAIN_RANGE = "most_common_domain_range"



class OntologyProperty(BaseOntologyParser):
    """
    A parser specifically designed for the PropertyMatcher pipeline.

    This parser loads RDF/OWL ontologies and ensures they contain the necessary
    structure for property matching, including:
    - Properties with rdfs:domain and rdfs:range declarations
    - Proper class hierarchies
    - Labels and annotations

    The PropertyMatcher pipeline requires:
    1. Properties defined with RDFS.domain and RDFS.range
    2. Entity labels accessible via get_n() function
    3. Support for blank nodes, restrictions, and unions
    4. RDF/OWL structure for property identification

    Attributes:
        language (str): Language code for label extraction (default: 'en')
        graph (rdflib.Graph): The loaded RDF graph

    Example:
        >>> parser = OntologyProperty(language='en', processing_strategy= ProcessingStrategy.NONE)
        >>> graph = parser.load_ontology("source_ontology.owl")
        >>> properties = parser.get_properties()
        >>> print(f"Found {len(properties)} properties")
    """

    def __init__(self, language: str = 'en', processing_strategy: ProcessingStrategy = ProcessingStrategy.NONE):
        """Initialize the ontology parser with the specified language."""
        self.language = language
        self.graph = None
        self.processing_strategy = processing_strategy

    def load_ontology(self, input_file_path: str):
        """
        Loads the RDF/OWL ontology from the given file path.

        The loaded graph maintains all RDF/OWL constructs needed by PropertyMatcher:
        - Property declarations (rdfs:domain, rdfs:range)
        - Class hierarchies
        - Blank nodes and restrictions
        - Labels and annotations
        - Inverse properties (owl:inverseOf)

        Args:
            input_file_path (str): Path to the ontology file (OWL, RDF/XML, Turtle, etc.)

        Returns:
            rdflib.Graph: The loaded RDF graph with all triples preserved
        """
        self.graph = Graph()
        self.graph.parse(input_file_path)
        if self.processing_strategy == ProcessingStrategy.MOST_COMMON_PAIRS:
            self.graph = most_common_pair(self.graph)
        elif self.processing_strategy == ProcessingStrategy.MOST_COMMON_DOMAIN_RANGE:
            self.graph = most_common_domain_range_pair(self.graph)
        return self.graph


    def extract_data(self, graph):
        """
        Extracts property-centric data from the RDF graph for the PropertyMatcher pipeline.

        This method identifies and structures properties suitable for matching:
        - Filters entities to find those with rdfs:domain and rdfs:range
        - Collects property labels and their domains/ranges
        - Preserves blank node structures for complex domain/range specifications
        - Maintains inverse property relationships

        Args:
            graph (rdflib.Graph): The RDFLib graph to extract properties from

        Returns:
            List[Dict]: A list of dictionaries, each representing a property with:
                - 'property_uri': The URI of the property
                - 'label': Human-readable label
                - 'domain': List of domain class URIs (may include blank nodes)
                - 'range': List of range class URIs (may include blank nodes)
                - 'inverse_of': URI of inverse property if declared, else None

        Example:
            >>> properties = parser.extract_data(graph)
            >>> prop = properties[0]
            >>> print(prop)
            >>> {
            >>>    'iri': 'http://example.org#hasAuthor',
            >>>    'label': 'has author',
            >>>    'domain': ['http://example.org#Publication'],
            >>>    'domain_text': ['has author', 'publication'],
            >>>    'range': ['http://example.org#Person'],
            >>>    'range_text': ['has author', 'person'],
            >>>    'inverse_of': 'http://example.org#authorOf',
            >>>    'inverse_label': 'authorOf'
            >>>}
        """
        properties = []

        # Identify all properties: entities with both rdfs:domain and rdfs:range
        for subject in graph.subjects():
            # Check if this is a property (has domain and range)
            has_domain = (subject, RDFS.domain, None) in graph
            has_range = (subject, RDFS.range, None) in graph

            if has_domain and has_range:
                # Extract property information
                prop_data = {
                    'iri': str(subject),
                    'label':   self._get_label(subject, graph),
                    'domain': [str(o) for o in graph.objects(subject, RDFS.domain)],
                    'range': [str(o) for o in graph.objects(subject, RDFS.range)],
                    'inverse_of': None,
                    'inverse_label': None,
                }

                # Check for inverse property
                inverse_uri = graph.value(subject, OWL.inverseOf)
                if inverse_uri:
                    prop_data['inverse_of'] = str(inverse_uri)
                    prop_data['inverse_label'] = self._get_label(inverse_uri, graph)

                domain_labels, _ = get_property_sentence(subject, graph, RDFS.domain)
                prop_data['domain_text'] = domain_labels

                range_labels, _ = get_property_sentence(subject, graph, RDFS.range)
                prop_data['range_text'] = range_labels

                properties.append(prop_data)

        return properties

    def _get_label(self, entity, graph):
        """
        Helper method to extract the label of an entity.

        Args:
            entity: RDF entity (URIRef or BNode)
            graph: RDFLib graph

        Returns:
            str: Label or local name of the entity
        """
        # Try rdfs:label first
        label = graph.value(entity, RDFS.label)
        if label:
            return str(label)

        # Fall back to extracting from URI
        entity_str = str(entity)
        if '#' in entity_str:
            return entity_str.split('#')[-1]
        elif '/' in entity_str:
            return entity_str.split('/')[-1]

        return entity_str

    def get_properties(self):
        """
        Convenience method to get all properties from the loaded graph.

        Returns:
            set: Set of all property URIs that have domain and range

        Raises:
            ValueError: If no ontology has been loaded yet
        """
        if self.graph is None:
            raise ValueError("No ontology loaded. Call load_ontology() first.")

        properties = set()
        for subject in self.graph.subjects():
            has_domain = (subject, RDFS.domain, None) in self.graph
            has_range = (subject, RDFS.range, None) in self.graph
            if has_domain and has_range:
                properties.add(subject)

        return properties

    def get_classes(self):
        """
        Get all classes from the loaded graph.

        Returns:
            set: Set of all class URIs (owl:Class and rdfs:Class)

        Raises:
            ValueError: If no ontology has been loaded yet
        """
        if self.graph is None:
            raise ValueError("No ontology loaded. Call load_ontology() first.")

        classes = set()
        for s in self.graph.subjects(RDF.type, OWL.Class):
            classes.add(s)
        for s in self.graph.subjects(RDF.type, RDFS.Class):
            classes.add(s)

        return classes


class PropertyOMDataset(OMDataset):
    """
    A dataset class for working with the Source-Target ontology.
    """
    track = track
    ontology_name = "Source-Target-Properties"
    source_ontology = OntologyProperty()
    target_ontology = OntologyProperty()

    def __init__(self, processing_strategy: ProcessingStrategy = ProcessingStrategy.NONE):
        """
        Initialize dataset with optional preprocessing strategy.

        Args:
            processing_strategy: Strategy for inferring domain/range
        """
        super().__init__()
        self.source_ontology = OntologyProperty(processing_strategy=processing_strategy)
        self.target_ontology = OntologyProperty(processing_strategy=processing_strategy)
