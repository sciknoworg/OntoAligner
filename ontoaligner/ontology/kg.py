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
KG dataset classes for the FLORA (Fuzzy Logic KG Alignment) aligner.

This module provides the **parsing layer** of the standard OntoAligner pipeline for
FLORA.  It covers three families of knowledge-graph benchmarks:

1. **OAEI / Turtle files** (:class:`FLORAOntology` / :class:`FLORAOMDataset`):
   Reads any single Turtle (``.ttl``) or RDF/XML (``.xml``) file.

2. **OpenEA benchmarks** (:class:`FLORAOpenEAKnowledgeBase` / :class:`FLORAOpenEAOMDataset`):
   Reads individual per-KG triple files (``rel_triples_*``, ``attr_triples_*``)
   one side at a time.  Ground-truth pairs are loaded separately by
   :class:`OpenEAAlignmentsParser` from the ``ent_links`` file.

3. **DBP15K benchmarks** (:class:`FLORADBpedia15KKnowledgeBase` / :class:`FLORADBpediaOMDataset`):
   Reads per-KG ID-mapped files (``rel_ids_*``, ``ent_ids_*``, ``triples_*``,
   ``att_triples_*``) one side at a time.  Ground-truth pairs are loaded
   separately by :class:`DBpediaAlignmentsParser` from ``ref_ent_ids``.

For every family the extracted data follows the same schema consumed by
:class:`~ontoaligner.encoder.flora.FLORAEncoder`:

- ``"entities"``   — ``[{"iri": str, "label": str}, ...]``
- ``"predicates"`` — predicate → count mapping
- ``"triples"``    — ``[(subject, predicate, object), ...]``
- ``"graph"``      — native :class:`Graph` object forwarded to the aligner

Ground-truth alignment files are parsed by:

- :class:`OpenEAAlignmentsParser`  — OpenEA ``ent_links`` TSV format.
- :class:`DBpediaAlignmentsParser` — DBP15K ``ref_ent_ids`` TSV format.

Classes:
    Graph,
    FLORAOntology,
    FLORAOpenEAKnowledgeBase,
    FLORADBpedia15KKnowledgeBase,
    OpenEAAlignmentsParser,
    DBpediaAlignmentsParser,
    FLORAOMDataset,
    FLORAOpenEAOMDataset,
    FLORADBpediaOMDataset
"""
from typing_extensions import override

from ..base import BaseOntologyParser, OMDataset, BaseAlignmentsParser

from typing import Any, Dict, List, Optional, Tuple
import codecs
import re
import os
import tempfile
import sys
from io import StringIO
from collections import defaultdict
from functools import reduce
from rdflib import Graph as RDFGraph
from abc import ABC
import xml.etree.ElementTree as ET

track = "KG Alignment - FLORA"
_BLACK_NODE_COUNTER=0

LITERAL_REGEX=re.compile('"([^"]*)"(@([a-z-]+))?(\\^\\^(.*))?') # Regex for literals
INTEGER_REGEX=re.compile('^"?[+-]?[0-9.]+"?$') # Regex for int values
FLOAT_REGEX = re.compile('^"?([+-])?([0-9.]+)"?$')

def is_inverse(rel):
    """ TRUE if the relation is an inverse relation """
    return rel[-1]=='-'

def invert(rel):
    """ Returns the inverse of a relation """
    return rel[:-1] if is_inverse(rel) else rel+'-'

def is_literal(term):
    try:
        return re.match(LITERAL_REGEX, term) or re.match(INTEGER_REGEX, term) or re.match(FLOAT_REGEX, term)
    except TypeError:
        return False # if there is no match or exception then the term is None

class Graph(object):
    """
    In-memory directed multigraph for FLORA knowledge-graph alignment.

    Triples are stored in two indices for efficient lookup:

    - ``index``    — ``{subject: {predicate: {objects}}}``
    - ``relindex`` — ``{predicate: {subject: {objects}}}``

    Inverse arcs are added automatically: for every ``(s, p, o)`` the
    triple ``(o, p+"-", s)`` is also stored so that graph traversal is
    bidirectional without extra lookups.

    Attributes:
        index (dict): Primary subject → predicate → objects index.
        relindex (dict): Predicate → subject → objects index.
        pred2num (dict | None): Cached predicate → fact-count mapping;
            invalidated on every :meth:`add` or :meth:`remove` call.
    """

    def __init__(self):
        self.index = {}
        self.relindex = {}

    def add(self, triple):
        """
        Add a ``(subject, predicate, object)`` triple to the graph.

        An inverse arc ``(object, predicate+"-", subject)`` is automatically
        inserted.  The ``pred2num`` cache is invalidated.

        Args:
            triple (Tuple[str, str, str]): The triple to add.
        """
        (subject, predicate, obj) = triple
        if subject not in self.index:
            self.index[subject] = {}
        m = self.index[subject]
        if predicate not in m:
            m[predicate] = set()
        m[predicate].add(obj)

        # relindex
        if predicate not in self.relindex:
            self.relindex[predicate] = {}
        m = self.relindex[predicate]
        if subject not in m:
            m[subject] = set()
        m[subject].add(obj)

        if not is_inverse(predicate):
            self.add((obj, invert(predicate), subject))
        self.pred2num = None

    def remove(self, triple):
        """
        Remove a ``(subject, predicate, object)`` triple from the graph.

        The corresponding inverse arc is also removed.
        The ``pred2num`` cache is invalidated.

        Args:
            triple (Tuple[str, str, str]): The triple to remove.
        """
        (subject, predicate, obj) = triple
        if subject not in self.index:
            return
        m = self.index[subject]
        if predicate not in m:
            return
        m[predicate].discard(obj)
        if len(m[predicate]) == 0:
            self.index[subject].pop(predicate)
            if len(self.index[subject]) == 0:
                self.index.pop(subject)
        if not is_inverse(predicate):
            self.remove((obj, invert(predicate), subject))
        self.pred2num = None

    def __contains__(self, triple):
        """
        Return ``True`` if the triple is present in the graph.

        Args:
            triple (Tuple[str, str, str]): ``(subject, predicate, object)``.

        Returns:
            bool: Whether the triple exists.
        """
        (subject, predicate, obj) = triple
        if subject not in self.index:
            return False
        m = self.index[subject]
        if predicate not in m:
            return False
        return obj in m[predicate]

    def __iter__(self):
        """
        Iterate over all ``(subject, predicate, object)`` triples in the graph.

        Yields:
            Tuple[str, str, str]: Each triple stored in the graph.
        """
        for s in self.index:
            for p in self.index[s]:
                for o in self.index[s][p]:
                    yield (s, p, o)

    def load_turtle_file(self, file, message=None):
        """
        Parse a Turtle file and add all its triples to this graph.

        Args:
            file (str): Path to the Turtle (``.ttl``) file.
            message (str | None): Optional progress message printed to stdout.
        """
        for triple in parse_turtle_triples(file, message):
            self.add(triple)

    def get_list(self, list_start):
        """
        Collect the elements of an RDF list starting at ``list_start``.

        Args:
            list_start (str): IRI of the first ``rdf:List`` node.

        Returns:
            List[str]: Ordered list of element IRIs / literals.
        """
        result = []
        while list_start and list_start != 'rdf:nil':
            result.extend(self.index[list_start].get('rdf:first', []))
            if 'rdf:rest' not in self.index[list_start]:
                break
            list_start = list(self.index[list_start]['rdf:rest'])[0]
        return result

    def predicates(self):
        """
        Return the predicate → fact-count mapping for this graph.

        The result is cached after the first call and recomputed whenever
        :meth:`add` or :meth:`remove` invalidates the cache.

        Returns:
            Dict[str, int]: Mapping of predicate IRI to the number of triples
            that use it (counting both forward and inverse arcs).
        """
        if not self.pred2num:
            self.num_facts_with_predicate("blah")
        return self.pred2num

    def attributes(self):
        """
        Return the set of all attribute predicates in the graph.

        A predicate is considered an *attribute* if at least one of its
        objects is a literal value.  Both the forward and inverse forms of
        each attribute predicate are included.

        Returns:
            Set[str]: Set of attribute predicate IRIs (and their inverses).
        """
        result = set()
        for predicate in self.relindex:
            if self.is_attribute(predicate):
                result.add(predicate)
                result.add(invert(predicate))  # add inverse
        return result

    def num_facts_with_predicate(self, predicate):
        """
        Return the number of triples that use the given predicate.

        Builds and caches the ``pred2num`` mapping on the first call.

        Args:
            predicate (str): Predicate IRI to look up.

        Returns:
            int: Count of triples with that predicate, or ``0`` if absent.
        """
        if self.pred2num:
            return self.pred2num[predicate] if predicate in self.pred2num else 0
        self.pred2num = {}
        for subject in self.index:
            for pred in self.index[subject]:
                if pred not in self.pred2num:
                    self.pred2num[pred] = 0
                self.pred2num[pred] += len(self.index[subject][pred])
        return self.num_facts_with_predicate(predicate)

    def is_attribute(self, pred):
        """
        Return ``True`` if at least one object of ``pred`` is a literal.

        Args:
            pred (str): Predicate IRI to test.

        Returns:
            bool: ``True`` if the predicate has any literal objects.
        """
        if pred in self.relindex:
            for literal in self.relindex[pred]:
                if is_literal(literal):
                    return True
        return False

    def local_functionality(self, subjects, preds):
        """
        Compute the local functionality score for a set of (subject, predicate) pairs.

        The score is ``1 / |common_objects|``, where ``common_objects`` is the
        intersection of object sets for all (subject, predicate) pairs.
        Returns ``0`` if the intersection is empty.

        Args:
            subjects (str | List[str]): One or more subject IRIs.
            preds    (str | List[str]): One or more predicate IRIs, parallel to ``subjects``.

        Returns:
            float: Local functionality value in ``[0, 1]``.

        Raises:
            ValueError: If ``subjects`` and ``preds`` have different lengths.
        """
        if not isinstance(subjects, (list, tuple)):
            subjects = [subjects]
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        if len(subjects) != len(preds):
            raise ValueError("The input size of subjects and predicates are not equal.")
        common_objs = None
        for i in range(len(subjects)):
            if common_objs is None:
                common_objs = self.index[subjects[i]][preds[i]]
            else:
                common_objs = common_objs & self.index[subjects[i]][preds[i]]
        try:
            value = 1.0 / len(common_objs)
        except ZeroDivisionError:
            value = 0
        return value

    def objects(self, subject=None, predicate=None):
        """
        Return all objects reachable from ``subject`` via ``predicate``.

        If ``subject`` is ``None`` every subject is iterated; if ``predicate``
        is ``None`` every predicate of the given subject is iterated.

        Args:
            subject   (str | None): Subject IRI filter; ``None`` means all subjects.
            predicate (str | None): Predicate IRI filter; ``None`` means all predicates.

        Returns:
            Set[str]: Set of matching object IRIs / literals.
        """
        # We create a copy here instead of using a generator
        # because the user loop may want to change the graph
        result = []
        if subject and subject not in self.index:
            return result
        for s in ([subject] if subject else self.index):
            for p in ([predicate] if predicate else self.index[s]):
                if p in self.index[s]:
                    result.extend(self.index[s][p])
        return set(result)

    def subjects(self, predicate=None, object=None):
        """
        Return all subjects that have ``object`` as an object via ``predicate``.

        Implemented by looking up the inverse predicate in :meth:`objects`.

        Args:
            predicate (str | None): Predicate IRI filter.
            object    (str | None): Object IRI / literal filter.

        Returns:
            Set[str]: Set of matching subject IRIs.
        """
        pred = invert(predicate) if predicate else None
        return self.objects(subject=object, predicate=pred)

    def triples_with_object(self, obj, predicates=[]):
        """
        Yield all triples whose object is ``obj``, optionally filtered by predicates.

        Implemented by inverting the predicate list and delegating to
        :meth:`triples_with_subject`.

        Args:
            obj        (str): Object IRI / literal to match.
            predicates (List[str]): If non-empty, restrict to these predicates.

        Yields:
            Tuple[str, str, str]: Matching ``(subject, predicate, object)`` triples.
        """
        return self.triples_with_subject(obj, [invert(p) for p in predicates])

    def triples_with_subject(self, subject, predicates=[]):
        """
        Yield all triples with the given subject, optionally filtered by predicates.

        Args:
            subject    (str): Subject IRI to match.
            predicates (List[str]): If non-empty, restrict to these predicates.

        Yields:
            Tuple[str, str, str]: Matching ``(subject, predicate, object)`` triples.
        """
        for predicate in predicates if len(predicates) else self.index[subject].keys():
            if predicate in self.index[subject]:
                for obj in self.index[subject][predicate]:
                    yield (subject, predicate, obj)

    def triples_with_predicate(self, *predicates):
        """
        Yield all triples that use any of the given predicates.

        Args:
            *predicates (str): One or more predicate IRIs to match.

        Yields:
            Tuple[str, str, str]: Matching ``(subject, predicate, object)`` triples.
        """
        for subject in self.index:
            for predicate in predicates:
                if predicate in self.index[subject]:
                    for object in self.index[subject][predicate]:
                        yield (subject, predicate, object)

    def head_triples_with_predicate_list(self, predicates_with_count):
        """
        Return triples grouped by head entity for heads that carry all listed predicates.

        Only subjects that appear in the ``relindex`` for *every* predicate in
        ``predicates_with_count`` are included.  At most 10 objects per
        (subject, predicate) pair are returned to bound memory usage.

        Args:
            predicates_with_count (Dict[str, int]): Mapping of predicate IRI to the
                minimum number of objects required for a subject to qualify.

        Returns:
            Dict[str, Set[Tuple]]: ``{subject: set of (subject, predicate, object) triples}``.
        """
        result = {}  # {head: pred: tail}
        heads = [set(self.relindex[pred].keys()) for pred in predicates_with_count]
        shared_heads = reduce(lambda x, y: x & y, heads)

        for predicate in set(predicates_with_count):
            for subject in shared_heads:
                if len(self.relindex[predicate][subject]) < predicates_with_count[predicate]:
                    continue
                if subject not in result:
                    result[subject] = set()
                pred2count_ = defaultdict(int)
                for obj in self.relindex[predicate][subject]:
                    result[subject].add((subject, predicate, obj))
                    pred2count_[predicate] += 1
                    if pred2count_[predicate] > 10:
                        break
        return result

    def print_to_writer(self, result):
        """
        Serialize the graph to a Turtle-like text representation.

        RDF list nodes (``_:list_*``) are expanded inline using the ``(...)``
        collection syntax.  Inverse predicates are omitted.

        Args:
            result: A writable text stream (e.g. ``io.StringIO``).
        """
        for subject in self.index:
            if subject.startswith("_:list_"):
                continue
            result.write("\n")
            result.write(subject)
            result.write(' ')
            hasPreviousPred = False
            for predicate in self.index[subject]:
                if is_inverse(predicate):
                    continue
                if hasPreviousPred:
                    result.write(' ;\n\t')
                hasPreviousPred = True
                result.write(predicate)
                result.write(' ')
                hasPrevious = False
                for obj in self.index[subject][predicate]:
                    if hasPrevious:
                        result.write(', ')
                    if obj.startswith("_:list_"):
                        result.write("(")
                        result.write(" ".join(self.get_list(obj)))
                        result.write(")")
                    else:
                        result.write(obj)
                    hasPrevious = True
            result.write(' .\n')

    def __str__(self):
        """
        Return a Turtle-like string representation of the graph.

        Returns:
            str: Full graph serialization as a string.
        """
        buffer = StringIO()
        buffer.write("# RDF Graph\n")
        self.print_to_writer(buffer)
        return buffer.getvalue()

    def some_subject(self):
        """
        Return an arbitrary subject IRI from the graph, or ``None`` if empty.

        Returns:
            str | None: One subject IRI, or ``None``.
        """
        for key in self.index:
            return key
        return None

    def __len__(self):
        """
        Return the total number of triples (facts) stored in the graph.

        Returns:
            int: Sum of fact counts across all predicates.
        """
        # Total number of facts
        count = 0
        for p in self.predicates():
            count += self.num_facts_with_predicate(p)
        return count


def print_error(*args, **kwargs):
    """Print an error message to *stderr*."""
    print(*args, file=sys.stderr, **kwargs)


def terms_and_separators(generator):
    """
    Yield tokens (terms and separators) from a character generator.

    Handles Turtle directives (``@prefix``, ``@base``), comments (``#``),
    quoted string literals (short and long form), URIs (``<...>``),
    collection / blank-node brackets, and prefixed / local names.

    Args:
        generator: An iterator that yields individual characters.

    Yields:
        str | None: The next Turtle token, or ``None`` at end-of-file.
    """
    push_back = None
    while True:
        # Scroll to next term
        while True:
            char = push_back if push_back else next(generator, None)
            push_back = None
            if not char:
                # end of file
                yield None
                return
            elif char == '@':
                # @base and @prefix
                for term in terms_and_separators(generator):
                    if not term:
                        print_error("Unexpected end of file in directive")
                        return
                    if term == '.':
                        break
            elif char == '#':
                # comments
                while char and char != '\n':
                    char = next(generator, None)
            elif char.isspace():
                # whitespace
                pass
            else:
                break

        # Strings
        if char == '"':
            second_char = next(generator, None)
            third_char = next(generator, None)
            if second_char == '"' and third_char == '"':
                # long string quote
                literal = ""
                while True:
                    char = next(generator, None)
                    if char:
                        literal = literal + char
                    else:
                        print_error("Unexpected end of file in literal", literal)
                        literal = literal + '"""'
                        break
                    if literal.endswith('"""'):
                        break
                literal = literal[:-3]
                char = None
            else:
                # Short string quote
                if second_char == '"':
                    literal = ''
                    char = third_char
                elif third_char == '"' and second_char != '\\':
                    literal = second_char
                    char = None
                else:
                    literal = [second_char, third_char]
                    if third_char == '\\' and second_char != '\\':
                        literal += next(generator, ' ')
                    while True:
                        char = next(generator, None)
                        if not char:
                            print_error("Unexpected end of file in literal", literal)
                            break
                        elif char == '\\':
                            literal += char
                            literal += next(generator, ' ')
                            continue
                        elif char == '"':
                            break
                        literal += char
                    char = None
                    literal = "".join(literal)
            # Make all literals simple literals without line breaks and quotes
            literal = literal.replace('\n', '\\n')\
                             .replace('\t', '\\t')\
                             .replace('\r', '')\
                             .replace('\\"', "'")\
                             .replace("\\u0022", "'")
            if not char:
                char = next(generator, None)
            if char == '^':
                # Datatypes
                next(generator, None)
                datatype = ''
                while True:
                    char = next(generator, None)
                    if not char:
                        print_error("Unexpected end of file in datatype of", literal)
                        break
                    if len(datatype) > 0 and datatype[0] != '<' and char != ':' and (
                            char < 'A' or char > 'z') and char != '/' and (char != '2' and char != '3') \
                            and char != 'ó' and char != 'ł':  # exceptions: m^2, /km2, ó, polishZłoty
                        push_back = char
                        break
                    datatype = datatype + char
                    if datatype.startswith('<') and datatype.endswith('>'):
                        break
                if not datatype or len(datatype) < 3:
                    print_error("Invalid literal datatype:", datatype)
                yield ('"' + literal + '"^^' + datatype)
            elif char == '@':
                # Languages
                language = ""
                while True:
                    char = next(generator, None)
                    if not char:
                        print_error("Unexpected end of file in language of", literal)
                        break
                    if (char >= 'A' and char <= 'Z') or (char >= 'a' and char <= 'z') or (
                            char >= '0' and char <= '9') or char == '-':
                        language += char
                        continue
                    push_back = char
                    break
                if not language or len(language) > 20 or len(language) < 2 or (
                        '-' in language and len(language[language.index('-'):]) > 9):
                    yield ('"' + literal + '"')
                else:
                    yield ('"' + literal + '"@' + language)
            else:
                push_back = char
                yield ('"' + literal + '"')
        elif char == '<':
            # URIs
            uri = []
            while char != '>':
                uri += char
                char = next(generator, None)
                if not char:
                    print_error("Unexpected end of file in URL", uri)
                    break
            uri += '>'
            yield "".join(uri)
        elif char in ['.', ',', ';', '[', ']', '(', ')']:
            # Separators
            yield char
        else:
            # Local names
            iri = []
            while not char.isspace() and char not in ['.', ',', ';', '[', ']', '"', "'", '^', '@', '(', ')']:
                iri += char
                char = next(generator, None)
                if not char:
                    print_error("Unexpected end of file in IRI", iri)
                    break
            push_back = char
            yield "".join(iri)

def generate_blank_node_name(subject, predicate=None):
    """
    Generate a human-readable blank-node name in the ``ys:`` namespace.

    The name is derived from the local parts of ``subject`` (and optionally
    ``predicate``) followed by a global counter so that names are unique
    within a parse run.

    Args:
        subject   (str): IRI of the subject associated with this blank node.
        predicate (str | None): IRI of the predicate (used for anonymous
            property nodes); ``None`` is allowed.

    Returns:
        str: A blank-node identifier such as ``"ys:Gene_subClassOf_42"``.
    """
    global _BLACK_NODE_COUNTER
    if ':' in subject:
        lastIndex = len(subject) - subject[::-1].index(':') - 1
        subject = subject[lastIndex + 1:] + "_"
    elif predicate:
        subject = ""
    if predicate and ':' in predicate:
        lastIndex = len(predicate) - predicate[::-1].index(':') - 1
        predicate = predicate[lastIndex + 1:]
    else:
        predicate = ""
    _BLACK_NODE_COUNTER += 1
    return "ys:" + subject + predicate + "_" + str(_BLACK_NODE_COUNTER)

def triples_from_terms(generator, predicates=None, given_subject=None):
    """
    Yield ``(subject, predicate, object)`` triples from a token generator.

    Handles the full Turtle grammar including ``;`` / ``,`` shorthand,
    inline blank-node blocks ``[...]``, and RDF collections ``(...)``.
    Expands ``a`` to ``rdf:type``.

    Args:
        generator     : Token generator as produced by :func:`terms_and_separators`.
        predicates    (List[str] | None): If given, only yield triples whose
            predicate is in this list.
        given_subject  (str | None): If inside a ``[...]`` block, the subject
            is supplied externally rather than read from the stream.

    Yields:
        Tuple[str, str, str]: Parsed ``(subject, predicate, object)`` triples.
    """
    while True:
        term = next(generator, None)
        if not term or term == ']':
            return
        if term == '.' or (term == ';' and given_subject):
            continue
        # If we're inside a [...]
        if given_subject:
            subject = given_subject
            if term != ',':
                predicate = term
                # If we're in a normal statement
        else:
            if term != ';' and term != ',':
                subject = term
            if term != ',':
                predicate = next(generator, None)
        if predicate == 'a':
            predicate = 'rdf:type'
        # read the object
        object = next(generator, None)
        if not object:
            print_error("File ended unexpectedly after", subject, predicate)
            return
        elif object in ['.', ',', ';']:
            print_error("Unexpected", object, "after", subject, predicate)
            return
        elif object == '(':
            list_node = generate_blank_node_name("list")
            previous_list_node = None
            yield (subject, predicate, list_node)
            while True:
                term = next(generator, None)
                if not term:
                    print_error("Unexpected end of file in collection (...)")
                    break
                elif term == ')':
                    break
                else:
                    if previous_list_node:
                        yield (previous_list_node, 'rdf:rest', list_node)
                    if term == '[':
                        term = generate_blank_node_name("element")
                        yield (list_node, 'rdf:first', term)
                        yield from triples_from_terms(generator, predicates, given_subject=term)
                    else:
                        yield (list_node, 'rdf:first', term)
                    previous_list_node = list_node
                    list_node = generate_blank_node_name("list")
            yield (previous_list_node, 'rdf:rest', 'rdf:nil')
        elif object == '[':
            object = generate_blank_node_name(subject, predicate)
            yield (subject, predicate, object)
            yield from triples_from_terms(generator, predicates, given_subject=object)
        else:
            if (not predicates) or (predicate in predicates):
                yield (subject, predicate, object)

def byte_generator(byte_reader):
    """
    Yield individual bytes from a binary reader.

    Args:
        byte_reader: A binary file object opened in ``"rb"`` mode.

    Yields:
        bytes: One byte at a time until EOF.
    """
    while True:
        b=byte_reader.read(1)
        if b:
            yield b
        else:
            break

def char_generator(byte_generator):
    """
    Decode a byte generator into a character generator using UTF-8.

    Args:
        byte_generator: An iterator of ``bytes`` objects.

    Returns:
        codecs.IncrementalDecoder iterator: Iterator of decoded characters.
    """
    return codecs.iterdecode(byte_generator, "utf-8")


def parse_turtle_triples(file, message=None, predicates=None):
    """
    Yield all ``(subject, predicate, object)`` triples from a Turtle file.

    Args:
        file       (str): Path to the Turtle (``.ttl``) file.
        message    (str | None): Optional progress label printed to stdout
            before and after parsing.
        predicates (List[str] | None): If given, only yield triples whose
            predicate is in this list.

    Yields:
        Tuple[str, str, str]: Each parsed triple.
    """
    if message:
        print(message+"... ",end="",flush=True)
    with open(file,"rb") as reader:
        yield from triples_from_terms(terms_and_separators(char_generator(byte_generator(reader))), predicates)
    if message:
        print("done", flush=True)


def parse_turtle_graph(file, message=None):
    """
    Parse a Turtle file into a :class:`Graph` object.

    Args:
        file    (str): Path to the Turtle (``.ttl``) file.
        message (str | None): Optional progress label passed to
            :func:`parse_turtle_triples`.

    Returns:
        Graph: A fully loaded :class:`Graph` containing all triples from the file.
    """
    graph=Graph()
    for triple in parse_turtle_triples(file, message):
        graph.add(triple)
    return graph


class FLORAOntology(BaseOntologyParser):
    """
    KG parser for single-file OAEI / Turtle knowledge graphs.

    Loads a Turtle (``.ttl``) or RDF/XML (``.xml``) file into FLORA's native
    :class:`Graph` data structure and extracts the information required by
    the FLORA alignment algorithm.

    The :meth:`parse` output is a **single-element list** containing a dict with:

    - ``"entities"``   — ``[{"iri": str, "label": str}]`` for every
      non-literal subject in the graph.
    - ``"predicates"`` — predicate → fact-count mapping.
    - ``"triples"``    — all ``(subject, predicate, object)`` tuples.
    - ``"graph"``      — the loaded :class:`Graph` object forwarded to the aligner.

    Example:
        >>> parser = FLORAOntology()
        >>> kg_data = parser.parse("kg1.ttl")
        >>> print(kg_data[0]["entities"][0])
        {'iri': 'http://example.org/E1', 'label': 'E1'}
        >>> assert isinstance(kg_data[0]["graph"], Graph)
    """

    def load_ontology(self, input_file_path: str) -> Any:
        """
        Load a Turtle or RDF/XML file into a FLORA :class:`Graph`.

        Turtle files are parsed directly; RDF/XML files are first converted
        to Turtle via ``rdflib`` using a temporary file.

        Args:
            input_file_path (str): Path to the ``.ttl`` or ``.xml`` file.

        Returns:
            Graph: The loaded FLORA :class:`Graph`.

        Raises:
            ValueError: If the file extension is neither ``.ttl`` nor ``.xml``.
        """
        # If already TTL → parse directly
        if input_file_path.endswith(".ttl"):
            return parse_turtle_graph(input_file_path)

        # XML → convert to turtle using a temporary file
        elif input_file_path.endswith(".xml"):
            rdf_graph = RDFGraph()
            rdf_graph.parse(input_file_path)
            with tempfile.NamedTemporaryFile(suffix=".ttl", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                rdf_graph.serialize(tmp_path, format="nt", encoding="utf-8")
                graph = parse_turtle_graph(tmp_path)
            finally:
                os.remove(tmp_path)
        else:
            raise ValueError("Unsupported format. Please use 'ttl' or 'xml'.")
        return graph

    def extract_data(self, graph: Any) -> List[Dict[str, Any]]:
        """
        Extract alignment-relevant data from a loaded FLORA :class:`Graph`.

        Args:
            graph (Graph): A fully loaded FLORA :class:`Graph`.

        Returns:
            List[Dict]: Single-element list with keys ``"entities"``,
            ``"predicates"``, ``"triples"``, and ``"graph"``.
        """
        entities = []
        seen = set()
        for subject in graph.subjects():
            if is_literal(subject) or subject in seen or not subject: # adding if subject is none!
                continue
            seen.add(subject)
            # Derive a readable label from the local IRI fragment
            if "#" in subject:
                label = subject.split("#")[-1]
            elif "/" in subject:
                label = subject.split("/")[-1]
            else:
                label = subject
            entities.append({"iri": subject, "label": label})

        return [{
            "entities":   entities,
            "predicates": graph.predicates(),
            "triples":    list(graph),
            "graph":      graph,
        }]

class FLORAOpenEAKnowledgeBase(ABC):
    """
    KG parser for one side of an OpenEA benchmark dataset.

    Reads a single KG's relational triple file and (optionally) its attribute
    triple file into a FLORA :class:`Graph`.  Call this class twice — once for
    each side — via :class:`FLORAOpenEAOMDataset`, which wires the two parsers
    together and calls :meth:`collect`.

    File layout expected (one KG at a time):

    - ``rel_triples_1`` or ``rel_triples_2`` — tab-separated ``(head, relation, tail)`` lines.
    - ``attr_triples_1`` or ``attr_triples_2`` — tab-separated ``(head, attribute, literal)``
      lines (optional; pass ``None`` to skip).

    Ground-truth entity links are loaded separately by
    :class:`OpenEAAlignmentsParser` from the ``ent_links`` file.

    Attributes:
        (none) — this class is stateless; all configuration is passed per call.

    Example:
        >>> from ontoaligner.ontology.flora import FLORAOpenEAKnowledgeBase
        >>> parser = FLORAOpenEAKnowledgeBase()
        >>> kg_data = parser.parse(
        ...     input_file_path="data/D_W_15K_V1/rel_triples_1",
        ...     attribute_path="data/D_W_15K_V1/attr_triples_1",
        ... )
        >>> print(len(kg_data["entities"]), "entities loaded")
    """

    def load_ontology(self, input_file_path: str, attribute_path: str) -> Any:
        """
        Load one KG from an OpenEA relational-triples file.

        Reads ``input_file_path`` as tab-separated ``(head, relation, tail)``
        lines.  If ``attribute_path`` is provided, attribute triples are also
        loaded and literals are normalised (line-break escaping, XSD datatype
        prefix compression).

        Args:
            input_file_path (str): Path to the relational triples file
                (e.g. ``rel_triples_1``).
            attribute_path  (str | None): Path to the attribute triples file
                (e.g. ``attr_triples_1``), or ``None`` to skip attribute loading.

        Returns:
            Graph: A loaded FLORA :class:`Graph` for this KG side.
        """
        kg  = Graph()
        with open(input_file_path,'r', encoding='utf-8') as file:
            for line in file.readlines():
                head, rel, tail = line.strip().split('\t')
                kg.add((head, rel, tail))
        if not attribute_path:
            with open(attribute_path, 'r', encoding='UTF-8') as attribute_file:
                for line in attribute_file.readlines():
                    head, attribute, literal = line.strip().split('\t')
                    if literal.startswith('"'):
                        # Datatypes
                        literal = literal.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '').replace('\\"',
                                                                                                              "'").replace(
                            "\\u0022", "'")
                        if '^^' in literal:
                            str_value, datatype = literal.split('^^')
                            prefixed_datatype = re.sub(r'<http://www\.w3\.org/2001/XMLSchema#([a-zA-Z0-9_-]+)>',
                                                       r'xsd:\1', datatype)
                            literal = '"' + str_value[1:-1] + '"^^' + prefixed_datatype
                            kg.add((head, attribute, literal))
                        else:
                            kg.add((head, attribute, literal))
                    else:
                        # Make all literals simple literals without line breaks and quotes
                        literal = literal.replace('\n', '\\n')\
                                         .replace('\t', '\\t')\
                                         .replace('\r', '')\
                                         .replace('\\"',"'")\
                                         .replace("\\u0022", "'")
                        kg.add((head, attribute, '"' + literal + '"'))
        return kg

    def extract_data(self, g: Any) -> Dict[str, Any]:
        """
        Extract the standard FLORA data dict from a loaded :class:`Graph`.

        Derives a human-readable label for each entity from the local fragment
        of its IRI (the part after ``#`` or the last ``/`` segment).

        Args:
            g (Graph): A fully loaded FLORA :class:`Graph`.

        Returns:
            Dict: A dict with keys ``"entities"``, ``"predicates"``,
            ``"triples"``, and ``"graph"``.
        """
        entities = []
        seen = set()
        for subject in g.subjects():
            if is_literal(subject) or subject in seen:
                continue
            seen.add(subject)
            if "#" in subject:
                label = subject.split("#")[-1]
            elif "/" in subject:
                label = subject.split("/")[-1]
            else:
                label = subject
            entities.append({"iri": subject, "label": label})
        return {
            "entities":   entities,
            "predicates": g.predicates(),
            "triples":    list(g),
            "graph":      g,
        }

    def parse(self, input_file_path: str, attribute_path: str) -> Dict[str, Any]:
        """
        Load and extract one KG from an OpenEA triple file.

        Convenience method that chains :meth:`load_ontology` and
        :meth:`extract_data`.

        Args:
            input_file_path (str): Path to the relational triples file.
            attribute_path  (str | None): Path to the attribute triples file,
                or ``None`` to skip attributes.

        Returns:
            Dict: Standard FLORA data dict with keys ``"entities"``,
            ``"predicates"``, ``"triples"``, and ``"graph"``.
        """
        graph = self.load_ontology(input_file_path, attribute_path)
        return self.extract_data(graph)


class FLORADBpedia15KKnowledgeBase(ABC):
    """
    KG parser for one side of a DBP15K benchmark dataset.

    Reads one KG's ID-mapped files into a FLORA :class:`Graph`.  Call this
    class twice — once for each side — via :class:`FLORADBpediaOMDataset`,
    which wires the two parsers together and calls :meth:`collect`.

    File layout expected (one KG at a time):

    - ``rel_ids_1`` / ``rel_ids_2``   — tab-separated ``(id, relation_iri)`` lines.
    - ``ent_ids_1`` / ``ent_ids_2``   — tab-separated ``(id, entity_iri)`` lines.
    - ``triples_1`` / ``triples_2``   — tab-separated ``(head_id, rel_id, tail_id)`` lines.
    - ``att_triples_1`` / ``att_triples_2`` — Turtle attribute triples (optional).
    - ``translated_google.txt``       — one translated entity name per line,
      aligned with ``ent_ids_1`` (optional, source-side only).

    Ground-truth entity links are loaded separately by
    :class:`DBpediaAlignmentsParser` from the ``ref_ent_ids`` file.

    Attributes:
        name (bool): When ``True`` (default), the local IRI fragment of each
            entity is added as an ``EA:label`` attribute triple.

    Example:
        >>> from ontoaligner.ontology.flora import FLORADBpedia15KKnowledgeBase
        >>> parser = FLORADBpedia15KKnowledgeBase()
        >>> kg_data = parser.parse(
        ...     rel_ids_path="data/dbp_zh_en_15k_v1/rel_ids_1",
        ...     translated_name_path=None,
        ...     ent_ids_path="data/dbp_zh_en_15k_v1/ent_ids_1",
        ...     triples_path="data/dbp_zh_en_15k_v1/triples_1",
        ...     att_triples_path="data/dbp_zh_en_15k_v1/att_triples_1",
        ... )
        >>> print(len(kg_data["entities"]), "entities loaded")
    """
    name = True

    def load_ontology(self,
                      rel_ids_path: str,
                      translated_name_path: str,
                      ent_ids_path: str,
                      triples_path: str,
                      att_triples_path: str) -> Any:
        """
        Load one KG from DBP15K ID-mapped files into a FLORA :class:`Graph`.

        Steps performed:

        1. Reads ``rel_ids_path`` to build an ``id → relation IRI`` map.
        2. Reads ``ent_ids_path`` to build an ``id → entity IRI`` map.
           If ``translated_name_path`` is given *and* this is the source side
           (path ends with ``_1``), translated names from
           ``translated_google.txt`` are used as ``EA:label`` triples instead
           of the raw IRI fragments.
        3. Reads ``triples_path`` and maps integer IDs back to IRIs.
        4. If ``att_triples_path`` is given, parses attribute triples from a
           Turtle-format file and normalises literal values.

        Args:
            rel_ids_path          (str): Path to the relation-ID mapping file.
            translated_name_path  (str | None): Path to ``translated_google.txt``
                for translated entity names, or ``None`` to use raw IRI fragments.
            ent_ids_path          (str): Path to the entity-ID mapping file.
            triples_path          (str): Path to the relational triples file.
            att_triples_path      (str | None): Path to the Turtle attribute
                triples file, or ``None`` to skip attribute loading.

        Returns:
            Graph: A loaded FLORA :class:`Graph` for this KG side.
        """
        kg = Graph()
        id2rel, id2ent = {}, {}
        with open(rel_ids_path, encoding='UTF-8') as rel_ids_file:
            for line in rel_ids_file.readlines():
                ids, rel = line.strip().split('\t')
                id2rel[int(ids)] = rel

        with open(ent_ids_path, encoding='UTF-8') as ent_ids_file:
            if (not translated_name_path) and ent_ids_path.endswith('_1'):
                ent_name_trans = {}
                with open(translated_name_path, encoding='UTF-8') as f2:
                    for line1, line2 in zip(ent_ids_file.readlines(), f2.readlines()):
                        ids, ent = line1.strip().split('\t')
                        ent_trans = line2.strip()
                        id2ent[int(ids)] = ent
                        ent_trans = ent_trans.replace('\n', '\\n')\
                                             .replace('\t', '\\t')\
                                             .replace('\r', '')\
                                             .replace('\\"',"'")\
                                             .replace("\\u0022", "'")
                        ent_name_trans[ent] = ent_trans
                        if self.name:  # add entity name as an attribute triple
                            kg.add((ent, 'EA:label', '"' + ent_trans + '"'))

            for line in ent_ids_file.readlines():
                ids, ent = line.strip().split('\t')
                ent_name = ent.split('/')[-1].replace('_', ' ')
                ent_name = ent_name.replace('\n', '\\n')\
                                    .replace('\t', '\\t')\
                                    .replace('\r', '')\
                                    .replace('\\"',"'")\
                                    .replace("\\u0022", "'")
                id2ent[int(ids)] = ent
                if self.name:  # add entity name as an attribute triple
                    kg.add((ent, 'EA:label', '"' + ent_name + '"'))

        with open(triples_path, encoding='UTF-8') as triples_file:
            for line in triples_file.readlines():
                head, rel, tail = line.strip().split('\t')
                head, rel, tail = int(head), int(rel), int(tail)
                kg.add((id2ent[head], id2rel[rel], id2ent[tail]))

        if not att_triples_path:
            for triple in parse_turtle_triples(att_triples_path):
                head, attribute, literal = triple
                literal = literal.replace('\n','\\n')\
                                 .replace('\t','\\t')\
                                 .replace('\r','')\
                                 .replace('\\"',"'")\
                                 .replace("\\u0022","'")
                if literal.startswith('"'): # Preprocess literals
                    # Datatypes
                    if '^^' in literal:
                        str_value, datatype = literal.split('^^')
                        prefixed_datatype = re.sub(r'<http://www\.w3\.org/2001/XMLSchema#([a-zA-Z0-9_-]+)>',
                                                   r'xsd:\1', datatype)
                        literal = '"'+str_value[1:-1]+'"^^'+prefixed_datatype
                        kg.add((head[1:-1], attribute[1:-1], literal))
                        continue

                    else: # other situation (not '^^')
                        kg.add((head[1:-1], attribute[1:-1], literal))
                else: # literal values lack ""
                    kg.add((head[1:-1], attribute[1:-1], '"'+literal+'"'))
        return kg

    def extract_data(self, g: Any) -> Dict[str, Any]:
        """
        Extract the standard FLORA data dict from a loaded :class:`Graph`.

        Args:
            g (Graph): A fully loaded FLORA :class:`Graph`.

        Returns:
            Dict: A dict with keys ``"entities"``, ``"predicates"``,
            ``"triples"``, and ``"graph"``.
        """
        entities = []
        seen = set()
        for subject in g.subjects():
            if is_literal(subject) or subject in seen:
                continue
            seen.add(subject)
            if "#" in subject:
                label = subject.split("#")[-1]
            elif "/" in subject:
                label = subject.split("/")[-1]
            else:
                label = subject
            entities.append({"iri": subject, "label": label})
        return {
            "entities":   entities,
            "predicates": g.predicates(),
            "triples":    list(g),
            "graph":      g,
        }

    def parse(self,
              rel_ids_path: str,
              translated_name_path: str,
              ent_ids_path: str,
              triples_path: str,
              att_triples_path: str) -> Dict[str, Any]:
        """
        Load and extract one KG from DBP15K ID-mapped files.

        Convenience method that chains :meth:`load_ontology` and
        :meth:`extract_data`.

        Args:
            rel_ids_path         (str): Path to the relation-ID mapping file.
            translated_name_path (str | None): Path to translated entity names,
                or ``None`` to use raw IRI fragments.
            ent_ids_path         (str): Path to the entity-ID mapping file.
            triples_path         (str): Path to the relational triples file.
            att_triples_path     (str | None): Path to the Turtle attribute
                triples file, or ``None`` to skip.

        Returns:
            Dict: Standard FLORA data dict with keys ``"entities"``,
            ``"predicates"``, ``"triples"``, and ``"graph"``.
        """
        graphs = self.load_ontology(rel_ids_path, translated_name_path, ent_ids_path, triples_path, att_triples_path)
        return self.extract_data(graphs)


class OpenEAAlignmentsParser(BaseAlignmentsParser):
    """
    Alignment parser for OpenEA ``ent_links`` TSV files.

    The ``ent_links`` file contains tab-separated entity-pair lines::

        http://kg1.org/E1\\thttp://kg2.org/E2
        ...

    Each line is converted to ``{"source": ..., "target": ..., "relation": "="}``
    in the output list.

    Example:
        >>> parser = OpenEAAlignmentsParser()
        >>> refs = parser.parse("data/D_W_15K_V1/ent_links")
        >>> print(refs[0])
        {'source': 'http://www.dbpedia.org/...', 'target': 'http://www.wikidata.org/...', 'relation': '='}
    """

    def load_ontology(self, input_file_path: str) -> Any:
        """
        Read the ``ent_links`` TSV file into a list of entity pairs.

        Args:
            input_file_path (str): Direct path to the ``ent_links`` TSV file.

        Returns:
            List[Tuple[str, str]]: Raw list of ``(entity1_iri, entity2_iri)`` pairs.
        """
        pairs = []
        with open(input_file_path, "r", encoding="utf-8") as file:
            for line in file:
                head, tail = line.strip().split('\t')
                pairs.append((head, tail))
        return pairs

    def extract_data(self, reference: Any) -> List[Dict]:
        """
        Convert raw entity pairs to OntoAligner alignment dicts.

        Args:
            reference (List[Tuple[str, str]]): List of ``(entity1, entity2)`` pairs.

        Returns:
            List[Dict]: Each dict has keys ``"source"``, ``"target"``, and
            ``"relation"`` (always ``"="``).
        """
        return [{"source": e1, "target": e2, "relation": "="} for e1, e2 in reference]


class DBpediaAlignmentsParser(BaseAlignmentsParser):
    """
    Alignment parser for DBP15K ``ref_ent_ids`` / ``sup_pairs`` TSV files.

    The ``ref_ent_ids`` file (and ``sup_pairs``) contain tab-separated
    entity-ID pairs::

        <entity_id_kg1>\\t<entity_id_kg2>
        ...

    IDs are mapped back to IRIs using ``ent_ids_1`` / ``ent_ids_2`` files
    found in the same directory.  If those mapping files are absent the raw
    values are treated as IRIs directly.

    Adopted from ``eval.load_dbp15k_ref``:  the supervised 30 % seed pairs
    (``sup_pairs``) and the test 70 % pairs (``ref_pairs``) are now both
    accessible via :meth:`parse_splits`.

    Example:
        >>> parser = DBpediaAlignmentsParser()
        >>> seed, ref = parser.parse_splits("data/dbp_zh_en_15k_v1/ref_ent_ids")
        >>> print(ref[0])
        {'source': 'http://zh.dbpedia.org/...', 'target': 'http://en.dbpedia.org/...', 'relation': '='}
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_id_maps(self, dir_path: str) -> Tuple[Dict[int, str], Dict[int, str]]:
        """Build ``{id → IRI}`` maps from ``ent_ids_1`` / ``ent_ids_2``."""
        id2ent1: Dict[int, str] = {}
        id2ent2: Dict[int, str] = {}
        for i, id2ent in enumerate([id2ent1, id2ent2], start=1):
            ent_ids_path = os.path.join(dir_path, f"ent_ids_{i}")
            if os.path.exists(ent_ids_path):
                with open(ent_ids_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        idx, iri = line.split("\t")
                        id2ent[int(idx)] = iri
        return id2ent1, id2ent2

    def _read_pair_file(
            self,
            file_path: str,
            id2ent1: Dict[int, str],
            id2ent2: Dict[int, str],
    ) -> List[Tuple[str, str]]:
        """Read a tab-separated pair file, resolving IDs to IRIs."""
        pairs: List[Tuple[str, str]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                e1_raw, e2_raw = parts[0], parts[1]
                if id2ent1:
                    e1 = id2ent1.get(int(e1_raw), e1_raw)
                    e2 = id2ent2.get(int(e2_raw), e2_raw)
                else:
                    e1, e2 = e1_raw, e2_raw
                pairs.append((e1, e2))
        return pairs

    # ------------------------------------------------------------------
    # BaseAlignmentsParser interface  (used by the default .parse() path)
    # ------------------------------------------------------------------

    def load_ontology(self, input_file_path: str) -> Any:
        """
        Read and resolve the ``ref_ent_ids`` file (test pairs only).

        Args:
            input_file_path: Path to ``ref_ent_ids``.

        Returns:
            List[Tuple[str, str]]: ``(iri1, iri2)`` pairs.
        """
        dir_path = os.path.dirname(input_file_path)
        id2ent1, id2ent2 = self._build_id_maps(dir_path)
        return self._read_pair_file(input_file_path, id2ent1, id2ent2)

    def extract_data(self, reference: Any) -> List[Dict]:
        """Convert raw ``(iri1, iri2)`` pairs to OntoAligner alignment dicts."""
        return [{"source": e1, "target": e2, "relation": "="} for e1, e2 in reference]

    def parse(self, input_file_path: str = "") -> List:
        """
        Parse the DBP15K test-pairs file (``ref_ent_ids``).

        Args:
            input_file_path: Path to the ``ref_ent_ids`` TSV file.

        Returns:
            List[Dict]: Parsed alignment pairs, or ``[]`` on error / empty path.
        """
        if not input_file_path:
            return []
        try:
            pairs = self.load_ontology(input_file_path)
            return self.extract_data(pairs)
        except Exception:
            return []

    # ------------------------------------------------------------------
    # CHANGE 1 — new method: load_sup_pairs
    # ------------------------------------------------------------------

    def load_sup_pairs(self, ref_ent_ids_path: str) -> List[Tuple[str, str]]:
        """Load the 30 % supervised seed pairs from the ``sup_pairs`` file.

        Adopted from ``eval.load_dbp15k_ref``: the ``sup_pairs`` file lives
        in the same directory as ``ref_ent_ids``.

        Args:
            ref_ent_ids_path: Path to ``ref_ent_ids`` (used to locate the
                directory and ``ent_ids_*`` mapping files).

        Returns:
            List[Tuple[str, str]]: ``(iri1, iri2)`` seed pairs, or ``[]`` if
            the file is absent.
        """
        dir_path = os.path.dirname(ref_ent_ids_path)
        sup_path = os.path.join(dir_path, "sup_pairs")
        if not os.path.exists(sup_path):
            return []
        id2ent1, id2ent2 = self._build_id_maps(dir_path)
        return self._read_pair_file(sup_path, id2ent1, id2ent2)

    # ------------------------------------------------------------------
    # CHANGE 2 — new method: parse_splits
    # ------------------------------------------------------------------

    def parse_splits(
            self, input_file_path: str = ""
    ) -> Tuple[List[Dict], List[Dict]]:
        """Parse both the seed (30 %) and test (70 %) pair files.

        Mirrors ``eval.load_dbp15k_ref``.

        Args:
            input_file_path: Path to the ``ref_ent_ids`` test-pairs file.

        Returns:
            Tuple ``(seed_pairs, ref_pairs)`` where each element is a list of
            alignment dicts with keys ``source``, ``target``, ``relation``.
        """
        ref_pairs = self.parse(input_file_path)
        raw_seed = self.load_sup_pairs(input_file_path)
        seed_pairs = self.extract_data(raw_seed)
        return seed_pairs, ref_pairs


class FLORAAlignmentsParser(BaseAlignmentsParser):

    def load_ontology(self, input_file_path: str) -> Any:
        tree = ET.parse(input_file_path)
        root = tree.getroot()
        ns = {
            'ns': 'http://knowledgeweb.semanticweb.org/heterogeneity/alignment',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        }
        cells = root.findall('.//ns:map/ns:Cell', ns)
        pairs = []
        for cell in cells:
            e1 = cell.find('ns:entity1', ns).attrib['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource']
            e2 = cell.find('ns:entity2', ns).attrib['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource']

            if '/class/' in e1:
                pairs.append((e1, e2, 'class'))
            elif '/property/' in e1:
                pairs.append((e1, e2, 'property'))
            elif '/resource/' in e1:
                pairs.append((e1, e2, 'instance'))
            else:
                raise ValueError('Unknown type of entity: {}'.format(e1))
        return pairs

    def extract_data(self, reference: Any) -> List[Dict]:
        return [{"source": e1, "target": e2, "type": type} for e1, e2, type in reference]


class FLORAOMDataset(OMDataset):
    """
    Dataset class for FLORA-based OAEI / Turtle-file alignment tasks.

    Ties together two :class:`FLORAOntology` parsers so users follow the
    standard OntoAligner pipeline:

    .. code-block:: python

        task    = FLORAOMDataset()
        dataset = task.collect(
            source_ontology_path="kg1.ttl",
            target_ontology_path="kg2.ttl",
            reference_matching_path="reference.xml",   # optional
        )
        # dataset["source"][0]["graph"]    -> loaded FLORA Graph for KG1
        # dataset["source"][0]["entities"] -> [{"iri": ..., "label": ...}, ...]
        # dataset["target"][0]["graph"]    -> loaded FLORA Graph for KG2

    Attributes:
        track (str): ``"KG Alignment - FLORA"``
        ontology_name (str): Human-readable dataset name; override as needed.
        source_ontology (FLORAOntology): Parser for the source KG.
        target_ontology (FLORAOntology): Parser for the target KG.
    """

    track: str = track
    ontology_name: str = "Source-Target"
    source_ontology: FLORAOntology = FLORAOntology()
    target_ontology: FLORAOntology = FLORAOntology()
    alignments: FLORAAlignmentsParser = FLORAAlignmentsParser()


class FLORAOpenEAOMDataset(OMDataset):
    """
    Dataset class for FLORA on OpenEA benchmarks.

    Wires two :class:`FLORAOpenEAKnowledgeBase` parsers (one per KG side) and
    an :class:`OpenEAAlignmentsParser` for the ground-truth links.  The custom
    :meth:`collect` method accepts explicit per-side file paths rather than a
    single shared directory.

    .. code-block:: python

        task    = FLORAOpenEAOMDataset()
        dataset = task.collect(
            source_kg_path="data/D_W_15K_V1/rel_triples_1",
            target_kg_path="data/D_W_15K_V1/rel_triples_2",
            source_kg_attribute_path="data/D_W_15K_V1/attr_triples_1",
            target_kg_attribute_path="data/D_W_15K_V1/attr_triples_2",
            reference_matching_path="data/D_W_15K_V1/ent_links",
        )
        # dataset["source"][0]["graph"]  -> FLORA Graph for KG1
        # dataset["target"][0]["graph"]  -> FLORA Graph for KG2
        # dataset["reference"]           -> list of {"source", "target", "relation"} dicts

    Attributes:
        track (str): ``"KG Alignment - FLORA"``
        ontology_name (str): Human-readable dataset name; override as needed.
        source_ontology (FLORAOpenEAKnowledgeBase): Parser for the source KG.
        target_ontology (FLORAOpenEAKnowledgeBase): Parser for the target KG.
        alignments (OpenEAAlignmentsParser): Ground-truth parser.
    """

    track: str = track
    ontology_name: str = "OpenEA"
    source_ontology: FLORAOpenEAKnowledgeBase = FLORAOpenEAKnowledgeBase()
    target_ontology: FLORAOpenEAKnowledgeBase = FLORAOpenEAKnowledgeBase()
    alignments: OpenEAAlignmentsParser = OpenEAAlignmentsParser()

    @override
    def collect(self,
                source_kg_path: str,
                target_kg_path: str,
                source_kg_attribute_path: str = None,
                target_kg_attribute_path: str = None,
                reference_matching_path: str = "") -> Dict:
        """
        Load both KGs and the ground-truth alignment for an OpenEA benchmark.

        Args:
            source_kg_path           (str): Path to the source relational triples
                file (e.g. ``rel_triples_1``).
            target_kg_path           (str): Path to the target relational triples
                file (e.g. ``rel_triples_2``).
            source_kg_attribute_path (str | None): Path to the source attribute
                triples file (e.g. ``attr_triples_1``), or ``None`` to skip.
            target_kg_attribute_path (str | None): Path to the target attribute
                triples file (e.g. ``attr_triples_2``), or ``None`` to skip.
            reference_matching_path  (str): Path to the ``ent_links`` ground-truth
                file.  Pass an empty string to skip reference loading.

        Returns:
            Dict: ``{"dataset-info": ..., "source": [kg1_data],
            "target": [kg2_data], "reference": [...]}``
        """
        source_kg = self.source_ontology.parse(source_kg_path, source_kg_attribute_path)
        target_kg = self.target_ontology.parse(target_kg_path, target_kg_attribute_path)
        return {
            "dataset-info": {"track": self.track, "ontology-name": self.ontology_name},
            "source":       source_kg,
            "target":       target_kg,
            "reference":    self.alignments.parse(input_file_path=reference_matching_path),
        }


class FLORADBpediaOMDataset(OMDataset):
    """
    Dataset class for FLORA on DBP15K benchmarks.

    Wires two :class:`FLORADBpedia15KKnowledgeBase` parsers (one per KG side)
    and a :class:`DBpediaAlignmentsParser` for the ground-truth links.  The
    custom :meth:`collect` method accepts explicit per-side file paths for all
    DBP15K input files.

    .. code-block:: python

        task    = FLORADBpediaOMDataset()
        dataset = task.collect(
            source_kg_rel_ids_path="data/dbp_zh_en_15k_v1/rel_ids_1",
            target_kg_rel_ids_path="data/dbp_zh_en_15k_v1/rel_ids_2",
            source_kg_ent_ids_path="data/dbp_zh_en_15k_v1/ent_ids_1",
            target_kg_ent_ids_path="data/dbp_zh_en_15k_v1/ent_ids_2",
            source_kg_triples_path="data/dbp_zh_en_15k_v1/triples_1",
            target_kg_triples_path="data/dbp_zh_en_15k_v1/triples_2",
            source_kg_att_triples_path="data/dbp_zh_en_15k_v1/att_triples_1",
            target_kg_att_triples_path="data/dbp_zh_en_15k_v1/att_triples_2",
            reference_matching_path="data/dbp_zh_en_15k_v1/ref_ent_ids",
        )
        # dataset["source"][0]["graph"]  -> FLORA Graph for KG1
        # dataset["target"][0]["graph"]  -> FLORA Graph for KG2
        # dataset["reference"]           -> list of {"source", "target", "relation"} dicts

    Attributes:
        track (str): ``"KG Alignment - FLORA"``
        ontology_name (str): Human-readable dataset name; override as needed.
        source_ontology (FLORADBpedia15KKnowledgeBase): Parser for the source KG.
        target_ontology (FLORADBpedia15KKnowledgeBase): Parser for the target KG.
        alignments (DBpediaAlignmentsParser): Ground-truth parser.
    """

    track: str = track
    ontology_name: str = "DBP15K"
    source_ontology: FLORADBpedia15KKnowledgeBase = FLORADBpedia15KKnowledgeBase()
    target_ontology: FLORADBpedia15KKnowledgeBase = FLORADBpedia15KKnowledgeBase()
    alignments: DBpediaAlignmentsParser = DBpediaAlignmentsParser()

    @override
    def collect(
            self,
            source_kg_rel_ids_path: str,
            target_kg_rel_ids_path: str,
            source_translated_name_path: Optional[str] = None,
            target_translated_name_path: Optional[str] = None,
            source_kg_ent_ids_path: Optional[str] = None,
            target_kg_ent_ids_path: Optional[str] = None,
            source_kg_triples_path: Optional[str] = None,
            target_kg_triples_path: Optional[str] = None,
            source_kg_att_triples_path: Optional[str] = None,
            target_kg_att_triples_path: Optional[str] = None,
            reference_matching_path: str = ""
    ) -> Dict:
        """
        Load both KGs and both reference splits for a DBP15K benchmark.

        Returns:
            Dict with keys:
                - ``"dataset-info"``  : track / dataset metadata
                - ``"source"``        : [kg1_data]
                - ``"target"``        : [kg2_data]
                - ``"reference"``     : 70 % test pairs (ref_ent_ids)
                - ``"seed_reference"``: 30 % supervised seed pairs (sup_pairs)
                  — pass these to FLORAAligner via the training_data argument
                  or as pre-loaded seed alignments.
        """
        source_kg = self.source_ontology.parse(
            rel_ids_path=source_kg_rel_ids_path,
            translated_name_path=source_translated_name_path,
            ent_ids_path=source_kg_ent_ids_path,
            triples_path=source_kg_triples_path,
            att_triples_path=source_kg_att_triples_path,
        )
        target_kg = self.target_ontology.parse(
            rel_ids_path=target_kg_rel_ids_path,
            translated_name_path=target_translated_name_path,
            ent_ids_path=target_kg_ent_ids_path,
            triples_path=target_kg_triples_path,
            att_triples_path=target_kg_att_triples_path,
        )

        # CHANGE 3: use parse_splits() instead of plain parse() so we get
        # both the supervised seed pairs and the held-out test pairs.
        seed_pairs, ref_pairs = self.alignments.parse_splits(
            input_file_path=reference_matching_path
        )

        return {
            "dataset-info": {"track": self.track, "ontology-name": self.ontology_name},
            "source": [source_kg],
            "target": [target_kg],
            "reference": ref_pairs,  # 70 % test pairs → use for evaluate_dbp15k()
            "seed_reference": seed_pairs,  # 30 % seed pairs → use as training_data
        }
