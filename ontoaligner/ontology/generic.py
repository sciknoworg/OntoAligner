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
from rdflib import Graph, URIRef, RDFS, SKOS, BNode, Literal
from rdflib.namespace import OWL, RDF, DC, DCTERMS, XSD
from typing import Any, Dict, List, Optional, Set, Iterable
from urllib.parse import unquote_plus, urlparse
from tqdm import tqdm
import re

from ..base import BaseOntologyParser, OMDataset

track = "Generic"

class GenericOntology(BaseOntologyParser):
    """
    An abstract base class for parsing OWL ontologies. This class defines methods to extract data such as
    names, labels, IRIs, children, parents, synonyms, and comments for ontology classes.
    """

    def __init__(self, language: str = 'en'):
        self.language = language

    def is_valid_label(self, label: str) -> Any:
        invalids = ['root', 'thing']
        if label.lower() in invalids:
            return None
        return label

    def get_label(self, owl_class: str) -> Any:
        """
        Extracts the label for a given URI in the specified language from the RDF graph.
        If no valid label is found, returns None.

        :param uri: URI of the entity to retrieve the label for.
        :return: The label in the specified language, or None if no label found.
        """
        entity = URIRef(owl_class)
        labels = list(self.graph.objects(subject=entity, predicate=RDFS.label))
        for label in labels:
            if hasattr(label, 'language') and label.language == self.language:
                return self.is_valid_label(str(label))
        if labels:
            first_label = str(labels[0])
            if not first_label.startswith("http"):
                return self.is_valid_label(first_label)
        if "#" in owl_class:
            local_name = owl_class.split("#")[-1]
        elif "/" in owl_class:
            local_name = owl_class.split("/")[-1]
        else:
            local_name = owl_class
        label = self.is_valid_label(local_name)
        if not label:
            return None
        return label

    def load_ontology(self, input_file_path: str) -> Any:
        """
        Loads an ontology from the specified file path using rdflib.
        """
        graph = Graph()
        graph.parse(input_file_path, format="xml")
        return graph

    def is_class(self, owl_class: URIRef) -> bool:
        """
        Checks if the given entity is an actual class (not a Restriction or property).
        """
        return (self.graph.value(owl_class, RDFS.subClassOf) is not None)

    def get_synonyms(self, owl_class: URIRef) -> List[Dict[str, str]]:
        """
        Retrieves synonyms for the given ontology class.

        Tries different properties (like `rdfs:label`, `skos:altLabel`).
        """
        synonyms = []
        # Search for alternative labels (SKOS altLabel)
        for alt_label in self.graph.objects(owl_class, SKOS.altLabel):
            synonyms.append({
                "iri": str(owl_class),
                "label": str(alt_label),
                "name": str(alt_label),
            })
        return synonyms

    def get_name(self, owl_class: URIRef) -> str:
        """
        Retrieves the name (IRI) of the class.
        """
        return self.get_label(owl_class)
        # return str(label) if label else None

    def get_iri(self, owl_class: URIRef) -> str:
        """
        Returns the IRI of the class.
        """
        return str(owl_class)

    def get_comments(self, owl_class: URIRef) -> List[str]:
        """
        Retrieves comments for the given class.
        """
        comments = []
        for comment in self.graph.objects(owl_class, RDFS.comment):
            comments.append(str(comment))
        return comments

    def get_parents(self, owl_class: URIRef) -> List[Dict[str, str]]:
        """
        Retrieves parent classes for the given ontology class.
        """
        parents = []
        for parent in self.graph.objects(owl_class, RDFS.subClassOf):
            if self.is_class(parent):
                name = self.get_name(parent)
                label = self.get_label(parent)
                iri = self.get_iri(parent)
                if label and iri:
                    parents.append({"iri": iri, "label": label, "name": name})
        return parents

    def get_childrens(self, owl_class: URIRef) -> List[Dict[str, str]]:
        """
        Retrieves child classes for the given ontology class.
        """
        childrens = []
        for child in self.graph.subjects(RDFS.subClassOf, owl_class):
            if self.is_class(child):
                name = self.get_name(child)
                label = self.get_label(child)
                iri = self.get_iri(child)
                if label and iri:
                    childrens.append({"iri": iri, "label": label, "name": name})
        return childrens

    def get_class_info(self, owl_class: URIRef) -> Any:
        """
        Collects all relevant information for a given ontology class.
        """
        label = self.get_label(owl_class)
        name = self.get_name(owl_class)
        iri = self.get_iri(owl_class)

        if not label or not iri:
            return None

        class_info = {
            "name": name,
            "iri": iri,
            "label": label,
            "childrens": self.get_childrens(owl_class),
            "parents": self.get_parents(owl_class),
            "synonyms": self.get_synonyms(owl_class),
            "comment": self.get_comments(owl_class)
        }
        return class_info

    def extract_data(self, graph: Any) -> List[Dict[str, Any]]:
        """
        Extracts and processes data from all classes in the ontology.
        """
        self.graph = graph
        parsed_ontology = []

        # Iterate over all classes explicitly defined as owl:Class
        for owl_class in tqdm(self.graph.subjects(RDF.type, OWL.Class)):
            if isinstance(owl_class, BNode):  # Skip blank nodes
                continue

            class_info = self.get_class_info(owl_class)
            if class_info:
                parsed_ontology.append(class_info)
        return parsed_ontology


class GenericOMDataset(OMDataset):
    """
    A dataset class for working with the Source-Target ontology.
    """
    track = track
    ontology_name = "Source-Target"
    source_ontology = GenericOntology()
    target_ontology = GenericOntology()


class OLaLaOntology(GenericOntology):
    """
    A parser for preserving OLaLa TextExtractorSet fields.
    """
    def __init__(self, language: str = "en"):
        super().__init__(language=language)
        self._annotation_properties = None
        self._annotation_properties_graph = None
        self.LABEL_NAMES = {"label", "preflabel", "altlabel", "hiddenlabel",
                            "name", "alternatename", "additionalname"}
        self.DESCRIPTION_NAMES = {"comment", "description", "definition", "abstract"}
        self.LABEL_LIKE_PROPERTIES = {
            str(RDFS.label),
            str(SKOS.prefLabel),
            str(SKOS.altLabel),
            str(SKOS.hiddenLabel),
            "http://schema.org/name",
            "https://schema.org/name",
            "http://schema.org/alternateName",
            "https://schema.org/alternateName",
            "http://schema.org/additionalName",
            "https://schema.org/additionalName",
        }

        self.DESCRIPTION_LIKE_PROPERTIES = {
            str(RDFS.comment),
            str(DC.description),
            str(DCTERMS.description),
            "http://schema.org/comment",
            "https://schema.org/comment",
            "http://schema.org/description",
            "https://schema.org/description",
            "http://dbpedia.org/ontology/description",
            "http://dbpedia.org/ontology/abstract",
            "http://purl.org/dc/terms/abstract",
        }


    def get_annotation_properties(self) -> Set[URIRef]:
        """
        Retrieves ontology annotation properties.

        Returns:
            Set[URIRef]: The declared annotation properties.
        """
        if (
            self._annotation_properties is not None
            and self._annotation_properties_graph is self.graph
        ):
            return self._annotation_properties

        properties = set()

        for predicate in self.graph.subjects(RDF.type, OWL.AnnotationProperty):
            if isinstance(predicate, URIRef):
                properties.add(predicate)

        self._annotation_properties = properties
        self._annotation_properties_graph = self.graph

        return self._annotation_properties

    def get_literal_lexical_form(self, literal: Literal) -> str:
        """
        Returns the lexical form of a literal.

        Parameters:
            literal (Literal): The RDF literal.

        Returns:
            str: The lexical form.
        """
        return str(literal)

    def get_uri_fragment(self, uri: str) -> str:
        """
        Returns the fragment after the last hashtag or slash.

        Parameters:
            uri (str): The URI.

        Returns:
            str: The URI fragment.
        """
        last_index = uri.rfind("#")
        if last_index >= 0:
            return uri[last_index + 1:]

        last_index = uri.rfind("/")
        if last_index >= 0:
            return uri[last_index + 1:]

        return uri

    def get_label_or_fragment_without_language_annotation(
        self,
        resource: URIRef,
    ) -> Optional[str]:
        """
        Returns the first rdfs:label or the URI fragment.

        Parameters:
            resource (URIRef): The resource.

        Returns:
            Optional[str]: The label or URI fragment.
        """
        for node in self.graph.objects(resource, RDFS.label):
            if isinstance(node, Literal):
                return self.get_literal_lexical_form(node)

        uri = str(resource)
        if uri is None:
            return None

        return self.get_uri_fragment(uri)

    def get_annotation_properties_recursion_deadlock_safe(
        self,
        resource: URIRef,
        annotation_properties: Set[URIRef],
        recursion_depth: int,
    ) -> List[str]:
        """
        Recursively extracts annotation property literals.

        Parameters:
            resource (URIRef): The resource to process.
            annotation_properties (Set[URIRef]): The annotation properties.
            recursion_depth (int): The current recursion depth.

        Returns:
            List[str]: The extracted literal lexical forms.
        """
        result = []

        if isinstance(resource, BNode):
            return result

        recursion_depth += 1

        for annotation_property in annotation_properties:
            for node in self.graph.objects(resource, annotation_property):
                if isinstance(node, (URIRef, BNode)):
                    if recursion_depth < 10:
                        result.extend(
                            self.get_annotation_properties_recursion_deadlock_safe(
                                node,
                                annotation_properties,
                                recursion_depth,
                            )
                        )
                    else:
                        return result
                elif isinstance(node, Literal):
                    result.append(self.get_literal_lexical_form(node))

        label = self.get_label_or_fragment_without_language_annotation(resource)
        if label is not None:
            result.append(label)

        return result

    def extract_annotation_property_texts(self, resource: URIRef) -> List[str]:
        """
        Extracts texts from annotation properties.

        Parameters:
            resource (URIRef): The resource.

        Returns:
            List[str]: The extracted annotation texts.
        """
        if resource is None or isinstance(resource, BNode):
            return []

        annotation_properties = self.get_annotation_properties()
        texts = self.get_annotation_properties_recursion_deadlock_safe(
            resource,
            annotation_properties,
            0,
        )

        return list({text for text in texts if text.strip() != ""})

    def is_literal_a_string(self, literal: Any) -> bool:
        """
        Checks whether a literal is accepted by TextExtractorAllStringLiterals.

        Parameters:
            literal (Any): The RDF literal.

        Returns:
            bool: True if the literal is a string literal.
        """
        if not isinstance(literal, Literal):
            return False
        datatype = literal.datatype
        if datatype is None:
            return True
        if str(datatype) == str(XSD.string):
            return True
        if str(datatype) == str(RDF.langString):
            return True
        language = literal.language
        if language is not None and language != "":
            return True
        return False

    def is_olala_text_property(self, predicate: URIRef) -> bool:
        """
        Checks whether a predicate should be used by OLaLa TextExtractorSet.

        Parameters:
            predicate (URIRef): The RDF predicate.

        Returns:
            bool: True if the predicate is label-like or description-like.
        """

        def has_property_label_fragment(predicate: URIRef) -> bool:
            """
            Checks whether the predicate fragment is label-like.
            """
            fragment = self.get_uri_fragment(str(predicate)).lower()
            return fragment in self.LABEL_NAMES

        def has_property_description_fragment(predicate: URIRef) -> bool:
            """
            Checks whether the predicate fragment is description-like.
            """
            fragment = self.get_uri_fragment(str(predicate)).lower()
            return fragment in self.DESCRIPTION_NAMES

        predicate_uri = str(predicate)

        return (
                predicate_uri in self.LABEL_LIKE_PROPERTIES
                or predicate_uri in self.DESCRIPTION_LIKE_PROPERTIES
                or has_property_label_fragment(predicate)
                or has_property_description_fragment(predicate)
        )

    def contains_mostly_numbers(self, term: str) -> bool:
        """
        Checks whether a term contains more than 50 percent numbers.

        Parameters:
            term (str): The term to check.

        Returns:
            bool: True if the term contains mostly numbers.
        """
        numbers = 0
        all_non_whitespace = 0
        for char in term:
            if "0" <= char <= "9":
                numbers += 1
            if not char.isspace():
                all_non_whitespace += 1
        if all_non_whitespace == 0:
            return False
        return numbers > all_non_whitespace / 2

    def _replace_camel_case(self, text: str, replacement: str) -> str:
        """
        Applies the MELT camel-case boundary rule.

        Parameters:
            text (str): The text to process.
            replacement (str): The boundary replacement.

        Returns:
            str: The processed text.
        """
        result = []

        for index, char in enumerate(text):
            if (
                    index > 0
                    and index + 1 < len(text)
                    and char.isupper()
                    and text[index + 1].islower()
                    and not text[index - 1].isspace()
                    and text[index - 1] != '"'
            ):
                result.append(replacement)
            result.append(char)

        return "".join(result)

    def normalize(self, text: str) -> str:
        """
        Normalizes text following MELT StringProcessing.normalize.

        Parameters:
            text (str): The text to normalize.

        Returns:
            List[str]: The normalized token list.
        """
        if text is None:
            return ""
        value = str(text).strip()
        value = self._replace_camel_case(value, "_")
        value = value.replace(" ", "_")
        value = value.lower()

        try:
            value = unquote_plus(value)
        except Exception:
            pass

        value = re.sub(r"[^a-zA-Z\d\s:_]", "_", value)
        value = re.sub(r"'s", "", value)
        value = re.sub(r"_+", "_", value)

        # Splits on underscores following Java String.split behavior.
        if value == "":
            return ""
        tokens = value.split("_")
        while tokens and tokens[-1] == "":
            tokens.pop()
        return " ".join(tokens)

    def get_host_of_uri(self, uri: str) -> str:
        """
        Extracts the host of a URI.
        """
        if uri is None or uri == "":
            return ""
        try:
            host = urlparse(str(uri)).hostname
            if host is None:
                return ""
            return host
        except ValueError:
            return ""

    def normalize_only_camel_case_and_underscore(self, text: str) -> str:
        """
        Normalizes only camel case and underscores.

        Parameters:
            text (str): The text to normalize.

        Returns:
            str: The normalized text.
        """
        if text is None:
            return ""
        value = str(text).strip()
        value = self._replace_camel_case(value, " ")
        value = value.replace("_", " ")
        value = re.sub(r" +", " ", value)
        return value

    def extract_text_extractor_set(self, resource: URIRef) -> Dict[str, Any]:
        """
        Extracts texts following MELT TextExtractorSet.

        Parameters:
            resource (URIRef): The resource.

        Returns:
            Dict[str, Any]: The extracted text fields.
        """
        def deduplicate_normalized_literals(texts: Iterable[str]) -> List[str]:
            """
            Deduplicates literals using MELT NormalizedLiteral behavior.
            """
            normalized_to_lexical: Dict[str, str] = {}
            for text in texts:
                lexical = str(text).strip()
                if not lexical:
                    continue
                normalized = self.normalize(lexical)
                if not normalized:
                    continue
                if normalized not in normalized_to_lexical:
                    normalized_to_lexical[normalized] = lexical

            return list(normalized_to_lexical.values())

        text_candidates = []
        direct_string_literals = []
        direct_text_literals = []
        longest_literal = ""

        for predicate, node in self.graph.predicate_objects(subject=resource):
            if not isinstance(node, Literal):
                continue

            if not self.is_literal_a_string(node):
                continue

            text = self.get_literal_lexical_form(node).strip()
            if not text:
                continue

            direct_string_literals.append({
                "predicate": str(predicate),
                "text": text,
                "language": node.language or "",
                "normalized_text": self.normalize(text),

            })

            if self.is_olala_text_property(predicate):
                text_candidates.append(text)
                direct_text_literals.append({
                    "predicate": str(predicate),
                    "text": text,
                    "language": node.language or ""
                })

            if len(text) > len(longest_literal):
                longest_literal = text

        if longest_literal != "":
            text_candidates.append(longest_literal)

        uri_fragment = self.get_uri_fragment(str(resource)).strip()
        is_uri_fragment_normalization_valid = False
        if uri_fragment and not self.contains_mostly_numbers(uri_fragment):
            text_candidates.append(uri_fragment)
            is_uri_fragment_normalization_valid = True


        annotation_texts = self.extract_annotation_property_texts(resource)
        text_candidates.extend(annotation_texts)

        texts = deduplicate_normalized_literals(text_candidates)
        normalized_texts = [self.normalize_only_camel_case_and_underscore(text) for text in texts]
        return {
            "texts": texts,
            "normalized_texts": normalized_texts,
            "uri_fragment": uri_fragment,
            "is_uri_fragment_normalization_valid": is_uri_fragment_normalization_valid,
            "longest_literal": longest_literal,
            "annotation_texts": annotation_texts,
            "direct_string_literals": direct_string_literals,
            "direct_text_literals": direct_text_literals,
        }

    def lang_tag_match(self, target: str) -> bool:
        """
        Checks whether a language tag matches the configured language.

        Parameters:
            target (str): The language tag.

        Returns:
            bool: True if the language tag matches.
        """
        return (
            self.language.lower() == target.lower()
            or (
                len(target) > len(self.language)
                and self.language.lower() == target[: len(self.language)].lower()
            )
        )

    def extract_only_label(self, owl: Dict) -> str:
        """
        Extracts one label following MELT TextExtractorOnlyLabel.

        Parameters:
            owl (Dict): A parsed ontology item.

        Returns:
            str: The selected normalized label.
        """
        def extract_property(owl: Dict, property_uri: str) -> str:
            """
            Extracts a literal for a property using TextExtractorOnlyLabel rules.
            """
            fallback = []
            for record in owl.get("direct_string_literals", []):
                if record.get("predicate") != property_uri:
                    continue
                lexical = record.get("text", "").strip()
                if not lexical:
                    continue
                language = record.get("language", "")
                if self.lang_tag_match(language):
                    return lexical
                if language == "":
                    fallback.append(lexical)
            if fallback:
                return fallback[0]
            return ""

        def extract_fragment(owl: Dict) -> str:
            """
            Extracts the URI fragment if it is not mostly numeric.
            """
            fragment = str(owl.get("uri_fragment", "")).strip()
            if not fragment:
                fragment = self.get_uri_fragment(owl.get("iri", "")).strip()
            if not fragment:
                return ""
            if self.contains_mostly_numbers(fragment):
                return ""
            return fragment

        property_order = [
            str(SKOS.prefLabel),
            str(RDFS.label),
        ]
        for property_uri in property_order:
            value = extract_property(owl=owl, property_uri=property_uri)
            if value:
                return self.normalize_only_camel_case_and_underscore(value)
        value = extract_fragment(owl)
        if value:
            return self.normalize_only_camel_case_and_underscore(value)
        property_order = [
            str(SKOS.altLabel),
            str(SKOS.hiddenLabel),
        ]
        for property_uri in property_order:
            value = extract_property(owl=owl, property_uri=property_uri)
            if value:
                return self.normalize_only_camel_case_and_underscore(value)
        return ""

    def get_class_info(self, owl_class: URIRef) -> Any:
        """
        Collects generic and OLaLa-specific information for a class.

        Parameters:
            owl_class (URIRef): The ontology class.

        Returns:
            Any: The collected class information.
        """
        class_info = super().get_class_info(owl_class)
        if class_info is None:
            return None

        extractor_set = self.extract_text_extractor_set(owl_class)
        host = self.get_host_of_uri(class_info['iri'])
        label = self.extract_only_label(extractor_set)

        class_info["olala"] = {
            "text_extractor_set": extractor_set["texts"],
            "normalized_text_extractor_set": extractor_set["normalized_texts"],
            "uri_fragment": extractor_set["uri_fragment"],
            "longest_literal": extractor_set["longest_literal"],
            "annotation_texts": extractor_set["annotation_texts"],
            "direct_string_literals": extractor_set["direct_string_literals"],
            "direct_text_literals": extractor_set["direct_text_literals"],
            "host": host,
            "label": label,
            "normalized_label": self.normalize(class_info['label']),
            "is_uri_fragment_normalization_valid": extractor_set["is_uri_fragment_normalization_valid"],
            "normalized_uri_fragment": self.normalize(extractor_set["uri_fragment"]),
        }

        return class_info


class OLaLaOMDataset(OMDataset):
    """
    A dataset class for OLaLa-enriched source-target ontologies.
    """
    track = track
    ontology_name = "OLaLa-Source-Target"
    source_ontology = OLaLaOntology()
    target_ontology = OLaLaOntology()
