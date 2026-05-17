"""
This script defines OLaLa lightweight matchers for ontology matching.

The high-precision matcher creates exact correspondences from normalized
labels and URI fragments.
"""

from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from ..lightweight import Lightweight


class OLaLaHighPrecisionMatcher(Lightweight):
    """
    A high-precision exact matcher for OLaLa.
    """

    def __init__(self, confidence: float = 1.0, **kwargs) -> None:
        """
        Initializes the OLaLa high-precision matcher.

        Parameters:
            confidence (float): The confidence assigned to exact matches.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.kwargs["confidence"] = confidence

    def get_string_representations(self, item: Dict[str, Any]) -> Set[str]:
        """
        Retrieves high-precision string representations for one entity.

        Parameters:
            item (Dict[str, Any]): The encoded ontology item.

        Returns:
            Set[str]: The normalized high-precision texts.
        """
        return {
            str(text).strip()
            for text in item.get("hp_texts", [])
            if text is not None and str(text).strip() != ""
        }

    def build_text_index(self, ontology: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        """
        Builds a text-to-IRI index for source ontology entities.

        Parameters:
            ontology (List[Dict[str, Any]]): The encoded source ontology.

        Returns:
            Dict[str, Set[str]]: The text-to-source-IRI index.
        """
        text_to_iris = {}

        for item in ontology:
            iri = item.get("iri")
            if iri is None:
                continue

            for text in self.get_string_representations(item):
                text_to_iris.setdefault(text, set()).add(iri)

        return text_to_iris

    def match_exact_texts(
        self,
        source_ontology: List[Dict[str, Any]],
        target_ontology: List[Dict[str, Any]],
    ) -> Set[Tuple[str, str]]:
        """
        Creates exact correspondences from normalized texts.

        Parameters:
            source_ontology (List[Dict[str, Any]]): The encoded source ontology.
            target_ontology (List[Dict[str, Any]]): The encoded target ontology.

        Returns:
            Set[Tuple[str, str]]: The exact candidate pairs.
        """
        text_to_source_iris = self.build_text_index(source_ontology)
        pairs = set()

        for target in target_ontology:
            target_iri = target.get("iri")
            if target_iri is None:
                continue

            for target_text in self.get_string_representations(target):
                source_iris = text_to_source_iris.get(target_text)
                if source_iris is None:
                    continue

                for source_iri in source_iris:
                    pairs.add((source_iri, target_iri))

        return pairs

    def filter_n_to_m(
        self,
        pairs: Set[Tuple[str, str]],
    ) -> Set[Tuple[str, str]]:
        """
        Removes N:M correspondences.

        Parameters:
            pairs (Set[Tuple[str, str]]): The candidate pairs.

        Returns:
            Set[Tuple[str, str]]: The remaining unambiguous pairs.
        """
        source_counts = Counter(source for source, _ in pairs)
        target_counts = Counter(target for _, target in pairs)

        return {
            (source, target)
            for source, target in pairs
            if source_counts[source] == 1 and target_counts[target] == 1
        }

    def generate(self, input_data: List) -> List:
        """
        Generates high-precision exact correspondences.

        Parameters:
            input_data (List): The encoded source and target ontologies.

        Returns:
            List: The high-precision correspondences.
        """
        source_ontology = input_data[0]
        target_ontology = input_data[1]

        pairs = self.match_exact_texts(
            source_ontology=source_ontology,
            target_ontology=target_ontology,
        )
        pairs = self.filter_n_to_m(pairs)

        return [
            {
                "source": source_iri,
                "target": target_iri,
                "score": self.kwargs["confidence"],
            }
            for source_iri, target_iri in sorted(pairs)
        ]

    def __str__(self):
        """
        Returns the string representation of the matcher.

        Returns:
            str: The string representation of the matcher.
        """
        return super().__str__() + "-OLaLaHighPrecisionMatcher"
