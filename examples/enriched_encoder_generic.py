from typing import List, Dict, Any
from ontoaligner.ontology import GenericOntology


class EnrichedGenericOntology(GenericOntology):
    def __init__(
            self,
            include_synonyms: bool = True,
            include_comments: bool = True,
            synonym_weight: str = "medium",  # "light", "medium", "heavy"
            comment_weight: str = "light",
            max_comment_words: int = 50
    ):
        """
        Initialize the enriched encoder.

        Args:
            include_synonyms: Whether to include synonyms in the text
            include_comments: Whether to include comments/definitions
            synonym_weight: How to present synonyms ("light", "medium", "heavy")
            comment_weight: How to present comments ("light", "medium", "heavy")
            max_comment_words: Maximum number of words from comments to include
        """
        super().__init__()
        self.include_synonyms = include_synonyms
        self.include_comments = include_comments
        self.synonym_weight = synonym_weight
        self.comment_weight = comment_weight
        self.max_comment_words = max_comment_words

    def _format_synonyms(self, synonyms: List[str]) -> str:
        """Format synonyms based on the weight setting."""
        if not synonyms:
            return ""
        syn_text = ", ".join(synonyms[:5])  # Limit to 5 synonyms, but this can also be in a argument!
        if self.synonym_weight == "light":
            return f" ({syn_text})"
        elif self.synonym_weight == "medium":
            return f" also known as: {syn_text}"
        else:  # heavy
            return f". Synonyms include: {syn_text}"

    def _format_comment(self, comments: List[str]) -> str:
        """Format comments based on the weight setting."""
        if not comments:
            return ""

        # Combine all comments and truncate them as a structured output for modeling LLM, retrieval, fuzzy or ... but it can be also fixed.
        full_comment = " ".join(comments)
        words = full_comment.split()

        if len(words) > self.max_comment_words:
            truncated = " ".join(words[:self.max_comment_words]) + "..."
        else:
            truncated = full_comment

        if self.comment_weight == "light":
            return f". {truncated}"
        elif self.comment_weight == "medium":
            return f". Definition: {truncated}"
        else:  # heavy
            return f". Formal definition: {truncated}"

    def get_owl_items(self, owl: Dict) -> Dict[str, Any]:
        """
        Extract and format concept information from OWL data.

        Args:
            owl: Dictionary containing 'iri', 'label', 'synonyms', and 'comment'

        Returns:
            Dictionary with 'iri' and enriched 'text'
        """

        text_parts = [owl["label"]]

        if self.include_synonyms and owl.get('synonyms'):
            synonym_text = self._format_synonyms(owl['synonyms'])
            if synonym_text:
                text_parts.append(synonym_text)

        if self.include_comments and owl.get('comment'):
            comment_text = self._format_comment(owl['comment'])
            if comment_text:
                text_parts.append(comment_text)

        enriched_text = "".join(text_parts)

        return {
            "iri": owl["iri"],
            "text": enriched_text.lower().strip()
        }

    def preprocess(self, source: List[Dict], target: List[Dict]) -> List[List[Dict]]:
        """
        Preprocess source and target ontologies.

        Args:
            source: List of source ontology concepts
            target: List of target ontology concepts

        Returns:
            List containing [source_encoded, target_encoded]
        """
        source_encoded = [self.get_owl_items(item) for item in source]
        target_encoded = [self.get_owl_items(item) for item in target]
        return [source_encoded, target_encoded]



example_source = [
    {
        "iri": "http://example.org/HeartDisease",
        "label": "Heart Disease",
        "synonyms": ["Cardiac Disease", "Heart Condition", "Cardiovascular Disease"],
        "comment": [
            "A disease affecting the heart or blood vessels. Common types include coronary artery disease and heart failure."],
        "parents": [{"label": "Cardiovascular System Disease"}]
    },
    {
        "iri": "http://example.org/Diabetes",
        "label": "Diabetes Mellitus",
        "synonyms": ["Diabetes", "DM", "Sugar Disease"],
        "comment": ["A metabolic disorder characterized by high blood sugar levels over a prolonged period."],
        "parents": [{"label": "Metabolic Disease"}]
    }
]

example_target = [
    {
        "iri": "http://another.org/CardiacAilment",
        "label": "Cardiac Ailment",
        "synonyms": ["Heart Problem"],
        "comment": ["Any disease or condition affecting the heart."],
        "parents": [{"label": "Circulatory Disease"}]
    }
]

configs = [
    ("Minimal (Label Only)", {
        "include_synonyms": False,
        "include_comments": False
    }),
    ("With Synonyms", {
        "include_synonyms": True,
        "include_comments": False,
        "synonym_weight": "medium"
    }),
    ("Full Enrichment", {
        "include_synonyms": True,
        "include_comments": True,
        "synonym_weight": "medium",
        "comment_weight": "medium",
        "max_comment_words": 20
    })
]

for config_name, config_params in configs:
    print(f"\n{config_name}:")
    print("-" * 60)
    ontology = EnrichedGenericOntology(**config_params)

    result = ontology.get_owl_items(example_source[0])
    print(f"Text: {result['text']}")

