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
Literal processing and embedding utilities for the FLORA knowledge graph alignment system.

This module provides tools for:
- Literal detection and parsing (RDF literals with language tags and datatypes)
- Literal comparison and normalization
- Semantic embedding-based literal similarity
- Bucket-based literal organization by type (dates, quantities, strings, etc.)

Key components:

- **Literal utilities** – :func:`is_literal`, :func:`split_literal`, :func:`is_readable`,
  :func:`is_inverse`, :func:`invert`.
- **Normalization** – :func:`numeric_normalization`, :func:`decode_unicode`.
- **Bucketing** – :func:`get_literal_buckets`, :func:`compare_literals`,
  :func:`compare_literals_identity`.
- **Embedding model** – :class:`FLORALiteralsEmbedding` for semantic similarity.
"""

from typing import List, Dict, Tuple, DefaultDict, Optional, Any
import os
import re
import math
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

#################################################################
#                    Regex Patterns                             #
#################################################################

# Regex for RDF literals with optional language tags and datatypes
LITERAL_REGEX = re.compile('"([^"]*)"(@([a-z-]+))?(\\^\\^(.*))?')

# Regex for float values
FLOAT_REGEX = re.compile('^"?([+-])?([0-9.]+)"?$')
INTEGER_REGEX = re.compile('^"?[+-]?[0-9.]+"?$') # Regex for int values

SCI_FLOAT_REGEX = re.compile('^"?([+-])?([0-9.]+[Ee][+-]?[0-9]+)"?$')

# Regex for numbers: post codes, phone numbers, etc.
NUMBER_REGEX = re.compile('[\d\W]+')

#################################################################
#                 Literal Utility Functions                     #
#################################################################


def is_inverse(rel: str) -> bool:
    """Check if a relation is marked as inverse.

    Args:
        rel: The relation string.

    Returns:
        True if the relation ends with '-' (inverse marker), False otherwise.
    """
    return rel[-1] == '-'


def invert(rel: str) -> str:
    """Toggle the inverse status of a relation.

    Removes the '-' suffix if present, or adds it if absent.

    Args:
        rel: The relation string.

    Returns:
        The inverted relation string.
    """
    return rel[:-1] if is_inverse(rel) else rel + '-'


def is_readable(txt: str) -> bool:
    """Check if a string is human-readable text.

    A string is considered readable if it contains multiple words or if the
    ratio of non-alphabetic characters is below 50%.

    Args:
        txt: The text to check.

    Returns:
        True if the text is readable, False otherwise (e.g., URLs, mostly symbols).
    """
    if len(txt.split()) > 1:
        return True
    non_alpha_ratio = sum(not c.isalpha() for c in txt) / len(txt)
    # Not a web link
    if txt.startswith('http'):
        return False
    return non_alpha_ratio < 0.5


def is_literal(term: Any) -> bool:
    """Check if a term is an RDF literal value.

    A term is a literal if it matches the RDF literal pattern or is a float value.

    Args:
        term: The term to check.

    Returns:
        True if the term is a literal, False otherwise.
    """
    try:
        return bool(re.match(LITERAL_REGEX, term) or re.match(FLOAT_REGEX, term) or re.match(INTEGER_REGEX, term))
    except Exception:
        return False


def numeric_normalization(term: str) -> str:
    """Normalize a numeric string by removing non-digit characters.

    Used for normalizing phone numbers, postal codes, etc.

    Args:
        term: The term to normalize (may be quoted).

    Returns:
        The normalized string containing only digits.
    """
    term = term.strip('"')
    normalized = re.sub(r'[^0-9]', '', term)
    return normalized


def decode_unicode(encoded_str: str) -> str:
    """Decode unicode escape sequences in a string.

    Handles both custom unicode encoding (_u####_) and standard unicode escapes (\\u####).

    Args:
        encoded_str: The string with encoded unicode sequences.

    Returns:
        The decoded string with unicode characters properly handled.
    """
    decoded_string = re.sub(r'_u([0-9A-F]{4})_', lambda x: chr(int(x.group(1), 16)), encoded_str)
    if '\\u' in decoded_string:
        try:
            s = decoded_string.encode('utf-8').decode('unicode_escape')
            return s.encode('utf-16', 'surrogatepass').decode('utf-16')
        except Exception:
            return decoded_string
    else:
        return decoded_string


def split_literal(term: str) -> Tuple[str, Optional[Any], Optional[str], Optional[str]]:
    """Parse an RDF literal into its components.

    Extracts the string value, numeric value (if applicable), language tag,
    and datatype from an RDF literal.

    Args:
        term: The RDF literal term to parse.

    Returns:
        A tuple of (literal_string, numeric_value, language_tag, datatype).
        numeric_value is None for non-numeric literals or for dates.
    """
    literal, _, lang, _, datatype = re.match(LITERAL_REGEX, term).groups()

    # Handle date types
    if datatype in ['xsd:date', 'xsd:gYear', 'xsd:gYearMonth', 'xsd:datetime']:
        return (literal, None, lang, datatype)

    # Try to parse as numeric value
    float_match = re.match(FLOAT_REGEX, literal)
    sci_float_match = re.match(SCI_FLOAT_REGEX, literal)
    if (float_match or sci_float_match) and lang is None:
        try:
            # Try parsing as integer first
            value = int(literal.strip('"'))
            if len(str(value)) != len(literal.strip('"')):
                # e.g., "06" != 6, "+3" != 3 (preserve format)
                return (literal, None, lang, datatype)
            return (literal, value, lang, datatype)
        except Exception:
            try:
                # Try parsing as float
                value = float(literal.strip('"'))
                return (literal, value, lang, datatype)
            except ValueError:
                # e.g., "23.78.9" (invalid version number)
                return (literal, None, lang, datatype)

    # Handle normalized numbers (phone numbers, postal codes, etc.)
    match_number = re.fullmatch(NUMBER_REGEX, literal)
    if match_number:
        value = numeric_normalization(literal)
        if len(value) >= 10:  # e.g., phone numbers
            return (literal, value, lang, datatype)

    # Handle string literals
    decoded_literal = decode_unicode(literal)
    return (decoded_literal, None, lang, 'xsd:string')


#################################################################
#              Literal Bucketing and Comparison                 #
#################################################################


def get_literal_buckets(kb: Any) -> Tuple[DefaultDict, DefaultDict, DefaultDict, DefaultDict]:
    """Organize literals from a knowledge base into type-specific buckets.

    Classifies literals into quantity (numbers), digit (IDs, versions), string,
    and date buckets for efficient comparison.

    Args:
        kb: The knowledge base object.

    Returns:
        A tuple of four buckets: (quantity_bucket, digit_bucket, string_bucket, date_bucket).
        Each bucket is a defaultdict mapping canonical values to lists of original literals.
    """
    quantity_bucket = defaultdict(list)  # e.g., integers, floats
    digit_bucket = defaultdict(list)  # e.g., IDs, code versions
    string_bucket = defaultdict(list)  # e.g., strings
    date_bucket = defaultdict(list)  # e.g., dates

    for obj in kb.objects():
        if is_literal(obj):
            literal, num_value, lang, datatype = split_literal(obj)

            # Date handling
            if datatype in ['xsd:date', 'xsd:gYear', 'xsd:gYearMonth', 'xsd:datetime', 'xsd:gMonthDay']:
                date_bucket[literal].append(obj)
                continue

            # Numeric handling
            if num_value is not None:
                if isinstance(num_value, (int, float)):
                    if datatype is not None:
                        quantity_bucket[num_value].append(obj)
                        continue
                    # e.g., "0" vs "0"^^xsd:decimal (preserve format)
                    digit_bucket[literal].append(obj)
                    continue
                # Normalized strings (phone numbers, postal codes)
                if isinstance(num_value, str) and len(num_value) > 0:
                    string_bucket[num_value].append(obj)
                continue

            # String handling
            string_bucket[literal].append(obj)
            string_bucket[literal.lower()].append(obj)  # Add lowercase variants

    return quantity_bucket, digit_bucket, string_bucket, date_bucket


def compare_literals(
    same_as_scores: Dict[str, Dict[str, float]],
    bucket1: DefaultDict,
    bucket2: DefaultDict,
    datatype: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare literals from two buckets and update similarity scores.

    Compares literals in the same type bucket and assigns similarity scores
    based on the datatype (quantity, date, digit, or string).

    Args:
        same_as_scores: Dictionary to update with similarity scores.
        bucket1: First literal bucket.
        bucket2: Second literal bucket.
        datatype: Type of literals ('string', 'quantity', 'date', or 'digit').

    Returns:
        Updated same_as_scores dictionary.

    Raises:
        ValueError: If datatype is not specified.
    """
    if datatype is None:
        raise ValueError('Datatype must be specified')

    if datatype == 'string':
        pass
    elif datatype == 'quantity':
        for key1 in bucket1:
            for key2 in bucket2:
                if math.isclose(key1, key2):
                    for obj1 in bucket1[key1]:
                        if obj1 not in same_as_scores:
                            same_as_scores[obj1] = {}
                        for obj2 in bucket2[key2]:
                            same_as_scores[obj1][obj2] = 1.0
    elif datatype == 'date':
        for key1 in bucket1:
            if key1 in bucket2:
                for obj1 in bucket1[key1]:
                    if obj1 not in same_as_scores:
                        same_as_scores[obj1] = {}
                    for obj2 in bucket2[key1]:
                        same_as_scores[obj1][obj2] = 1.0
                continue
            # Handle partial date matches (e.g., 2023-01-15 matches 2023-01)
            last_dash = key1.rfind('-')
            while last_dash != -1:
                key1_partial = key1[:last_dash]
                if key1_partial in bucket2:
                    for obj1 in bucket1[key1]:
                        if obj1 not in same_as_scores:
                            same_as_scores[obj1] = {}
                        # Score based on specificity
                        score = (key1_partial.count('-') + 1) / 3
                        for obj2 in bucket2[key1_partial]:
                            same_as_scores[obj1][obj2] = score
                    break
                last_dash = key1.rfind('-', 0, last_dash)
    else:  # digits (IDs, code versions, etc.)
        for key in bucket1:
            if key in bucket2 and len(key.strip('"')) > 0:
                for obj1 in bucket1[key]:
                    same_as_scores[obj1] = {}
                    for obj2 in bucket2[key]:
                        same_as_scores[obj1][obj2] = 1.0
    return same_as_scores


def compare_literals_identity(
    same_as_scores: Dict[str, Dict[str, float]],
    bucket1: DefaultDict,
    bucket2: DefaultDict,
) -> Dict[str, Dict[str, float]]:
    """Compare literals using exact string matching only.

    Finds exact matches between literals in two buckets and assigns
    perfect similarity scores.

    Args:
        same_as_scores: Dictionary to update with similarity scores.
        bucket1: First literal bucket.
        bucket2: Second literal bucket.

    Returns:
        Updated same_as_scores dictionary.
    """
    for key in bucket1:
        if key in bucket2:
            for obj1 in bucket1[key]:
                if obj1 not in same_as_scores:
                    same_as_scores[obj1] = {}
                for obj2 in bucket2[key]:
                    same_as_scores[obj1][obj2] = 1.0
    return same_as_scores


#################################################################
#              Semantic Embedding Model                         #
#################################################################


class FLORALiteralsEmbedding:
    """Semantic embedding model for RDF literal similarity computation.

    Uses transformer-based embeddings (e.g., PEARL) to compute semantic similarity
    between string literals for knowledge graph alignment.
    """

    def __init__(
        self,
        model_id: str = 'Lihuchen/pearl_small',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        identity: bool = False,
        emb_path: Optional[str] = None,
    ) -> None:
        """Initialize the embedding model.

        Args:
            model_id: Hugging Face model ID for the embedding model.
            device: Device to load the model on ('cuda' or 'cpu').
            identity: If True, do not load the model and tokenizer (used for string identity only).
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.kg1_pretrained = None
        self.kg2_pretrained = None
        if not identity:
            if emb_path is not None:
                try:
                    self.kg1_pretrained = self.load_embeddings(emb_path)
                    self.kg2_pretrained = self.load_embeddings(emb_path)
                except Exception as e:
                    raise RuntimeError(f"Error loading embeddings from {emb_path}: {e} \n "
                                       f"Please ensure the embedding files exist and are in the correct format. ")
            else:
                self.model = AutoModel.from_pretrained(model_id)
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.model.to(device)
        self.device = device

    def encode(self, input_texts: List[str]) -> torch.Tensor:
        """Encode text inputs into embedding vectors.

        Args:
            input_texts: List of text strings to encode.

        Returns:
            Tensor of normalized embedding vectors.
        """
        batch_dict = self.tokenizer(
            input_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        outputs = self.model(**batch_dict)
        embeddings = self.averaged_pooling(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def averaged_pooling(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean pooling of token embeddings with attention masking.

        Args:
            last_hidden_states: Hidden states from the transformer model.
            attention_mask: Attention mask indicating valid tokens.

        Returns:
            Pooled embeddings of shape (batch_size, hidden_dim).
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def similarity(self, input_texts: List[str]) -> List[List[float]]:
        """Compute pairwise cosine similarity between the first text and others.

        Args:
            input_texts: List of texts (first text compared against others).

        Returns:
            List of similarity scores.
        """
        embeddings = self.encode(input_texts)
        scores = embeddings[:1] @ embeddings[1:].T
        return scores.tolist()

    def embedding_strings(
        self,
        kb: Any,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """Compute embeddings for all readable string literals in a knowledge base.

        Args:
            kb: The knowledge base object.
            batch_size: Number of texts to process per batch.

        Returns:
            Dictionary with 'id' (literal to index mapping) and 'emb' (embedding matrix).
        """
        literal2id = {}
        literals = []
        cnt = 0
        for obj in kb.objects():
            if is_literal(obj):
                term, _, _, datatype = split_literal(obj)
                if len(term) <= 1:
                    continue
                if (not is_readable(term)) and datatype == 'xsd:string':
                    continue
                if datatype == 'xsd:string' and is_readable(term):
                    if term in literal2id:
                        continue
                    literal2id[term] = cnt
                    literals.append(term)
                    cnt += 1

        # Validate correctness
        for ltr in literals:
            idx = literal2id[ltr]
            assert literals[idx] == ltr

        # Compute embeddings in batches
        embeddings_list = []
        for i in tqdm(
            range(0, len(literals), batch_size),
            desc="        Computing embeddings",
        ):
            batch_literals = literals[i:i + batch_size]
            with torch.no_grad():
                embeddings = self.encode(batch_literals).cpu()
                embeddings_list.append(embeddings)

        # Stack embeddings into a matrix
        embedding_matrix = torch.cat(embeddings_list, dim=0)
        embedding_matrix = embedding_matrix.numpy()
        return {"id": literal2id, "emb": embedding_matrix}

    def map_literals(
        self,
        kb1: Any,
        kb2: Any,
        same_as_score: Dict[str, Dict[str, float]],
        identity: bool = False,
        threshold: float = 0.5,
    ) -> Dict[str, Dict[str, float]]:
        """Map literal alignments between two knowledge bases.

        Uses type-specific buckets and semantic embeddings to find matching literals.

        Args:
            kb1: First knowledge base.
            kb2: Second knowledge base.
            same_as_score: Dictionary to accumulate similarity scores.
            identity: If True, only use exact string matching.
            threshold: Minimum similarity score for semantic matches.

        Returns:
            Updated same_as_score dictionary.
        """
        map_scores = {}
        quantity_bucket1, digit_bucket1, str_bucket1, date_bucket1 = get_literal_buckets(kb1)
        quantity_bucket2, digit_bucket2, str_bucket2, date_bucket2 = get_literal_buckets(kb2)

        if not identity:
            # Dates
            map_scores = compare_literals(map_scores, date_bucket1, date_bucket2, 'date')
            # Compare numbers
            map_scores = compare_literals(map_scores, quantity_bucket1, quantity_bucket2, 'quantity')
            map_scores = compare_literals(map_scores, digit_bucket1, digit_bucket2, 'digit')

            # Compare strings - first get exact match
            map_scores = compare_literals_identity(map_scores, str_bucket1, str_bucket2)

            # Compute semantic embeddings
            kb1_embedding = self.embedding_strings(kb=kb1) if self.kg1_pretrained is None else self.kg1_pretrained
            literal2id_kb1, embedding_matrix_kb1 = kb1_embedding['id'], kb1_embedding['emb']

            kb2_embedding = self.embedding_strings(kb=kb2) if self.kg2_pretrained is None else self.kg2_pretrained
            literal2id_kb2, embedding_matrix_kb2 = kb2_embedding['id'], kb2_embedding['emb']

            id2literal_kb2 = {v: k for k, v in literal2id_kb2.items()}

            # Compute similarity matrix
            similarity_mat = embedding_matrix_kb1 @ embedding_matrix_kb2.T

            for key1 in str_bucket1:
                # Skip if exact match already found
                if key1 in str_bucket2 and len(key1.strip('"')) > 0:
                    continue
                if key1 in literal2id_kb1:
                    max_indices = similarity_mat[literal2id_kb1[key1], :].argsort()[::-1]
                    for max_idx in max_indices[:1]:
                        # Verify the match is in string bucket
                        if id2literal_kb2[max_idx] not in str_bucket2:
                            continue
                        if similarity_mat[literal2id_kb1[key1], max_idx] < threshold:
                            continue
                        for obj1 in str_bucket1[key1]:
                            if obj1 in map_scores and \
                                    round(max(map_scores.get(obj1, {None: 0}).values()), 2) >= 1.0:
                                continue
                            if obj1 not in map_scores:
                                map_scores[obj1] = {}
                            for obj2 in str_bucket2[id2literal_kb2[max_idx]]:
                                map_scores[obj1][obj2] = similarity_mat[literal2id_kb1[key1], max_idx]
        else:
            # Identity mapping only
            map_scores = compare_literals_identity(map_scores, str_bucket1, str_bucket2)
            map_scores = compare_literals_identity(map_scores, date_bucket1, date_bucket2)
            map_scores = compare_literals_identity(map_scores, quantity_bucket1, quantity_bucket2)
            map_scores = compare_literals_identity(map_scores, digit_bucket1, digit_bucket2)

        # Load scores into same_as_score
        for literal1 in map_scores:
            if literal1 not in kb1.index:
                continue
            # Check for empty mapping
            if literal1 not in same_as_score:
                same_as_score[literal1] = map_scores[literal1].copy()
        return same_as_score

    def encode_save(
        self,
        kb1: Any,
        kb2: Any,
        emb_path: str,
        batch_size: int = 128,
    ) -> None:
        """Compute and save embeddings for two knowledge bases.

        Args:
            kb1: First knowledge base.
            kb2: Second knowledge base.
            emb_path: Directory path to save embedding files.
            batch_size: Number of texts to process per batch.
        """
        kb1_emb = self.embedding_strings(kb1, batch_size)
        kb2_emb = self.embedding_strings(kb2, batch_size)

        # Save embeddings
        if not os.path.exists(emb_path):
            os.makedirs(emb_path)
        with open(os.path.join(emb_path, "kb1.pkl"), "wb") as f:
            pickle.dump(kb1_emb, f)
        with open(os.path.join(emb_path, "kb2.pkl"), "wb") as f:
            pickle.dump(kb2_emb, f)

    def load_embeddings(
        self,
        emb_path: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load precomputed embeddings for two knowledge bases.

        Args:
            emb_path: Directory path where embedding files are saved.
        """
        with open(os.path.join(emb_path, "kb1.pkl"), "rb") as f:
            kb1_emb = pickle.load(f)
        with open(os.path.join(emb_path, "kb2.pkl"), "rb") as f:
            kb2_emb = pickle.load(f)
        return kb1_emb, kb2_emb
