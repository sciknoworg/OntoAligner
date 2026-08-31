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
import datetime
import logging
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, TextIO
import bioregistry
import curies
import sssom_pydantic as spd

logger = logging.getLogger(__name__)

# Mapping of simple relation tokens to SSSOM predicate CURIEs
PREDICATE_LOOKUPS = {"=": curies.vocabulary.exact_match}

def sssom_alignment_generator(
    matchings: Iterable[Mapping[str, Any]],
    path: str | Path | TextIO,
    *,
    converter: Optional[curies.Converter] = None,
    metadata: Optional[spd.MappingSet] = None,
    **kwargs: Any,
) -> None:
    """Convert matchings to SSSOM objects and write a TSV to disk.

    Parameters
    - matchings: iterable of dict-like matchings (must contain at least
      'source' and 'target').
    - path: file path or open file-like object to write to.
    - converter: optional curies.Converter to parse URIs into CURIEs.
    - metadata: optional SSSOM sssom_pydantic.MappingSet metadata object.
    - kwargs: forwarded to sssom_pydantic.write.
    """
    if converter is None:
        converter = bioregistry.get_default_converter()

    mappings = to_semantic_mappings(matchings, converter)
    try:
        spd.write(mappings, path, converter=converter, metadata=metadata, **kwargs)
    except Exception as exc:
        logger.exception("Failed to write SSSOM output: %s", exc)
        raise


def to_semantic_mappings(
    matchings: Iterable[Mapping[str, Any]],
    converter: Optional[curies.Converter] = None
) -> List[spd.SemanticMapping]:
    """Convert an iterable of matchings to a list of SSSOM sssom_pydantic.SemanticMapping.

    Invalid or unparsable matchings are skipped but logged at debug level.
    """
    if converter is None:
        converter = bioregistry.get_default_converter()

    mappings: List[spd.SemanticMapping] = []
    for idx, matching in enumerate(matchings):
        try:
            sm = convert_matching(matching, converter)
        except Exception:  # be explicit about skipping only problematic entries
            logger.debug("Skipping mapping at index %d due to conversion error", idx, exc_info=True)
            continue
        if sm is not None:
            mappings.append(sm)
    return mappings


def convert_matching(matching: Mapping[str, Any], converter: curies.Converter) -> Optional[spd.SemanticMapping]:
    """Convert a single OntoAligner matching dict to a sssom_pydantic.SemanticMapping.

    Expected keys in ``matching``:
    - 'source': source URI/IRI
    - 'target': target URI/IRI
    - optional 'score': confidence value between 0 and 1
    - optional 'relation': one of the keys in PREDICATE_LOOKUPS (defaults to "=")

    Returns None if the subject or object cannot be parsed by the converter.
    """
    # Required fields
    source = matching.get("source")
    target = matching.get("target")
    if source is None or target is None:
        logger.debug("Matching missing source/target: %r", matching)
        return None

    subject = converter.parse_uri(source)
    object_ = converter.parse_uri(target)
    if subject is None or object_ is None:
        logger.debug("Could not parse source/target into CURIEs: %r -> %r", source, target)
        return None

    # Confidence handling
    confidence: Optional[float] = None
    if "score" in matching and matching.get("score") is not None:
        try:
            confidence = float(matching.get("score"))
        except (TypeError, ValueError):
            logger.debug("Invalid confidence value: %r", matching.get("score"))
            confidence = None

    relation = str(matching.get("relation", "="))
    predicate = PREDICATE_LOOKUPS.get(relation)
    if predicate is None:
        logger.debug("Unknown relation token %r, falling back to unspecified_matching_process", relation)
        predicate = curies.vocabulary.unspecified_matching_process

    return spd.SemanticMapping(
        subject=subject,
        predicate=predicate,
        object=object_,
        justification= curies.vocabulary.unspecified_matching_process,
        confidence=confidence,
        mapping_date=datetime.date.today(),  # noqa: DTZ011
    )
