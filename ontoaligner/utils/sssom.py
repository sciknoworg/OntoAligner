"""Utilities for writing SSSOM."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, TextIO

import bioregistry
import curies
import sssom_pydantic
from curies import vocabulary as v
from sssom_pydantic import MappingSet, SemanticMapping

__all__ = [
    "to_semantic_mappings",
    "write_sssom",
]


def write_sssom(
    matchings: list[dict],
    path: str | Path | TextIO,
    *,
    converter: curies.Converter | None = None,
    metadata: MappingSet | None = None,
    **kwargs: Any,
) -> None:
    """Convert matchings to SSSOM objects and write a TSV to disk.

    :param matchings: OntoAligner matchings
    :param path: The path or file to write to.
    :param converter:
    :param metadata:
    :param kwargs:
    :return:
    """
    if converter is None:
        converter = bioregistry.get_default_converter()
    mappings = to_semantic_mappings(matchings, converter)
    sssom_pydantic.write(
        mappings, path, converter=converter, metadata=metadata, **kwargs
    )


LOOKUPS = {
    "=": v.exact_match,
}


def to_semantic_mappings(
    matchings: list[dict],
    converter: curies.Converter | None = None,
) -> list[SemanticMapping]:
    if converter is None:
        converter = bioregistry.get_default_converter()
    mappings = []
    for matching in matchings:
        try:
            mapping = _to_semantic_mapping(matching, converter)
        except Exception:  # noqa: BLE001
            continue
        else:
            if mapping is not None:
                mappings.append(mapping)
    return mappings


def _to_semantic_mapping(
    matching: dict, converter: curies.Converter
) -> SemanticMapping | None:
    sub = converter.parse_uri(matching["source"])
    obj = converter.parse_uri(matching["target"])
    if sub is None or obj is None:
        return None
    return SemanticMapping(
        subject=sub,
        predicate=LOOKUPS[matching.get("relation", "=")],
        object=obj,
        # depending on the aligner, this might be one of several
        # other things from the SEMAPV namespace
        justification=v.unspecified_matching_process,
        confidence=matching.get("score"),
        mapping_date=datetime.date.today(),  # noqa: DTZ011
        # TODO in follow-up, add additional fields such as:
        #  mapping tool, license, etc. from
        #  https://mapping-commons.github.io/sssom/dev/
    )
