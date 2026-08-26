"""Utilities for writing SSSOM."""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

import bioregistry
import curies
import sssom_pydantic
from curies import vocabulary as v
from sssom_pydantic import MappingSet, SemanticMapping

__all__ = [
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
    matchings: list[dict], converter: curies.Converter
) -> list[SemanticMapping]:
    mappings = []
    for matching in matchings:
        try:
            mapping = _to_semantic_mapping(matching, converter)
        except Exception:
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
        confidence=matching.get("score"),
    )
