"""Tests for SSSOM export."""

import unittest
from pathlib import Path
import json
from ontoaligner.utils.sssom import to_semantic_mappings
from curies import Converter, NamableReference
from sssom_pydantic import SemanticMapping
from sssom_pydantic.testing import assert_semantic_mapping_equal
import datetime

HERE = Path(__file__).parent.resolve()
TEST_DATA = HERE.parent.joinpath("data", "ensemble_alignments.json")


class TestSSSOM(unittest.TestCase):
    """A test case for SSSOM export."""

    def test_sssom(self) -> None:
        today = datetime.date.today()
        with TEST_DATA.open() as file:
            matchings = json.load(file)
        converter = Converter.from_prefix_map(
            {
                "fishtraits": "https://kos.lifewatch.eu/thesauri/fishtraits/",
                "zooplanktontraits": "https://kos.lifewatch.eu/thesauri/zooplanktontraits/",
            }
        )
        expected = SemanticMapping.exact(
            "fishtraits:c_10",
            "zooplanktontraits:c_13",
            mapping_date=today,
        )
        actual = to_semantic_mappings(matchings, converter=converter)
        assert_semantic_mapping_equal(self, expected, actual[0])
