"""Tests for SSSOM export using sssom_pydantic."""

import unittest
from pathlib import Path
import json
from io import StringIO
from unittest.mock import Mock, patch

from ontoaligner.utils.sssom_pydantic import (
    to_semantic_mappings,
    convert_matching,
    sssom_alignment_generator,
    PREDICATE_LOOKUPS,
)
from curies import Converter
import curies

HERE = Path(__file__).parent.resolve()
TEST_DATA = HERE.parent.joinpath("data", "ensemble_alignments.json")


class TestPredicateLookups(unittest.TestCase):
    """Test cases for PREDICATE_LOOKUPS dictionary."""

    def test_predicate_lookups_contains_exact_match(self):
        """Test that PREDICATE_LOOKUPS contains '='."""
        self.assertIn("=", PREDICATE_LOOKUPS)

    def test_exact_match_value(self):
        """Test that '=' maps to exact_match."""
        self.assertEqual(
            PREDICATE_LOOKUPS["="],
            curies.vocabulary.exact_match
        )

    def test_predicate_lookups_not_empty(self):
        """Test that PREDICATE_LOOKUPS is not empty."""
        self.assertGreater(len(PREDICATE_LOOKUPS), 0)


class TestConvertMatchingEdgeCases(unittest.TestCase):
    """Test cases for convert_matching function edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = Converter.from_prefix_map(
            {
                "fishtraits": "https://kos.lifewatch.eu/thesauri/fishtraits/",
                "zooplanktontraits": "https://kos.lifewatch.eu/thesauri/zooplanktontraits/",
            }
        )

    def test_matching_missing_source_returns_none(self):
        """Test that matching without source returns None."""
        matching = {
            "target": "https://kos.lifewatch.eu/thesauri/zooplanktontraits/c_13",
        }
        result = convert_matching(matching, self.converter)
        self.assertIsNone(result)

    def test_matching_missing_target_returns_none(self):
        """Test that matching without target returns None."""
        matching = {
            "source": "https://kos.lifewatch.eu/thesauri/fishtraits/c_10",
        }
        result = convert_matching(matching, self.converter)
        self.assertIsNone(result)

    def test_matching_with_none_source_returns_none(self):
        """Test that matching with None source returns None."""
        matching = {
            "source": None,
            "target": "https://kos.lifewatch.eu/thesauri/zooplanktontraits/c_13",
        }
        result = convert_matching(matching, self.converter)
        self.assertIsNone(result)

    def test_matching_with_none_target_returns_none(self):
        """Test that matching with None target returns None."""
        matching = {
            "source": "https://kos.lifewatch.eu/thesauri/fishtraits/c_10",
            "target": None,
        }
        result = convert_matching(matching, self.converter)
        self.assertIsNone(result)

    def test_matching_with_unparseable_source_returns_none(self):
        """Test that unparseable source returns None."""
        matching = {
            "source": "http://unparseable.example.org/unknown/entity",
            "target": "https://kos.lifewatch.eu/thesauri/zooplanktontraits/c_13",
        }
        result = convert_matching(matching, self.converter)
        self.assertIsNone(result)

    def test_matching_with_unparseable_target_returns_none(self):
        """Test that unparseable target returns None."""
        matching = {
            "source": "https://kos.lifewatch.eu/thesauri/fishtraits/c_10",
            "target": "http://unparseable.example.org/unknown/entity",
        }
        result = convert_matching(matching, self.converter)
        self.assertIsNone(result)


class TestToSemanticMappings(unittest.TestCase):
    """Test cases for to_semantic_mappings function."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = Converter.from_prefix_map(
            {
                "fishtraits": "https://kos.lifewatch.eu/thesauri/fishtraits/",
                "zooplanktontraits": "https://kos.lifewatch.eu/thesauri/zooplanktontraits/",
            }
        )

    def test_empty_matchings_returns_empty_list(self):
        """Test with empty matchings list."""
        result = to_semantic_mappings([], converter=self.converter)
        self.assertEqual(result, [])
        self.assertIsInstance(result, list)

    def test_returns_list(self):
        """Test that to_semantic_mappings returns a list."""
        result = to_semantic_mappings([], converter=self.converter)
        self.assertIsInstance(result, list)

    def test_skips_matching_with_none_source(self):
        """Test that matchings with None source are skipped."""
        matchings = [
            {
                "source": None,
                "target": "https://kos.lifewatch.eu/thesauri/zooplanktontraits/c_13",
            }
        ]
        result = to_semantic_mappings(matchings, converter=self.converter)
        self.assertEqual(len(result), 0)

    def test_skips_matching_with_none_target(self):
        """Test that matchings with None target are skipped."""
        matchings = [
            {
                "source": "https://kos.lifewatch.eu/thesauri/fishtraits/c_10",
                "target": None,
            }
        ]
        result = to_semantic_mappings(matchings, converter=self.converter)
        self.assertEqual(len(result), 0)

    def test_skips_unparseable_matchings(self):
        """Test that unparseable matchings are skipped."""
        matchings = [
            {
                "source": "http://unknown.org/entity1",
                "target": "https://kos.lifewatch.eu/thesauri/zooplanktontraits/c_13",
            }
        ]
        result = to_semantic_mappings(matchings, converter=self.converter)
        self.assertEqual(len(result), 0)

    def test_default_converter_when_none(self):
        """Test that default converter is used when None is passed."""
        # Use GOinically known URIs
        matchings = [
            {
                "source": "http://purl.obolibrary.org/obo/GO_0008150",
                "target": "http://purl.obolibrary.org/obo/GO_0008151",
            }
        ]
        result = to_semantic_mappings(matchings, converter=None)
        self.assertIsInstance(result, list)

    def test_generator_input_is_processed(self):
        """Test that generator input is properly iterated."""
        def matching_generator():
            yield {
                "source": None,
                "target": "https://kos.lifewatch.eu/thesauri/zooplanktontraits/c_13",
            }
            yield {
                "source": "https://kos.lifewatch.eu/thesauri/fishtraits/c_10",
                "target": None,
            }

        result = to_semantic_mappings(matching_generator(), converter=self.converter)

        # Both should be skipped
        self.assertEqual(len(result), 0)

    def test_processes_multiple_matchings(self):
        """Test processing multiple matchings."""
        matchings = [
            {"source": None, "target": "target1"},  # Skip
            {"source": "source2", "target": None},  # Skip
            {"source": None, "target": None},  # Skip
        ]
        result = to_semantic_mappings(matchings, converter=self.converter)
        self.assertEqual(len(result), 0)


class TestSSSOMAlignmentGeneratorBasic(unittest.TestCase):
    """Test cases for sssom_alignment_generator function."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = Converter.from_prefix_map(
            {
                "fishtraits": "https://kos.lifewatch.eu/thesauri/fishtraits/",
                "zooplanktontraits": "https://kos.lifewatch.eu/thesauri/zooplanktontraits/",
            }
        )

    @patch('ontoaligner.utils.sssom_pydantic.spd.write')
    def test_write_is_called(self, mock_write):
        """Test that spd.write is called."""
        output = StringIO()
        matchings = [
            {
                "source": "https://kos.lifewatch.eu/thesauri/fishtraits/c_10",
                "target": "https://kos.lifewatch.eu/thesauri/zooplanktontraits/c_13",
            }
        ]

        sssom_alignment_generator(
            matchings,
            output,
            converter=self.converter,
        )

        mock_write.assert_called_once()

    @patch('ontoaligner.utils.sssom_pydantic.to_semantic_mappings')
    @patch('ontoaligner.utils.sssom_pydantic.spd.write')
    def test_to_semantic_mappings_called(self, mock_write, mock_to_mappings):
        """Test that to_semantic_mappings is called."""
        mock_to_mappings.return_value = []

        output = StringIO()
        matchings = [
            {
                "source": "https://kos.lifewatch.eu/thesauri/fishtraits/c_10",
                "target": "https://kos.lifewatch.eu/thesauri/zooplanktontraits/c_13",
            }
        ]

        sssom_alignment_generator(
            matchings,
            output,
            converter=self.converter,
        )

        mock_to_mappings.assert_called_once()

    @patch('ontoaligner.utils.sssom_pydantic.bioregistry.get_default_converter')
    def test_default_converter_used_when_none(self, mock_default_converter):
        """Test that default converter is used when None is passed."""
        mock_converter = Mock()
        mock_default_converter.return_value = mock_converter

        with patch('ontoaligner.utils.sssom_pydantic.to_semantic_mappings') as mock_to_mappings:
            with patch('ontoaligner.utils.sssom_pydantic.spd.write'):
                mock_to_mappings.return_value = []

                output = StringIO()
                matchings = []

                sssom_alignment_generator(
                    matchings,
                    output,
                    converter=None,
                )

                mock_default_converter.assert_called_once()


class TestTestDataIntegrity(unittest.TestCase):
    """Test the integrity and structure of test data."""

    def test_test_data_file_exists(self):
        """Test that test data file exists."""
        self.assertTrue(TEST_DATA.exists())

    def test_test_data_is_valid_json(self):
        """Test that test data is valid JSON."""
        with TEST_DATA.open() as file:
            matchings = json.load(file)

        self.assertIsInstance(matchings, list)

    def test_test_data_has_content(self):
        """Test that test data has content."""
        with TEST_DATA.open() as file:
            matchings = json.load(file)

        self.assertGreater(len(matchings), 0)

    def test_test_data_structure(self):
        """Test structure of test data entries."""
        with TEST_DATA.open() as file:
            matchings = json.load(file)

        for matching in matchings:
            self.assertIn("source", matching)
            self.assertIn("target", matching)
            self.assertIsInstance(matching["source"], str)
            self.assertIsInstance(matching["target"], str)

    def test_test_data_has_scores(self):
        """Test that test data contains score values."""
        with TEST_DATA.open() as file:
            matchings = json.load(file)

        matchings_with_score = [m for m in matchings if "score" in m]
        self.assertGreater(len(matchings_with_score), 0)


class TestConverterParsing(unittest.TestCase):
    """Test CURIE converter parsing of test data."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = Converter.from_prefix_map(
            {
                "fishtraits": "https://kos.lifewatch.eu/thesauri/fishtraits/",
                "zooplanktontraits": "https://kos.lifewatch.eu/thesauri/zooplanktontraits/",
            }
        )

    def test_converter_parses_fishtraits_uri(self):
        """Test that converter can parse fishtraits URIs."""
        uri = "https://kos.lifewatch.eu/thesauri/fishtraits/c_10"
        result = self.converter.parse_uri(uri)

        self.assertIsNotNone(result)
        self.assertEqual(result.prefix, "fishtraits")
        self.assertEqual(result.identifier, "c_10")

    def test_converter_parses_zooplanktontraits_uri(self):
        """Test that converter can parse zooplanktontraits URIs."""
        uri = "https://kos.lifewatch.eu/thesauri/zooplanktontraits/c_13"
        result = self.converter.parse_uri(uri)

        self.assertIsNotNone(result)
        self.assertEqual(result.prefix, "zooplanktontraits")
        self.assertEqual(result.identifier, "c_13")

    def test_converter_cannot_parse_unknown_uri(self):
        """Test that converter cannot parse unknown URIs."""
        uri = "http://unknown.example.org/entity"
        result = self.converter.parse_uri(uri)

        self.assertIsNone(result)

    def test_converter_from_test_data(self):
        """Test converter with actual test data."""
        with TEST_DATA.open() as file:
            matchings = json.load(file)

        # Test first few matchings
        for matching in matchings[:3]:
            source_uri = matching["source"]
            target_uri = matching["target"]

            source_result = self.converter.parse_uri(source_uri)
            target_result = self.converter.parse_uri(target_uri)

            self.assertIsNotNone(source_result)
            self.assertIsNotNone(target_result)
            self.assertEqual(source_result.prefix, "fishtraits")
            self.assertEqual(target_result.prefix, "zooplanktontraits")


if __name__ == "__main__":
    unittest.main()
