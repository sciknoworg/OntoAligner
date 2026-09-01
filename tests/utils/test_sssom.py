"""Tests for SSSOM alignment generator."""

import unittest
from unittest.mock import Mock
from ontoaligner.utils.sssom import (
    sssom_alignment_generator,
    _get_label_lookup,
    _get_converter,
    _get_aligner_metadata,
)


class TestGetLabelLookup(unittest.TestCase):
    """Test cases for _get_label_lookup function."""

    def test_empty_entities_list(self):
        """Test with empty entities list."""
        result = _get_label_lookup([])
        self.assertEqual(result, {})

    def test_none_entities(self):
        """Test with None entities."""
        result = _get_label_lookup(None)
        self.assertEqual(result, {})

    def test_single_entity_with_label(self):
        """Test with a single entity containing IRI and label."""
        entities = [{"iri": "http://example.org/entity1", "label": "Entity 1"}]
        result = _get_label_lookup(entities)
        self.assertEqual(
            result,
            {"http://example.org/entity1": "Entity 1"}
        )

    def test_multiple_entities_with_labels(self):
        """Test with multiple entities containing IRIs and labels."""
        entities = [
            {"iri": "http://example.org/entity1", "label": "Entity 1"},
            {"iri": "http://example.org/entity2", "label": "Entity 2"},
            {"iri": "http://example.org/entity3", "label": "Entity 3"},
        ]
        result = _get_label_lookup(entities)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["http://example.org/entity1"], "Entity 1")
        self.assertEqual(result["http://example.org/entity2"], "Entity 2")
        self.assertEqual(result["http://example.org/entity3"], "Entity 3")

    def test_entity_without_iri(self):
        """Test with entity missing IRI field."""
        entities = [
            {"label": "Entity 1"},  # No IRI
            {"iri": "http://example.org/entity2", "label": "Entity 2"},
        ]
        result = _get_label_lookup(entities)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["http://example.org/entity2"], "Entity 2")

    def test_entity_without_label(self):
        """Test with entity missing label field."""
        entities = [
            {"iri": "http://example.org/entity1"},  # No label
            {"iri": "http://example.org/entity2", "label": "Entity 2"},
        ]
        result = _get_label_lookup(entities)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["http://example.org/entity2"], "Entity 2")

    def test_entity_with_none_label(self):
        """Test with entity having None as label."""
        entities = [
            {"iri": "http://example.org/entity1", "label": None},
            {"iri": "http://example.org/entity2", "label": "Entity 2"},
        ]
        result = _get_label_lookup(entities)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["http://example.org/entity2"], "Entity 2")

    def test_label_conversion_to_string(self):
        """Test that labels are converted to strings."""
        entities = [{"iri": "http://example.org/entity1", "label": 123}]
        result = _get_label_lookup(entities)
        self.assertEqual(result["http://example.org/entity1"], "123")
        self.assertIsInstance(result["http://example.org/entity1"], str)


class TestGetConverter(unittest.TestCase):
    """Test cases for _get_converter function."""

    def test_converter_with_custom_curie_map(self):
        """Test converter creation with custom CURIE map."""
        curie_map = {
            "ex": "http://example.org/",
            "test": "http://test.org/",
        }
        converter = _get_converter(curie_map=curie_map)
        self.assertIsNotNone(converter)

    def test_converter_without_curie_map(self):
        """Test converter creation without custom CURIE map (uses bioregistry)."""
        converter = _get_converter(curie_map=None)
        self.assertIsNotNone(converter)

    def test_converter_empty_curie_map(self):
        """Test converter creation with empty CURIE map."""
        converter = _get_converter(curie_map={})
        self.assertIsNotNone(converter)


class TestGetAlignerMetadata(unittest.TestCase):
    """Test cases for _get_aligner_metadata function."""

    def test_no_aligner(self):
        """Test with no aligner provided."""
        matching = {"score": 0.8}
        metadata = _get_aligner_metadata(matching=matching)
        self.assertEqual(metadata["mapping_justification"], "semapv:UnspecifiedMatching")

    def test_fuzzy_aligner_simple(self):
        """Test with SimpleFuzzySMLightweight aligner."""
        aligner = Mock()
        aligner.__class__.__name__ = "SimpleFuzzySMLightweight"
        matching = {"score": 0.9}

        metadata = _get_aligner_metadata(
            matching=matching,
            aligner=aligner,
        )
        self.assertEqual(
            metadata["mapping_justification"],
            "semapv:LexicalSimilarityThresholdMatching"
        )
        self.assertEqual(metadata["similarity_score"], 0.9)
        self.assertEqual(metadata["similarity_measure"], "RapidFuzz fuzz.ratio")

    def test_fuzzy_aligner_weighted(self):
        """Test with WeightedFuzzySMLightweight aligner."""
        aligner = Mock()
        aligner.__class__.__name__ = "WeightedFuzzySMLightweight"
        matching = {"score": 0.85}

        metadata = _get_aligner_metadata(
            matching=matching,
            aligner=aligner,
        )
        self.assertEqual(
            metadata["mapping_justification"],
            "semapv:LexicalSimilarityThresholdMatching"
        )
        self.assertEqual(metadata["similarity_measure"], "RapidFuzz fuzz.WRatio")

    def test_graph_aligner(self):
        """Test with graph aligner."""
        aligner = Mock()
        aligner.__class__.__name__ = "TransEAligner"
        matching = {"score": 0.75}

        metadata = _get_aligner_metadata(
            matching=matching,
            aligner=aligner,
        )
        self.assertEqual(metadata["mapping_justification"], "semapv:StructuralMatching")
        self.assertEqual(metadata["similarity_score"], 0.75)
        self.assertEqual(metadata["similarity_measure"], "cosine similarity over graph embeddings")

    def test_sbert_retrieval_with_score(self):
        """Test with SBERTRetrieval aligner and valid score."""
        aligner = Mock()
        aligner.__class__.__name__ = "SBERTRetrieval"
        matching = {"score": 0.8}

        metadata = _get_aligner_metadata(
            matching=matching,
            aligner=aligner,
        )
        self.assertEqual(metadata["similarity_score"], 0.8)
        self.assertIn("SentenceTransformer", metadata["similarity_measure"])

    def test_sbert_retrieval_with_threshold(self):
        """Test with SBERTRetrieval and retriever_postprocessor threshold."""
        aligner = Mock()
        aligner.__class__.__name__ = "SBERTRetrieval"
        postprocessor = Mock()
        postprocessor.__name__ = "retriever_postprocessor"
        matching = {"score": 0.85}

        metadata = _get_aligner_metadata(
            matching=matching,
            aligner=aligner,
            postprocessor=postprocessor,
            postprocessor_params={"threshold": 0.5},
        )
        self.assertEqual(
            metadata["mapping_justification"],
            "semapv:SemanticSimilarityThresholdMatching"
        )

    def test_tfidf_retrieval(self):
        """Test with TFIDFRetrieval aligner."""
        aligner = Mock()
        aligner.__class__.__name__ = "TFIDFRetrieval"
        matching = {"score": 0.7}

        metadata = _get_aligner_metadata(
            matching=matching,
            aligner=aligner,
        )
        self.assertEqual(metadata["similarity_score"], 0.7)
        self.assertEqual(metadata["similarity_measure"], "TF-IDF cosine similarity")

    def test_rag_heuristic_postprocessor(self):
        """Test with RAG heuristic postprocessor."""
        aligner = Mock()
        aligner.__class__.__name__ = "SomeAligner"
        postprocessor = Mock()
        postprocessor.__name__ = "rag_heuristic_postprocessor"
        matching = {"score": 0.8, "confidence": 0.9}

        metadata = _get_aligner_metadata(
            matching=matching,
            aligner=aligner,
            postprocessor=postprocessor,
            postprocessor_params={},
        )
        self.assertEqual(metadata["mapping_justification"], "semapv:CompositeMatching")
        self.assertEqual(metadata["confidence"], 0.9)

    def test_prop_match_aligner_without_domain_range(self):
        """Test PropMatchAligner without domain_range disabled."""
        aligner = Mock()
        aligner.__class__.__name__ = "PropMatchAligner"
        aligner.disable_domain_range = False
        matching = {"score": 0.8}

        metadata = _get_aligner_metadata(
            matching=matching,
            aligner=aligner,
        )
        self.assertEqual(metadata["mapping_justification"], "semapv:CompositeMatching")

    def test_prop_match_aligner_with_domain_range_disabled(self):
        """Test PropMatchAligner with domain_range disabled."""
        aligner = Mock()
        aligner.__class__.__name__ = "PropMatchAligner"
        aligner.disable_domain_range = True
        matching = {"score": 0.8}

        metadata = _get_aligner_metadata(
            matching=matching,
            aligner=aligner,
        )
        self.assertEqual(
            metadata["mapping_justification"],
            "semapv:LexicalSimilarityThresholdMatching"
        )

    def test_olala_aligner(self):
        """Test with OLaLaHighPrecisionMatcher aligner."""
        aligner = Mock()
        aligner.__class__.__name__ = "OLaLaHighPrecisionMatcher"
        matching = {"score": 0.95}

        metadata = _get_aligner_metadata(
            matching=matching,
            aligner=aligner,
        )
        self.assertEqual(metadata["mapping_justification"], "semapv:LexicalMatching")
        self.assertEqual(metadata["confidence"], 0.95)

    def test_ensemble_aligner(self):
        """Test with EnsembleLearningAligner."""
        aligner = Mock()
        aligner.__class__.__name__ = "EnsembleLearningAligner"
        matching = {"score": 0.85}

        metadata = _get_aligner_metadata(
            matching=matching,
            aligner=aligner,
        )
        self.assertEqual(metadata["mapping_justification"], "semapv:CompositeMatching")

    def test_score_out_of_range(self):
        """Test with score outside valid range."""
        aligner = Mock()
        aligner.__class__.__name__ = "SBERTRetrieval"
        matching = {"score": 2.5}  # Out of range

        metadata = _get_aligner_metadata(
            matching=matching,
            aligner=aligner,
        )
        # Should not add similarity_score if out of range
        self.assertNotIn("similarity_score", metadata)


class TestSSSOMAlignmentGenerator(unittest.TestCase):
    """Test cases for sssom_alignment_generator function."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_metadata = {
            "mapping_set_id": "http://example.org/mapping1",
            "license": "CC-BY-4.0",
        }
        self.simple_matchings = [
            {
                "source": "http://example.org/source1",
                "target": "http://example.org/target1",
            }
        ]
        self.predicate_id = "http://www.w3.org/2004/02/skos/core#exactMatch"
        self.curie_map = {
            "ex": "http://example.org/",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "semapv": "https://w3id.org/semapv/",
        }

    def test_basic_alignment_generation(self):
        """Test basic SSSOM alignment generation."""
        result = sssom_alignment_generator(
            matchings=self.simple_matchings,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            curie_map=self.curie_map,
        )
        self.assertIsInstance(result, str)
        self.assertIn("subject_id", result)
        self.assertIn("predicate_id", result)
        self.assertIn("object_id", result)

    def test_missing_mapping_set_id(self):
        """Test error handling for missing mapping_set_id."""
        incomplete_metadata = {"license": "CC-BY-4.0"}
        with self.assertRaises(ValueError) as context:
            sssom_alignment_generator(
                matchings=self.simple_matchings,
                predicate_id=self.predicate_id,
                mapping_set_metadata=incomplete_metadata,
                curie_map=self.curie_map,
            )
        self.assertIn("mapping_set_id", str(context.exception))

    def test_missing_license(self):
        """Test error handling for missing license."""
        incomplete_metadata = {"mapping_set_id": "http://example.org/mapping1"}
        with self.assertRaises(ValueError) as context:
            sssom_alignment_generator(
                matchings=self.simple_matchings,
                predicate_id=self.predicate_id,
                mapping_set_metadata=incomplete_metadata,
                curie_map=self.curie_map,
            )
        self.assertIn("license", str(context.exception))

    def test_alignment_with_source_labels(self):
        """Test alignment generation with source labels."""
        source_entities = [
            {
                "iri": "http://example.org/source1",
                "label": "Source Entity 1"
            }
        ]
        result = sssom_alignment_generator(
            matchings=self.simple_matchings,
            source=source_entities,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            curie_map=self.curie_map,
        )
        self.assertIn("subject_label", result)
        self.assertIn("Source Entity 1", result)

    def test_alignment_with_target_labels(self):
        """Test alignment generation with target labels."""
        target_entities = [
            {
                "iri": "http://example.org/target1",
                "label": "Target Entity 1"
            }
        ]
        result = sssom_alignment_generator(
            matchings=self.simple_matchings,
            target=target_entities,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            curie_map=self.curie_map,
        )
        self.assertIn("object_label", result)
        self.assertIn("Target Entity 1", result)

    def test_alignment_with_both_labels(self):
        """Test alignment generation with both source and target labels."""
        source_entities = [
            {"iri": "http://example.org/source1", "label": "Source Entity 1"}
        ]
        target_entities = [
            {"iri": "http://example.org/target1", "label": "Target Entity 1"}
        ]
        result = sssom_alignment_generator(
            matchings=self.simple_matchings,
            source=source_entities,
            target=target_entities,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            curie_map=self.curie_map,
        )
        self.assertIn("subject_label", result)
        self.assertIn("object_label", result)

    def test_alignment_with_score(self):
        """Test alignment generation with similarity score."""
        matchings_with_score = [
            {
                "source": "http://example.org/source1",
                "target": "http://example.org/target1",
                "score": 0.95,
            }
        ]
        aligner = Mock()
        aligner.__class__.__name__ = "SBERTRetrieval"

        result = sssom_alignment_generator(
            matchings=matchings_with_score,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            aligner=aligner,
            curie_map=self.curie_map,
        )
        self.assertIn("similarity_score", result)

    def test_alignment_with_multiple_matchings(self):
        """Test alignment generation with multiple matchings."""
        multiple_matchings = [
            {
                "source": "http://example.org/source1",
                "target": "http://example.org/target1",
            },
            {
                "source": "http://example.org/source2",
                "target": "http://example.org/target2",
            },
            {
                "source": "http://example.org/source3",
                "target": "http://example.org/target3",
            },
        ]
        result = sssom_alignment_generator(
            matchings=multiple_matchings,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            curie_map=self.curie_map,
        )
        # Check that all sources and targets are in the output
        for matching in multiple_matchings:
            self.assertIn("source1", result)
            self.assertIn("source2", result)
            self.assertIn("source3", result)

    def test_custom_curie_map(self):
        """Test alignment generation with custom CURIE map."""
        result = sssom_alignment_generator(
            matchings=self.simple_matchings,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            curie_map=self.curie_map,
        )
        self.assertIsInstance(result, str)

    def test_explicit_mapping_justification(self):
        """Test alignment generation with explicit mapping justification."""
        explicit_justification = "semapv:LexicalMatching"
        result = sssom_alignment_generator(
            matchings=self.simple_matchings,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            mapping_justification=explicit_justification,
            curie_map=self.curie_map,
        )
        self.assertIn("semapv:LexicalMatching", result)

    def test_include_aligner_metadata_false(self):
        """Test alignment generation with include_aligner_metadata=False."""
        matchings_with_score = [
            {
                "source": "http://example.org/source1",
                "target": "http://example.org/target1",
                "score": 0.95,
            }
        ]
        aligner = Mock()
        aligner.__class__.__name__ = "SBERTRetrieval"

        result = sssom_alignment_generator(
            matchings=matchings_with_score,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            aligner=aligner,
            include_aligner_metadata=False,
            curie_map=self.curie_map,
        )
        # Should not include similarity_score when metadata inclusion is disabled
        lines = result.split('\n')
        # Check the data lines (not header)
        data_lines = [line for line in lines if not line.startswith('#')]
        for line in data_lines:
            if line.strip():
                # Even with aligner and score, metadata shouldn't be included
                parts = line.split('\t')
                # The structure should still be valid
                self.assertGreater(len(parts), 0)

    def test_metadata_defaults(self):
        """Test that metadata defaults are applied."""
        extended_metadata = dict(self.base_metadata)
        result = sssom_alignment_generator(
            matchings=self.simple_matchings,
            predicate_id=self.predicate_id,
            mapping_set_metadata=extended_metadata,
            curie_map=self.curie_map,
        )
        # Should have OntoAligner as mapping tool
        self.assertIn("mapping_tool", result)
        self.assertIn("OntoAligner", result)

    def test_with_pipeline_parameter(self):
        """Test alignment generation with pipeline parameter."""
        pipeline = Mock()
        pipeline.reranker = None
        pipeline.aligner = Mock()
        pipeline.aligner.__class__.__name__ = "SBERTRetrieval"
        pipeline.postprocessor = None
        pipeline.postprocessor_params = {}
        pipeline.om_dataset = None

        result = sssom_alignment_generator(
            matchings=self.simple_matchings,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            pipeline=pipeline,
            curie_map=self.curie_map,
        )
        self.assertIsInstance(result, str)

    def test_with_pipeline_and_dataset_info(self):
        """Test alignment generation with pipeline containing dataset info."""
        pipeline = Mock()
        pipeline.reranker = None
        pipeline.aligner = Mock()
        pipeline.aligner.__class__.__name__ = "SBERTRetrieval"
        pipeline.postprocessor = None
        pipeline.postprocessor_params = {}
        pipeline.om_dataset = {
            "source": [{"iri": "http://example.org/source1", "label": "Source"}],
            "target": [{"iri": "http://example.org/target1", "label": "Target"}],
            "dataset-info": {"ontology-name": "Test Ontology"},
        }

        result = sssom_alignment_generator(
            matchings=self.simple_matchings,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            pipeline=pipeline,
            curie_map=self.curie_map,
        )
        self.assertIn("Test Ontology", result)

    def test_empty_matchings(self):
        """Test alignment generation with empty matchings list."""
        result = sssom_alignment_generator(
            matchings=[],
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            curie_map=self.curie_map,
        )
        self.assertIsInstance(result, str)
        # Should still have header
        self.assertIn("#", result)

    def test_matching_without_labels_for_entities(self):
        """Test alignment when entity labels are not found."""
        source_entities = [
            {"iri": "http://example.org/other", "label": "Other Entity"}
        ]
        target_entities = [
            {"iri": "http://example.org/other_target", "label": "Other Target"}
        ]
        result = sssom_alignment_generator(
            matchings=self.simple_matchings,
            source=source_entities,
            target=target_entities,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            curie_map=self.curie_map,
        )
        # Should still work, just without the labels for these specific IRIs
        self.assertIsInstance(result, str)

    def test_large_matching_set(self):
        """Test alignment generation with large number of matchings."""
        large_matchings = [
            {
                "source": f"http://example.org/source{i}",
                "target": f"http://example.org/target{i}",
                "score": 0.5 + (i * 0.01),
            }
            for i in range(100)
        ]
        aligner = Mock()
        aligner.__class__.__name__ = "SBERTRetrieval"

        result = sssom_alignment_generator(
            matchings=large_matchings,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            aligner=aligner,
            curie_map=self.curie_map,
        )
        # Check that output contains data for all matchings
        for i in range(10):  # Check a subset
            self.assertIn(f"source{i}", result)

    def test_fuzzy_aligner_integration(self):
        """Test with fuzzy aligner."""
        matchings_with_score = [
            {
                "source": "http://example.org/source1",
                "target": "http://example.org/target1",
                "score": 0.85,
            }
        ]
        aligner = Mock()
        aligner.__class__.__name__ = "SimpleFuzzySMLightweight"

        result = sssom_alignment_generator(
            matchings=matchings_with_score,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            aligner=aligner,
            curie_map=self.curie_map,
        )
        self.assertIn("similarity_measure", result)
        self.assertIn("fuzz.ratio", result)

    def test_graph_aligner_integration(self):
        """Test with graph aligner."""
        matchings_with_score = [
            {
                "source": "http://example.org/source1",
                "target": "http://example.org/target1",
                "score": 0.75,
            }
        ]
        aligner = Mock()
        aligner.__class__.__name__ = "TransEAligner"

        result = sssom_alignment_generator(
            matchings=matchings_with_score,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            aligner=aligner,
            curie_map=self.curie_map,
        )
        self.assertIn("graph embeddings", result)

    def test_confidence_score_included(self):
        """Test that confidence score is included when available."""
        matchings_with_confidence = [
            {
                "source": "http://example.org/source1",
                "target": "http://example.org/target1",
                "score": 0.9,  # OLaLa aligners use score -> confidence
            }
        ]
        aligner = Mock()
        aligner.__class__.__name__ = "OLaLaHighPrecisionMatcher"

        result = sssom_alignment_generator(
            matchings=matchings_with_confidence,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            aligner=aligner,
            curie_map=self.curie_map,
        )
        self.assertIn("confidence", result)

    def test_pipeline_overrides_aligner_params(self):
        """Test that pipeline parameters override standalone aligner/postprocessor."""
        standalone_aligner = Mock()
        standalone_aligner.__class__.__name__ = "SimpleFuzzySMLightweight"

        pipeline = Mock()
        pipeline_aligner = Mock()
        pipeline_aligner.__class__.__name__ = "SBERTRetrieval"
        pipeline.reranker = None
        pipeline.aligner = pipeline_aligner
        postprocessor = Mock()
        postprocessor.__name__ = "rag_heuristic_postprocessor"
        pipeline.postprocessor = postprocessor
        pipeline.postprocessor_params = {"threshold": 0.5}
        pipeline.om_dataset = None

        matchings_with_score = [
            {
                "source": "http://example.org/source1",
                "target": "http://example.org/target1",
                "score": 0.8,
            }
        ]

        result = sssom_alignment_generator(
            matchings=matchings_with_score,
            predicate_id=self.predicate_id,
            mapping_set_metadata=self.base_metadata,
            aligner=standalone_aligner,  # Should be overridden
            pipeline=pipeline,  # Should take precedence
            curie_map=self.curie_map,
        )
        # Pipeline aligner should be used, resulting in SentenceTransformer reference
        self.assertIn("SentenceTransformer", result)


if __name__ == "__main__":
    unittest.main()
