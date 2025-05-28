import unittest
from ontoaligner.aligner.lightweight.lightweight import Lightweight, FuzzySMLightweight
from rapidfuzz import fuzz


class TestLightweightAligner(unittest.TestCase):
    def setUp(self):
        self.lightweight = Lightweight(fuzzy_sm_threshold=0.5)
        self.fuzzy_lightweight = FuzzySMLightweight(fuzzy_sm_threshold=0.7)

        # Test data
        self.source_ontology = [
            {
                "iri": "http://example.org/source1",
                "text": "Computer Science",
                "label": "Computer Science",
            },
            {
                "iri": "http://example.org/source2",
                "text": "Information Technology",
                "label": "Information Technology",
            },
        ]

        self.target_ontology = [
            {
                "iri": "http://example.org/target1",
                "text": "Computer Sciences",
                "label": "Computer Sciences",
            },
            {
                "iri": "http://example.org/target2",
                "text": "Information Tech",
                "label": "Information Tech",
            },
            {
                "iri": "http://example.org/target3",
                "text": "Software Engineering",
                "label": "Software Engineering",
            },
        ]

    def test_lightweight_initialization(self):
        """Test initialization of Lightweight aligner"""
        self.assertEqual(self.lightweight.kwargs["fuzzy_sm_threshold"], 0.5)
        self.assertEqual(str(self.lightweight), "Lightweight")

    def test_lightweight_init_retriever(self):
        """Test init_retriever method of Lightweight aligner"""
        # Should not raise any exception
        try:
            self.lightweight.init_retriever(None)
        except Exception as e:
            self.fail(f"init_retriever raised an exception: {str(e)}")

    def test_lightweight_generate(self):
        """Test generate method of Lightweight aligner"""
        # Base Lightweight generate method should return None
        result = self.lightweight.generate([self.source_ontology, self.target_ontology])
        self.assertIsNone(result)

    def test_fuzzy_lightweight_initialization(self):
        """Test initialization of FuzzySMLightweight aligner"""
        self.assertEqual(self.fuzzy_lightweight.kwargs["fuzzy_sm_threshold"], 0.7)
        self.assertEqual(str(self.fuzzy_lightweight), "Lightweight")

    def test_fuzzy_lightweight_ratio_estimate(self):
        """Test ratio_estimate method of FuzzySMLightweight aligner"""
        # Should not raise any exception and return None (as it's a placeholder)
        result = self.fuzzy_lightweight.ratio_estimate()
        self.assertIsNone(result)

    def test_fuzzy_lightweight_calculate_similarity(self):
        """Test calculate_similarity method of FuzzySMLightweight aligner"""
        # Set up ratio_estimate to use RapidFuzz ratio
        self.fuzzy_lightweight.ratio_estimate = lambda: fuzz.ratio

        # Test case 1: Exact match
        source = "Computer Science"
        candidates = ["Computer Science", "Information Tech", "Software Engineering"]
        idx, score = self.fuzzy_lightweight.calculate_similarity(source, candidates)
        self.assertEqual(idx, 0)
        self.assertEqual(score, 1.0)  # Score should be normalized between 0 and 1

        # Test case 2: Close match with different casing
        source = "computer science"
        candidates = ["Computer Science", "Information Tech", "Software Engineering"]
        idx, score = self.fuzzy_lightweight.calculate_similarity(source, candidates)
        self.assertEqual(idx, 0)
        self.assertGreater(
            score, 0.9
        )  # Should be high similarity despite case difference

        # Test case 3: Partial match
        source = "Computer"
        candidates = ["Computer Science", "Information Tech", "Software Engineering"]
        idx, score = self.fuzzy_lightweight.calculate_similarity(source, candidates)
        self.assertEqual(idx, 0)
        self.assertLess(score, 1.0)  # Should be partial match

        # Test case 4: No good match
        source = "Mathematics"
        candidates = ["Computer Science", "Information Tech", "Software Engineering"]
        idx, score = self.fuzzy_lightweight.calculate_similarity(source, candidates)
        self.assertLess(score, 0.7)  # Compare with normalized threshold

    def test_fuzzy_lightweight_generate(self):
        """Test generate method of FuzzySMLightweight aligner"""
        # Set up ratio_estimate to use RapidFuzz ratio
        self.fuzzy_lightweight.ratio_estimate = lambda: fuzz.ratio

        predictions = self.fuzzy_lightweight.generate(
            [self.source_ontology, self.target_ontology]
        )

        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)

        # Check prediction structure
        for pred in predictions:
            self.assertIsInstance(pred, dict)  # Predictions should be dictionaries
            self.assertIn("source", pred)  # Changed back to match actual output format
            self.assertIn("target", pred)  # Changed back to match actual output format
            self.assertIn("score", pred)
            self.assertIsInstance(pred["source"], str)  # Should be IRI string
            self.assertIsInstance(pred["target"], str)  # Should be IRI string
            self.assertIsInstance(pred["score"], (int, float))
            self.assertGreaterEqual(
                pred["score"], self.fuzzy_lightweight.kwargs["fuzzy_sm_threshold"]
            )

    def test_fuzzy_lightweight_threshold_filtering(self):
        """Test threshold filtering in FuzzySMLightweight aligner"""
        # Test with different thresholds
        thresholds = [0.5, 0.7, 0.9, 0.99]
        test_cases = [
            {
                "source": {
                    "iri": "http://example.org/src1",
                    "text": "Computer Science",
                    "label": "Computer Science",
                },
                "target": {
                    "iri": "http://example.org/tgt1",
                    "text": "Computer Sciences",
                    "label": "Computer Sciences",
                },
                "expected_matches": [
                    True,
                    True,
                    True,
                    False,
                ],  # Whether it should match at each threshold
            },
            {
                "source": {
                    "iri": "http://example.org/src2",
                    "text": "Information Technology",
                    "label": "Information Technology",
                },
                "target": {
                    "iri": "http://example.org/tgt2",
                    "text": "Info Tech",
                    "label": "Info Tech",
                },
                "expected_matches": [True, False, False, False],
            },
        ]

        for threshold, test_case in [(t, tc) for t in thresholds for tc in test_cases]:
            aligner = FuzzySMLightweight(fuzzy_sm_threshold=threshold)
            aligner.ratio_estimate = lambda: fuzz.ratio

            predictions = aligner.generate(
                [[test_case["source"]], [test_case["target"]]]
            )

            match_found = len(predictions) > 0
            expected_match = test_case["expected_matches"][thresholds.index(threshold)]

            self.assertEqual(
                match_found,
                expected_match,
                f"Failed with threshold {threshold} for {test_case['source']['text']} -> {test_case['target']['text']}",
            )


if __name__ == "__main__":
    unittest.main()
