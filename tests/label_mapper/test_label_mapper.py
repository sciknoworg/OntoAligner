import unittest
from sklearn.linear_model import LogisticRegression
from ontoaligner.postprocess.label_mapper import (
    LabelMapper,
    TFIDFLabelMapper,
    SBERTLabelMapper,
)


class TestLabelMapper(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.default_label_dict = {
            "yes": ["yes", "correct", "true"],
            "no": ["no", "incorrect", "false"],
        }
        self.custom_label_dict = {
            "match": ["match", "equivalent", "same"],
            "no_match": ["no match", "different", "distinct"],
        }

    def test_label_mapper_initialization(self):
        """Test the initialization of the base LabelMapper class."""
        # Test with default label dictionary
        mapper = LabelMapper()
        self.assertEqual(mapper.labels, ["yes", "no"])
        self.assertEqual(len(mapper.x_train), len(mapper.y_train))

        # Test with custom label dictionary
        mapper = LabelMapper(label_dict=self.custom_label_dict)
        self.assertEqual(mapper.labels, ["match", "no_match"])
        self.assertEqual(len(mapper.x_train), len(mapper.y_train))

    def test_label_mapper_validation(self):
        """Test the validation of predictions."""
        mapper = LabelMapper()
        # Test with valid predictions
        valid_preds = ["yes", "no"]
        mapper.validate_predicts(valid_preds)  # Should not raise any exception

        # Test with invalid predictions
        invalid_preds = ["yes", "maybe", "no"]
        with self.assertRaises(AssertionError):
            mapper.validate_predicts(invalid_preds)


class TestTFIDFLabelMapper(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.label_dict = {
            "yes": ["yes", "correct", "true"],
            "no": ["no", "incorrect", "false"],
        }
        self.classifier = LogisticRegression()
        self.mapper = TFIDFLabelMapper(
            classifier=self.classifier, ngram_range=(1, 1), label_dict=self.label_dict
        )

    def test_tfidf_mapper_initialization(self):
        """Test the initialization of TFIDFLabelMapper."""
        self.assertIsNotNone(self.mapper.model)
        self.assertEqual(self.mapper.labels, ["yes", "no"])

    def test_tfidf_mapper_fit_predict(self):
        """Test the fit and predict methods of TFIDFLabelMapper."""
        # Fit the mapper
        self.mapper.fit()

        # Test predictions
        test_inputs = ["yes", "correct", "no", "false", "maybe"]
        predictions = self.mapper.predict(test_inputs)

        # Verify predictions
        self.assertEqual(len(predictions), len(test_inputs))
        for pred in predictions:
            self.assertIn(pred.lower(), ["yes", "no"])


class TestSBERTLabelMapper(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.label_dict = {
            "yes": ["yes", "correct", "true"],
            "no": ["no", "incorrect", "false"],
        }
        self.model_id = "all-MiniLM-L12-v2"  # Using a small model for testing
        self.mapper = SBERTLabelMapper(
            model_id=self.model_id, label_dict=self.label_dict
        )

    def test_sbert_mapper_initialization(self):
        """Test the initialization of SBERTLabelMapper."""
        self.assertIsNotNone(self.mapper.embedder)
        self.assertIsNotNone(self.mapper.classifier)
        self.assertEqual(self.mapper.labels, ["yes", "no"])

    def test_sbert_mapper_fit_predict(self):
        """Test the fit and predict methods of SBERTLabelMapper."""
        # Fit the mapper
        self.mapper.fit()

        # Test predictions
        test_inputs = ["yes", "correct", "no", "false", "maybe"]
        predictions = self.mapper.predict(test_inputs)

        # Verify predictions
        self.assertEqual(len(predictions), len(test_inputs))
        for pred in predictions:
            self.assertIn(pred.lower(), ["yes", "no"])


if __name__ == "__main__":
    unittest.main()
