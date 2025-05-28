import unittest
import torch
from unittest.mock import MagicMock, patch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ontoaligner.aligner.rag.rag import (
    RAGBasedDecoderLLMArch,
    RAGBasedOpenAILLMArch,
    RAG,
    AutoModelDecoderRAGLLM,
    AutoModelDecoderRAGLLMV2,
)


class TestRAGBasedDecoderLLMArch(unittest.TestCase):
    def setUp(self):
        self.model = RAGBasedDecoderLLMArch()
        # Mock tokenizer
        self.model.tokenizer = MagicMock()
        self.model.tokenizer.encode.return_value = [0, 1]  # Mock tokenizer output
        self.model.tokenizer.decode.return_value = "test"  # Mock decoding

        # Custom answer set for testing
        self.test_answer_set = {
            "yes": ["yes", "correct", "true"],
            "no": ["no", "incorrect", "false"],
        }

    def test_initialization(self):
        """Test initialization with default and custom answer sets"""
        # Test default initialization
        model = RAGBasedDecoderLLMArch()
        self.assertTrue(all(key in model.ANSWER_SET for key in ["yes", "no"]))

        # Test custom answer set
        model = RAGBasedDecoderLLMArch(answer_set=self.test_answer_set)
        self.assertEqual(model.ANSWER_SET, self.test_answer_set)

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_load(self, mock_model_class, mock_tokenizer_class):
        """Test model loading and token ID initialization"""
        # Set up mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [0, 1]
        mock_tokenizer_class.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        # Set up model attributes
        self.model.model = AutoModelForCausalLM
        self.model.tokenizer = AutoTokenizer
        self.model.kwargs = {
            "device": "cpu",
            "huggingface_access_token": "dummy_token",
            "device_map": None,
        }

        self.model.load("dummy_path")

        # Verify tokenizer and model were loaded
        mock_tokenizer_class.assert_called_once_with("dummy_path")
        mock_model_class.assert_called_once_with("dummy_path", token="dummy_token")

    def test_check_answer_set_tokenizer(self):
        """Test answer set tokenizer validation"""
        # Mock tokenizer behavior
        self.model.tokenizer = MagicMock()
        # Mock encode to return exactly 2 tokens for valid input
        self.model.tokenizer.encode.return_value = [0, 1]  # Mock valid tokenization
        self.model.tokenizer.decode.side_effect = lambda x: (
            "test" if x == [0, 1] else "other"
        )

        # Mock the answer set
        self.model.ANSWER_SET = {"yes": ["test"], "no": ["test2"]}

        # Test with valid input that should return exactly 2 tokens
        self.model.tokenizer.input_ids = [0, 1]  # Mock input_ids property
        self.model.tokenizer.return_value = MagicMock(
            input_ids=[0, 1]
        )  # Mock tokenizer call result
        result = self.model.check_answer_set_tokenizer("test")
        self.assertTrue(
            result
        )  # Should return True since tokenization returns exactly 2 tokens

    @patch("torch.no_grad")
    def test_generate_for_llm(self, mock_no_grad):
        """Test LLM generation"""
        self.model.model = MagicMock()
        self.model.tokenizer = MagicMock()
        self.model.kwargs = {"max_new_tokens": 10}

        test_input = {"input_ids": torch.tensor([[1, 2, 3]])}
        self.model.generate_for_llm(test_input)

        # Verify model.generate was called
        self.model.model.generate.assert_called_once()


class TestRAGBasedOpenAILLMArch(unittest.TestCase):
    def setUp(self):
        self.model = RAGBasedOpenAILLMArch()
        # Set up default answer set
        self.model.ANSWER_SET = {
            "yes": ["yes", "correct", "true"],
            "no": ["no", "incorrect", "false"],
        }

    def test_initialization(self):
        """Test basic initialization"""
        self.assertEqual(str(self.model), "RAGBasedOpenAILLMArch")
        self.assertIsNotNone(self.model.ANSWER_SET)
        self.assertTrue(all(key in self.model.ANSWER_SET for key in ["yes", "no"]))
        self.assertIsInstance(self.model.ANSWER_SET, dict)

    def test_post_processor(self):
        """Test post-processing of generated texts"""
        # Create mock OpenAI response objects
        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock()]
        mock_response1.choices[0].message.content = "Yes, that's correct."

        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock()]
        mock_response2.choices[0].message.content = "No, that's wrong."

        test_responses = [mock_response1, mock_response2]
        results = self.model.post_processor(test_responses)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(test_responses))


class TestRAG(unittest.TestCase):
    @patch("ontoaligner.aligner.rag.rag.RAGBasedDecoderLLMArch")
    def setUp(self, mock_llm_class):
        # Create mock configs
        self.retriever_config = {"param1": "value1"}
        self.llm_config = {"param2": "value2"}

        # Create mock Retrieval and LLM instances
        self.mock_retrieval = MagicMock()
        self.mock_llm = MagicMock()

        # Set up the mock LLM class
        mock_llm_class.return_value = self.mock_llm

        # Create a mock for the base class initialization
        with patch("ontoaligner.base.model.BaseOMModel.__init__") as mock_base_init:
            mock_base_init.return_value = None
            # Initialize RAG with mock configs and components
            with patch.object(RAG, "Retrieval", return_value=self.mock_retrieval):
                with patch.object(RAG, "LLM", return_value=self.mock_llm):
                    self.rag = RAG(
                        retriever_config=self.retriever_config,
                        llm_config=self.llm_config,
                    )
                    # Set up the kwargs manually since we mocked the base init
                    self.rag.kwargs = {
                        "retriever-config": self.retriever_config,
                        "llm-config": self.llm_config,
                    }

    # def test_initialization(self):
    #     """Test RAG initialization with configs"""
    #     self.assertEqual(self.rag.kwargs["retriever-config"], self.retriever_config)
    #     self.assertEqual(self.rag.kwargs["llm-config"], self.llm_config)

    # @patch("ontoaligner.aligner.rag.rag.RAGBasedDecoderLLMArch")
    # def test_load(self, mock_llm_class):
    #     """Test model loading"""
    #     # Set up mock LLM instance
    #     mock_llm = MagicMock()
    #     mock_llm_class.return_value = mock_llm

    #     # Set up mock Retrieval
    #     self.rag.Retrieval = MagicMock()
    #     self.rag.LLM = mock_llm_class

    #     self.rag.load("llm_path", "ir_path")

    #     # Verify both retrieval and LLM models are loaded
    #     self.rag.Retrieval.return_value.load.assert_called_once_with("ir_path")
    #     mock_llm.load.assert_called_once_with("llm_path")

    # def test_generate(self):
    #     """Test the generate method"""
    #     # Mock necessary components
    #     ir_output = "ir_output"
    #     llm_output = "llm_output"
    #     self.rag.ir_generate = MagicMock(return_value=ir_output)
    #     self.rag.llm_generate = MagicMock(return_value=llm_output)
    #     self.rag.kwargs = {
    #         "retriever-config": {"threshold": 0.5},
    #         "llm-config": self.llm_config,
    #     }

    #     input_data = ["test_input"]
    #     result = self.rag.generate(input_data)

    #     # Verify the generation pipeline
    #     self.rag.ir_generate.assert_called_once_with(input_data)
    #     self.rag.llm_generate.assert_called_once()

    #     # Check the result structure
    #     self.assertEqual(
    #         result, [{"ir-outputs": ir_output}, {"llm-output": llm_output}]
    #     )


class TestAutoModelDecoderRAGLLM(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelDecoderRAGLLM()

    # def test_initialization(self):
    #     """Test initialization and attributes"""
    #     self.assertEqual(
    #         str(self.model), "RAGBasedDecoderLLMArch-AutoModel"
    #     )  # Updated to match actual implementation
    #     self.assertEqual(self.model.tokenizer, AutoTokenizer)
    #     self.assertEqual(self.model.model, AutoModelForCausalLM)


class TestAutoModelDecoderRAGLLMV2(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelDecoderRAGLLMV2()

    def test_initialization(self):
        """Test initialization and attributes"""
        self.assertEqual(
            str(self.model), "RAGBasedDecoderLLMArch-AutoModelV2"
        )  # Updated to match actual implementation
        self.assertEqual(self.model.tokenizer, AutoTokenizer)
        self.assertEqual(self.model.model, AutoModelForCausalLM)

    @patch("torch.no_grad")
    def test_get_probas_yes_no(self, mock_no_grad):
        """Test probability calculation for yes/no answers"""
        # Mock outputs with scores
        mock_outputs = MagicMock()
        mock_outputs.scores = [torch.rand(1, 10)]  # Random scores for testing

        self.model.answer_sets_token_id = {"yes": [1, 2, 3], "no": [4, 5, 6]}

        probas = self.model.get_probas_yes_no(mock_outputs)
        self.assertIsInstance(probas, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
