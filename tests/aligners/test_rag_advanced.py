import pytest
import torch
from unittest.mock import MagicMock
from ontoaligner.aligner.rag.rag import (
    RAGBasedDecoderLLMArch,
    OpenAIRAGLLM,
)


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [0, 1]
    tokenizer.decode.return_value = "test"
    tokenizer.eos_token_id = 2
    return tokenizer


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.generate.return_value = MagicMock(
        scores=[torch.rand(1, 10)], sequences=torch.tensor([[1, 2, 3]])
    )
    return model


def test_model_initialization_parameters():
    """Test model initialization with different parameter combinations."""
    # Test with minimal parameters
    model = RAGBasedDecoderLLMArch()
    assert model is not None
    assert "yes" in model.ANSWER_SET
    assert "no" in model.ANSWER_SET

    # Test with custom answer set
    custom_answers = {"yes": ["positive", "affirmative"], "no": ["negative", "false"]}
    model = RAGBasedDecoderLLMArch(answer_set=custom_answers)
    assert model.ANSWER_SET == custom_answers

    # Test with device specification
    model = RAGBasedDecoderLLMArch(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    assert "device" in model.kwargs


def test_model_string_representations():
    """Test string representations of different model types."""
    models = [
        (RAGBasedDecoderLLMArch(), "RAGBasedDecoderLLMArch"),
        (OpenAIRAGLLM(), "RAGBasedOpenAILLMArch-OpenAILLM"),
    ]

    for model, expected_str in models:
        assert str(model) == expected_str


def test_model_generation_with_different_inputs(mock_tokenizer, mock_model):
    """Test model generation with various input types."""
    model = RAGBasedDecoderLLMArch()
    model.tokenizer = mock_tokenizer
    model.model = mock_model
    model.kwargs = {"max_new_tokens": 10}

    # Test with different input types
    inputs = [
        {"input_ids": torch.tensor([[1, 2, 3]])},
        {
            "input_ids": torch.tensor([[4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        },
        {
            "input_ids": torch.tensor([[7, 8, 9]]),
            "token_type_ids": torch.tensor([[0, 0, 0]]),
        },
    ]

    for input_data in inputs:
        output = model.generate_for_llm(input_data)
        assert output is not None
        assert hasattr(output, "scores")
        assert hasattr(output, "sequences")


def test_openai_model_response_handling():
    """Test OpenAI model's handling of different response formats."""
    model = OpenAIRAGLLM()

    # Test various response formats
    test_responses = [
        "Yes, these concepts are equivalent.",
        "No, these are different concepts.",
        "These concepts appear to be the same.",
        "The concepts are not related.",
        "Based on the context, yes.",
        "Cannot determine the relationship.",
    ]

    mock_responses = []
    for response in test_responses:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = response
        mock_responses.append(mock_response)

    sequences, probas = model.post_processor(mock_responses)

    assert len(sequences) == len(test_responses)
    assert all(s in ["yes", "no"] for s in sequences)
    assert all(0 <= p <= 1 for p in probas)


def test_model_error_handling():
    """Test model's error handling capabilities."""
    model = RAGBasedDecoderLLMArch()

    # Test with invalid input
    with pytest.raises(Exception):
        model.generate(None)

    # Test with empty input
    with pytest.raises(Exception):
        model.generate([])

    # Test with malformed input
    with pytest.raises(Exception):
        model.generate([{"invalid": "data"}])
