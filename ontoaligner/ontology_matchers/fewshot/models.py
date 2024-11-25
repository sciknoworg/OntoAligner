# -*- coding: utf-8 -*-
"""
This script defines a collection of classes that extend the FewShotRAG model, each combining a specific
retrieval model and language model (LLM) configuration. These specialized configurations are tailored
for various retrieval and generation tasks using different retrieval backends (Ada and BERT) and
LLMs (OpenAI, AutoModelDecoderRAG, MambaSSM, etc.). Each class also overrides the string representation
to identify the model configuration.
"""

from .fewshot import FewShotRAG
from ..rag.models import OpenAIRAGLLM, AutoModelDecoderRAGLLM, AutoModelDecoderRAGLLMV2, MambaSSMRAGLLM
from ..retrieval.models import AdaRetrieval, SBERTRetrieval


class LLaMALLMAdaRetrieverFSRAG(FewShotRAG):
    """
    FewShotRAG model with Ada retrieval and AutoModelDecoderRAG as the language model (LLM).
    """
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns:
            str: The class name appended to the parent class's string representation.
        """
        return super().__str__() + "-LLaMALLMAdaRetrieverFSRAG"


class LLaMALLMBERTRetrieverFSRAG(FewShotRAG):
    """
    FewShotRAG model with BERT retrieval and AutoModelDecoderRAG as the language model (LLM).
    """
    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns:
            str: The class name appended to the parent class's string representation.
        """
        return super().__str__() + "-LLaMALLMBERTRetrieverFSRAG"


class MistralLLMAdaRetrieverFSRAG(FewShotRAG):
    """
    FewShotRAG model with Ada retrieval and AutoModelDecoderRAG as the language model (LLM).
    """
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns:
            str: The class name appended to the parent class's string representation.
        """
        return super().__str__() + "-MistralLLMAdaRetrieverFSRAG"


class MistralLLMBERTRetrieverFSRAG(FewShotRAG):
    """
    FewShotRAG model with BERT retrieval and AutoModelDecoderRAG as the language model (LLM).
    """
    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns:
            str: The class name appended to the parent class's string representation.
        """
        return super().__str__() + "-MistralLLMBERTRetrieverFSRAG"


class GPTOpenAILLMAdaRetrieverFSRAG(FewShotRAG):
    """
    FewShotRAG model with Ada retrieval and OpenAIRAGLLM as the language model (LLM).
    """
    Retrieval = AdaRetrieval
    LLM = OpenAIRAGLLM

    def __str__(self):
        """
        Returns:
            str: The class name appended to the parent class's string representation.
        """
        return super().__str__() + "-GPTOpenAILLMAdaRetrieverFSRAG"


class GPTOpenAILLMBERTRetrieverFSRAG(FewShotRAG):
    """
    FewShotRAG model with BERT retrieval and OpenAIRAGLLM as the language model (LLM).
    """
    Retrieval = SBERTRetrieval
    LLM = OpenAIRAGLLM

    def __str__(self):
        """
        Returns:
            str: The class name appended to the parent class's string representation.
        """
        return super().__str__() + "-GPTOpenAILLMBERTRetrieverFSRAG"


class FalconLLMAdaRetrieverFSRAG(FewShotRAG):
    """
    FewShotRAG model with Ada retrieval and AutoModelDecoderRAGLLMV2 as the language model (LLM).
    """
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        """
        Returns:
            str: The class name appended to the parent class's string representation.
        """
        return super().__str__() + "-FalconLLMAdaRetrieverFSRAG"


class FalconLLMBERTRetrieverFSRAG(FewShotRAG):
    """
    FewShotRAG model with BERT retrieval and AutoModelDecoderRAGLLMV2 as the language model (LLM).
    """
    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        """
        Returns:
            str: The class name appended to the parent class's string representation.
        """
        return super().__str__() + "-FalconLLMBERTRetrieverFSRAG"


class VicunaLLMAdaRetrieverFSRAG(FewShotRAG):
    """
    FewShotRAG model with Ada retrieval and AutoModelDecoderRAG as the language model (LLM).
    """
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns:
            str: The class name appended to the parent class's string representation.
        """
        return super().__str__() + "-VicunaLLMAdaRetrieverFSRAG"


class VicunaLLMBERTRetrieverFSRAG(FewShotRAG):
    """
    FewShotRAG model with BERT retrieval and AutoModelDecoderRAG as the language model (LLM).
    """
    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns:
            str: The class name appended to the parent class's string representation.
        """
        return super().__str__() + "-VicunaLLMBERTRetrieverFSRAG"


class MPTLLMAdaRetrieverFSRAG(FewShotRAG):
    """
    FewShotRAG model with Ada retrieval and AutoModelDecoderRAGLLMV2 as the language model (LLM).
    """
    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        """
        Returns:
            str: The class name appended to the parent class's string representation.
        """
        return super().__str__() + "-MPTLLMAdaRetrieverFSRAG"


class MPTLLMBERTRetrieverFSRAG(FewShotRAG):
    """
    FewShotRAG model with BERT retrieval and AutoModelDecoderRAGLLMV2 as the language model (LLM).
    """
    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        """
        Returns:
            str: The class name appended to the parent class's string representation.
        """
        return super().__str__() + "-MPTLLMBERTRetrieverFSRAG"


class MambaLLMAdaRetrieverFSRAG(FewShotRAG):
    """
    FewShotRAG model with Ada retrieval and MambaSSMRAGLLM as the language model (LLM).
    """
    Retrieval = AdaRetrieval
    LLM = MambaSSMRAGLLM

    def __str__(self):
        """
        Returns:
            str: The class name appended to the parent class's string representation.
        """
        return super().__str__() + "-MambaLLMAdaRetrieverFSRAG"


class MambaLLMBERTRetrieverFSRAG(FewShotRAG):
    """
    FewShotRAG model with BERT retrieval and MambaSSMRAGLLM as the language model (LLM).
    """
    Retrieval = SBERTRetrieval
    LLM = MambaSSMRAGLLM

    def __str__(self):
        """
        Returns:
            str: The class name appended to the parent class's string representation.
        """
        return super().__str__() + "-MambaLLMBERTRetrieverFSRAG"
