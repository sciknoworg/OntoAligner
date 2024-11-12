# -*- coding: utf-8 -*-
"""
This script defines a collection of classes that extend the FewShotRAG model, each combining a specific
retrieval model and language model (LLM) configuration. These specialized configurations are tailored
for various retrieval and generation tasks using different retrieval backends (Ada and BERT) and
LLMs (OpenAI, AutoModelDecoderRAG, MambaSSM, etc.). Each class also overrides the string representation
to identify the model configuration.

Classes:
- LLaMALLMAdaRetrieverFSRAG: Uses Ada for retrieval and AutoModelDecoderRAG for LLM with FewShotRAG.
- LLaMALLMBERTRetrieverFSRAG: Uses BERT for retrieval and AutoModelDecoderRAG for LLM with FewShotRAG.
- MistralLLMAdaRetrieverFSRAG: Uses Ada for retrieval and AutoModelDecoderRAG for LLM with FewShotRAG.
- MistralLLMBERTRetrieverFSRAG: Uses BERT for retrieval and AutoModelDecoderRAG for LLM with FewShotRAG.
- GPTOpenAILLMAdaRetrieverFSRAG: Uses Ada for retrieval and OpenAIRAGLLM for LLM with FewShotRAG.
- GPTOpenAILLMBERTRetrieverFSRAG: Uses BERT for retrieval and OpenAIRAGLLM for LLM with FewShotRAG.
- FalconLLMAdaRetrieverFSRAG: Uses Ada for retrieval and AutoModelDecoderRAGLLMV2 for LLM with FewShotRAG.
- FalconLLMBERTRetrieverFSRAG: Uses BERT for retrieval and AutoModelDecoderRAGLLMV2 for LLM with FewShotRAG.
- VicunaLLMAdaRetrieverFSRAG: Uses Ada for retrieval and AutoModelDecoderRAG for LLM with FewShotRAG.
- VicunaLLMBERTRetrieverFSRAG: Uses BERT for retrieval and AutoModelDecoderRAG for LLM with FewShotRAG.
- MPTLLMAdaRetrieverFSRAG: Uses Ada for retrieval and AutoModelDecoderRAGLLMV2 for LLM with FewShotRAG.
- MPTLLMBERTRetrieverFSRAG: Uses BERT for retrieval and AutoModelDecoderRAGLLMV2 for LLM with FewShotRAG.
- MambaLLMAdaRetrieverFSRAG: Uses Ada for retrieval and MambaSSMRAGLLM for LLM with FewShotRAG.
- MambaLLMBERTRetrieverFSRAG: Uses BERT for retrieval and MambaSSMRAGLLM for LLM with FewShotRAG.

Each class inherits from FewShotRAG and defines a unique combination of Retrieval and LLM models.
"""

from .fewshot import FewShotRAG
from ..rag.models import OpenAIRAGLLM, AutoModelDecoderRAGLLM, AutoModelDecoderRAGLLMV2, MambaSSMRAGLLM
from ..retrieval.models import AdaRetrieval, BERTRetrieval


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
    Retrieval = BERTRetrieval
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
    Retrieval = BERTRetrieval
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
    Retrieval = BERTRetrieval
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
    Retrieval = BERTRetrieval
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
    Retrieval = BERTRetrieval
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
    Retrieval = BERTRetrieval
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
    Retrieval = BERTRetrieval
    LLM = MambaSSMRAGLLM

    def __str__(self):
        """
        Returns:
            str: The class name appended to the parent class's string representation.
        """
        return super().__str__() + "-MambaLLMBERTRetrieverFSRAG"
