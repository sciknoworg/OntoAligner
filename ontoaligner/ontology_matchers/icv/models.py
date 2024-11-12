# -*- coding: utf-8 -*-
"""
Script for integrating ICV-based language models with various retrieval mechanisms.

This script defines classes that combine different LLM and retrieval model pairings with
ICV-based language modeling architectures. Each class pairs a specific retrieval model
(e.g., AdaRetrieval, BERTRetrieval) with an LLM model variant (e.g., AutoModelDecoderICVLLM,
AutoModelDecoderICVLLMV2) for enhanced ontology matching and retrieval-based NLP tasks.

Classes:
- LLaMALLMAdaRetrieverICVRAG: Combines LLaMA-based LLM with AdaRetrieval.
- LLaMALLMBERTRetrieverICVRAG: Combines LLaMA-based LLM with BERTRetrieval.
- FalconLLMAdaRetrieverICVRAG: Combines Falcon-based LLM with AdaRetrieval.
- FalconLLMBERTRetrieverICVRAG: Combines Falcon-based LLM with BERTRetrieval.
- VicunaLLMAdaRetrieverICVRAG: Combines Vicuna-based LLM with AdaRetrieval.
- VicunaLLMBERTRetrieverICVRAG: Combines Vicuna-based LLM with BERTRetrieval.
- MPTLLMAdaRetrieverICVRAG: Combines MPT-based LLM with AdaRetrieval.
- MPTLLMBERTRetrieverICVRAG: Combines MPT-based LLM with BERTRetrieval.
"""

from .icv import ICV, AutoModelDecoderICVLLM, AutoModelDecoderICVLLMV2
from ..retrieval.models import AdaRetrieval, BERTRetrieval


class LLaMALLMAdaRetrieverICVRAG(ICV):
    """
    Class for pairing LLaMA-based LLM with AdaRetrieval for ICV-based ontology matching.

    This class combines the LLaMA language model with the AdaRetrieval model, using ICV-based
adapters to fine-tune the embeddings for improved ontology matching and retrieval tasks.

    Attributes:
        Retrieval : type
            Specifies AdaRetrieval as the retrieval mechanism.
        LLM : type
            Specifies AutoModelDecoderICVLLM as the language model architecture.
    """

    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderICVLLM

    def __str__(self):
        """
        Returns a string representation of the class instance, indicating the LLM-Retrieval pairing.

        Returns:
            str
                String format representing the class and the paired models.
        """
        return super().__str__() + "-LLaMALLMAdaRetrieverICVRAG"


class LLaMALLMBERTRetrieverICVRAG(ICV):
    """
    Class for pairing LLaMA-based LLM with BERTRetrieval for ICV-based ontology matching.

    This class combines the LLaMA language model with BERTRetrieval, using ICV-based adapters
    to fine-tune the embeddings for improved ontology matching and retrieval tasks.

    Attributes:
        Retrieval : type
            Specifies BERTRetrieval as the retrieval mechanism.
        LLM : type
            Specifies AutoModelDecoderICVLLM as the language model architecture.
    """

    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderICVLLM

    def __str__(self):
        """
        Returns a string representation of the class instance, indicating the LLM-Retrieval pairing.

        Returns:
            str
                String format representing the class and the paired models.
        """
        return super().__str__() + "-LLaMALLMBERTRetrieverICVRAG"


class FalconLLMAdaRetrieverICVRAG(ICV):
    """
    Class for pairing Falcon-based LLM with AdaRetrieval for ICV-based ontology matching.

    This class combines the Falcon language model with the AdaRetrieval model, using ICV-based
adapters to fine-tune the embeddings for improved ontology matching and retrieval tasks.

    Attributes:
        Retrieval : type
            Specifies AdaRetrieval as the retrieval mechanism.
        LLM : type
            Specifies AutoModelDecoderICVLLMV2 as the language model architecture.
    """

    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderICVLLMV2

    def __str__(self):
        """
        Returns a string representation of the class instance, indicating the LLM-Retrieval pairing.

        Returns:
            str
                String format representing the class and the paired models.
        """
        return super().__str__() + "-FalconLLMAdaRetrieverICVRAG"


class FalconLLMBERTRetrieverICVRAG(ICV):
    """
    Class for pairing Falcon-based LLM with BERTRetrieval for ICV-based ontology matching.

    This class combines the Falcon language model with BERTRetrieval, using ICV-based adapters
    to fine-tune the embeddings for improved ontology matching and retrieval tasks.

    Attributes:
        Retrieval : type
            Specifies BERTRetrieval as the retrieval mechanism.
        LLM : type
            Specifies AutoModelDecoderICVLLMV2 as the language model architecture.
    """

    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderICVLLMV2

    def __str__(self):
        """
        Returns a string representation of the class instance, indicating the LLM-Retrieval pairing.

        Returns:
            str
                String format representing the class and the paired models.
        """
        return super().__str__() + "-FalconLLMBERTRetrieverICVRAG"


class VicunaLLMAdaRetrieverICVRAG(ICV):
    """
    Class for pairing Vicuna-based LLM with AdaRetrieval for ICV-based ontology matching.

    This class combines the Vicuna language model with the AdaRetrieval model, using ICV-based
    adapters to fine-tune the embeddings for improved ontology matching and retrieval tasks.

    Attributes:
        Retrieval : type
            Specifies AdaRetrieval as the retrieval mechanism.
        LLM : type
            Specifies AutoModelDecoderICVLLM as the language model architecture.
    """

    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderICVLLM

    def __str__(self):
        """
        Returns a string representation of the class instance, indicating the LLM-Retrieval pairing.

        Returns:
            str
                String format representing the class and the paired models.
        """
        return super().__str__() + "-VicunaLLMAdaRetrieverICVRAG"


class VicunaLLMBERTRetrieverICVRAG(ICV):
    """
    Class for pairing Vicuna-based LLM with BERTRetrieval for ICV-based ontology matching.

    This class combines the Vicuna language model with BERTRetrieval, using ICV-based adapters
    to fine-tune the embeddings for improved ontology matching and retrieval tasks.

    Attributes:
        Retrieval : type
            Specifies BERTRetrieval as the retrieval mechanism.
        LLM : type
            Specifies AutoModelDecoderICVLLM as the language model architecture.
    """

    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderICVLLM

    def __str__(self):
        """
        Returns a string representation of the class instance, indicating the LLM-Retrieval pairing.

        Returns:
            str
                String format representing the class and the paired models.
        """
        return super().__str__() + "-VicunaLLMBERTRetrieverICVRAG"


class MPTLLMAdaRetrieverICVRAG(ICV):
    """
    Class for pairing MPT-based LLM with AdaRetrieval for ICV-based ontology matching.

    This class combines the MPT language model with the AdaRetrieval model, using ICV-based
    adapters to fine-tune the embeddings for improved ontology matching and retrieval tasks.

    Attributes:
        Retrieval : type
            Specifies AdaRetrieval as the retrieval mechanism.
        LLM : type
            Specifies AutoModelDecoderICVLLMV2 as the language model architecture.
    """

    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderICVLLMV2

    def __str__(self):
        """
        Returns a string representation of the class instance, indicating the LLM-Retrieval pairing.

        Returns:
            str
                String format representing the class and the paired models.
        """
        return super().__str__() + "-MPTLLMAdaRetrieverICVRAG"


class MPTLLMBERTRetrieverICVRAG(ICV):
    """
    Class for pairing MPT-based LLM with BERTRetrieval for ICV-based ontology matching.

    This class combines the MPT language model with BERTRetrieval, using ICV-based adapters
    to fine-tune the embeddings for improved ontology matching and retrieval tasks.

    Attributes:
        Retrieval : type
            Specifies BERTRetrieval as the retrieval mechanism.
        LLM : type
            Specifies AutoModelDecoderICVLLMV2 as the language model architecture.
    """

    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderICVLLMV2

    def __str__(self):
        """
        Returns a string representation of the class instance, indicating the LLM-Retrieval pairing.

        Returns:
            str
                String format representing the class and the paired models.
        """
        return super().__str__() + "-MPTLLMBERTRetrieverICVRAG"
