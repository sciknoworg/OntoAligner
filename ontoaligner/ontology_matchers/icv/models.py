# -*- coding: utf-8 -*-
"""
Script for integrating ICV-based language models with various retrieval mechanisms.

This script defines classes that combine different LLM and retrieval model pairings with
ICV-based language modeling architectures. Each class pairs a specific retrieval model
(e.g., AdaRetrieval, BERTRetrieval) with an LLM model variant (e.g., AutoModelDecoderICVLLM,
AutoModelDecoderICVLLMV2) for enhanced ontology matching and retrieval-based NLP tasks.
"""

from .icv import ICV, AutoModelDecoderICVLLM, AutoModelDecoderICVLLMV2
from ..retrieval.models import AdaRetrieval, SBERTRetrieval


class LLaMALLMAdaRetrieverICVRAG(ICV):
    """
    Class for pairing LLaMA-based LLM with AdaRetrieval for ICV-based ontology matching.
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
    """

    Retrieval = SBERTRetrieval
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
    """

    Retrieval = SBERTRetrieval
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
    """

    Retrieval = SBERTRetrieval
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
    """

    Retrieval = SBERTRetrieval
    LLM = AutoModelDecoderICVLLMV2

    def __str__(self):
        """
        Returns a string representation of the class instance, indicating the LLM-Retrieval pairing.

        Returns:
            str
                String format representing the class and the paired models.
        """
        return super().__str__() + "-MPTLLMBERTRetrieverICVRAG"
