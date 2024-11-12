# -*- coding: utf-8 -*-
"""
This script defines a series of Retrieval-Augmented Generation (RAG) classes that combine different retrieval models
and language models (LLMs). Each class specializes in pairing a specific retrieval model (e.g., AdaRetrieval, BERTRetrieval)
with a specific language model (e.g., AutoModelDecoderRAGLLM, OpenAIRAGLLM). These classes are designed to perform
retrieval-augmented generation tasks for various configurations of models.
"""

from .rag import RAG, OpenAIRAGLLM, AutoModelDecoderRAGLLMV2, MambaSSMRAGLLM, AutoModelDecoderRAGLLM
from ..retrieval.models import AdaRetrieval, BERTRetrieval


class LLaMALLMAdaRetrieverRAG(RAG):
    """
    LLaMALLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    This class provides functionality for performing retrieval-augmented generation (RAG) using AdaRetrieval for the
    retrieval step and AutoModelDecoderRAGLLM for the language modeling step.

    Attributes:
        Retrieval (class): The retrieval model used by this class, AdaRetrieval.
        LLM (class): The language model used by this class, AutoModelDecoderRAGLLM.
    """

    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns a string representation of the LLaMALLMAdaRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the LLaMALLMAdaRetrieverRAG class.
        """
        return super().__str__() + "-LLaMALLMAdaRetrieverRAG"


class LLaMALLMBERTRetrieverRAG(RAG):
    """
    LLaMALLMBERTRetrieverRAG class combines the BERTRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    This class provides functionality for performing retrieval-augmented generation (RAG) using BERTRetrieval for the
    retrieval step and AutoModelDecoderRAGLLM for the language modeling step.

    Attributes:
        Retrieval (class): The retrieval model used by this class, BERTRetrieval.
        LLM (class): The language model used by this class, AutoModelDecoderRAGLLM.
    """

    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns a string representation of the LLaMALLMBERTRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the LLaMALLMBERTRetrieverRAG class.
        """
        return super().__str__() + "-LLaMALLMBERTRetrieverRAG"


class MistralLLMAdaRetrieverRAG(RAG):
    """
    MistralLLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    This class provides functionality for performing retrieval-augmented generation (RAG) using AdaRetrieval for the
    retrieval step and AutoModelDecoderRAGLLM for the language modeling step.

    Attributes:
        Retrieval (class): The retrieval model used by this class, AdaRetrieval.
        LLM (class): The language model used by this class, AutoModelDecoderRAGLLM.
    """

    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns a string representation of the MistralLLMAdaRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the MistralLLMAdaRetrieverRAG class.
        """
        return super().__str__() + "-MistralLLMAdaRetrieverRAG"


class MistralLLMBERTRetrieverRAG(RAG):
    """
    MistralLLMBERTRetrieverRAG class combines the BERTRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    This class provides functionality for performing retrieval-augmented generation (RAG) using BERTRetrieval for the
    retrieval step and AutoModelDecoderRAGLLM for the language modeling step.

    Attributes:
        Retrieval (class): The retrieval model used by this class, BERTRetrieval.
        LLM (class): The language model used by this class, AutoModelDecoderRAGLLM.
    """

    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns a string representation of the MistralLLMBERTRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the MistralLLMBERTRetrieverRAG class.
        """
        return super().__str__() + "-MistralLLMBERTRetrieverRAG"


class GPTOpenAILLMAdaRetrieverRAG(RAG):
    """
    GPTOpenAILLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the OpenAIRAGLLM language model.
    This class provides functionality for performing retrieval-augmented generation (RAG) using AdaRetrieval for the
    retrieval step and OpenAIRAGLLM for the language modeling step.

    Attributes:
        Retrieval (class): The retrieval model used by this class, AdaRetrieval.
        LLM (class): The language model used by this class, OpenAIRAGLLM.
    """

    Retrieval = AdaRetrieval
    LLM = OpenAIRAGLLM

    def __str__(self):
        """
        Returns a string representation of the GPTOpenAILLMAdaRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the GPTOpenAILLMAdaRetrieverRAG class.
        """
        return super().__str__() + "-GPTOpenAILLMAdaRetrieverRAG"


class GPTOpenAILLMBERTRetrieverRAG(RAG):
    """
    GPTOpenAILLMBERTRetrieverRAG class combines the BERTRetrieval retrieval model with the OpenAIRAGLLM language model.
    This class provides functionality for performing retrieval-augmented generation (RAG) using BERTRetrieval for the
    retrieval step and OpenAIRAGLLM for the language modeling step.

    Attributes:
        Retrieval (class): The retrieval model used by this class, BERTRetrieval.
        LLM (class): The language model used by this class, OpenAIRAGLLM.
    """

    Retrieval = BERTRetrieval
    LLM = OpenAIRAGLLM

    def __str__(self):
        """
        Returns a string representation of the GPTOpenAILLMBERTRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the GPTOpenAILLMBERTRetrieverRAG class.
        """
        return super().__str__() + "-GPTOpenAILLMBERTRetrieverRAG"


class FalconLLMAdaRetrieverRAG(RAG):
    """
    FalconLLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the AutoModelDecoderRAGLLMV2 language model.
    This class provides functionality for performing retrieval-augmented generation (RAG) using AdaRetrieval for the
    retrieval step and AutoModelDecoderRAGLLMV2 for the language modeling step.

    Attributes:
        Retrieval (class): The retrieval model used by this class, AdaRetrieval.
        LLM (class): The language model used by this class, AutoModelDecoderRAGLLMV2.
    """

    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        """
        Returns a string representation of the FalconLLMAdaRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the FalconLLMAdaRetrieverRAG class.
        """
        return super().__str__() + "-FalconLLMAdaRetriever"


class FalconLLMBERTRetrieverRAG(RAG):
    """
    FalconLLMBERTRetrieverRAG class combines the BERTRetrieval retrieval model with the AutoModelDecoderRAGLLMV2 language model.
    This class provides functionality for performing retrieval-augmented generation (RAG) using BERTRetrieval for the
    retrieval step and AutoModelDecoderRAGLLMV2 for the language modeling step.

    Attributes:
        Retrieval (class): The retrieval model used by this class, BERTRetrieval.
        LLM (class): The language model used by this class, AutoModelDecoderRAGLLMV2.
    """

    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        """
        Returns a string representation of the FalconLLMBERTRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the FalconLLMBERTRetrieverRAG class.
        """
        return super().__str__() + "-FalconLLMBERTRetrieverRAG"


class VicunaLLMAdaRetrieverRAG(RAG):
    """
    VicunaLLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    This class provides functionality for performing retrieval-augmented generation (RAG) using AdaRetrieval for the
    retrieval step and AutoModelDecoderRAGLLM for the language modeling step.

    Attributes:
        Retrieval (class): The retrieval model used by this class, AdaRetrieval.
        LLM (class): The language model used by this class, AutoModelDecoderRAGLLM.
    """

    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns a string representation of the VicunaLLMAdaRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the VicunaLLMAdaRetrieverRAG class.
        """
        return super().__str__() + "-VicunaLLMAdaRetrieverRAG"


class VicunaLLMBERTRetrieverRAG(RAG):
    """
    VicunaLLMBERTRetrieverRAG class combines the BERTRetrieval retrieval model with the AutoModelDecoderRAGLLM language model.
    This class provides functionality for performing retrieval-augmented generation (RAG) using BERTRetrieval for the
    retrieval step and AutoModelDecoderRAGLLM for the language modeling step.

    Attributes:
        Retrieval (class): The retrieval model used by this class, BERTRetrieval.
        LLM (class): The language model used by this class, AutoModelDecoderRAGLLM.
    """

    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLM

    def __str__(self):
        """
        Returns a string representation of the VicunaLLMBERTRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the VicunaLLMBERTRetrieverRAG class.
        """
        return super().__str__() + "-VicunaLLMBERTRetrieverRAG"


class MPTLLMAdaRetrieverRAG(RAG):
    """
    MPTLLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the AutoModelDecoderRAGLLMV2 language model.
    This class provides functionality for performing retrieval-augmented generation (RAG) using AdaRetrieval for the
    retrieval step and AutoModelDecoderRAGLLMV2 for the language modeling step.

    Attributes:
        Retrieval (class): The retrieval model used by this class, AdaRetrieval.
        LLM (class): The language model used by this class, AutoModelDecoderRAGLLMV2.
    """

    Retrieval = AdaRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        """
        Returns a string representation of the MPTLLMAdaRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the MPTLLMAdaRetrieverRAG class.
        """
        return super().__str__() + "-MPTLLMAdaRetrieverRAG"


class MPTLLMBERTRetrieverRAG(RAG):
    """
    MPTLLMBERTRetrieverRAG class combines the BERTRetrieval retrieval model with the AutoModelDecoderRAGLLMV2 language model.
    This class provides functionality for performing retrieval-augmented generation (RAG) using BERTRetrieval for the
    retrieval step and AutoModelDecoderRAGLLMV2 for the language modeling step.

    Attributes:
        Retrieval (class): The retrieval model used by this class, BERTRetrieval.
        LLM (class): The language model used by this class, AutoModelDecoderRAGLLMV2.
    """

    Retrieval = BERTRetrieval
    LLM = AutoModelDecoderRAGLLMV2

    def __str__(self):
        """
        Returns a string representation of the MPTLLMBERTRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the MPTLLMBERTRetrieverRAG class.
        """
        return super().__str__() + "-MPTLLMBERTRetrieverRAG"


class MambaLLMAdaRetrieverRAG(RAG):
    """
    MambaLLMAdaRetrieverRAG class combines the AdaRetrieval retrieval model with the MambaSSMRAGLLM language model.
    This class provides functionality for performing retrieval-augmented generation (RAG) using AdaRetrieval for the
    retrieval step and MambaSSMRAGLLM for the language modeling step.

    Attributes:
        Retrieval (class): The retrieval model used by this class, AdaRetrieval.
        LLM (class): The language model used by this class, MambaSSMRAGLLM.
    """

    Retrieval = AdaRetrieval
    LLM = MambaSSMRAGLLM

    def __str__(self):
        """
        Returns a string representation of the MambaLLMAdaRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the MambaLLMAdaRetrieverRAG class.
        """
        return super().__str__() + "-MambaLLMAdaRetrieverRAG"


class MambaLLMBERTRetrieverRAG(RAG):
    """
    MambaLLMBERTRetrieverRAG class combines the BERTRetrieval retrieval model with the MambaSSMRAGLLM language model.
    This class provides functionality for performing retrieval-augmented generation (RAG) using BERTRetrieval for the
    retrieval step and MambaSSMRAGLLM for the language modeling step.

    Attributes:
        Retrieval (class): The retrieval model used by this class, BERTRetrieval.
        LLM (class): The language model used by this class, MambaSSMRAGLLM.
    """

    Retrieval = BERTRetrieval
    LLM = MambaSSMRAGLLM

    def __str__(self):
        """
        Returns a string representation of the MambaLLMBERTRetrieverRAG class, appending the specific retrieval and LLM
        configuration to the base string.

        Returns:
            str: A string representation of the MambaLLMBERTRetrieverRAG class.
        """
        return super().__str__() + "-MambaLLMBERTRetrieverRAG"
