# -*- coding: utf-8 -*-
from ontomap.ontology_matchers.icv.models import (
    FalconLLMAdaICV,
    FalconLLMBertICV,
    LLaMA7BLLMAdaICV,
    LLaMA7BLLMBertICV,
    MPTLLMAdaICV,
    MPTLLMBertICV,
    VicunaLLMAdaICV,
    VicunaLLMBertICV,
)
from ontomap.ontology_matchers.lightweight.models import (
    SimpleFuzzySMLightweight,
    TokenSetFuzzySMLightweight,
    WeightedFuzzySMLightweight,
)
from ontomap.ontology_matchers.llm.models import (
    ChatGPTOpenAILLM,
    FlanT5XLEncoderDecoderLM,
    FlanT5XXLEncoderDecoderLM,
    GPT4OpenAILLM,
    LLaMA7BDecoderLM,
    LLaMA13BDecoderLM,
    MistralLM7BDecoderLM,
    WizardLM13BDecoderLM,
)
from ontomap.ontology_matchers.rag.models import (
    ChatGPTOpenAIAdaRAG,
    FalconLLMAdaRAG,
    FalconLLMBertRAG,
    LLaMA7BLLMAdaRAG,
    LLaMA7BLLMBertRAG,
    MistralLLMAdaRAG,
    MistralLLMBertRAG,
    MPTLLMAdaRAG,
    MPTLLMBertRAG,
    VicunaLLMAdaRAG,
    VicunaLLMBertRAG,
    MambaLLMAdaRAG,
    MambaLLMBertRAG
)
from ontomap.ontology_matchers.retrieval.models import (
    AdaRetrieval,
    BERTRetrieval,
    BM25Retrieval,
    FlanT5XLRetrieval,
    FlanT5XXLRetrieval,
    SpecterBERTRetrieval,
    SVMBERTRetrieval,
    TFIDFRetrieval,
)
from ontomap.ontology_matchers.fewshot.models import (
    ChatGPTOpenAIAdaFewShot,
    FalconLLMAdaFewShot,
    FalconLLMBertFewShot,
    LLaMA7BLLMAdaFewShot,
    LLaMA7BLLMBertFewShot,
    MistralLLMAdaFewShot,
    MistralLLMBertFewShot,
    MPTLLMAdaFewShot,
    MPTLLMBertFewShot,
    VicunaLLMAdaFewShot,
    VicunaLLMBertFewShot,
    MambaLLMAdaFewShot,
    MambaLLMBertFewShot
)

MatcherCatalog = {
    "naiv-conv-oaei": {
        "FlanT5XL": FlanT5XLEncoderDecoderLM,
        "FlanT5XXL": FlanT5XXLEncoderDecoderLM,
        "LLaMA7B": LLaMA7BDecoderLM,
        "LLaMA13B": LLaMA13BDecoderLM,
        "Wizard13B": WizardLM13BDecoderLM,
        "Mistral7B": MistralLM7BDecoderLM,
        "ChatGPT": ChatGPTOpenAILLM,
        "GPT4": GPT4OpenAILLM,
    },
    "lightweight": {
        "SimpleFuzzySM": SimpleFuzzySMLightweight,
        "WeightedFuzzySM": WeightedFuzzySMLightweight,
        "TokenSetFuzzySM": TokenSetFuzzySMLightweight,
    },
    "rag": {
        "LLaMA7BAdaRAG": LLaMA7BLLMAdaRAG,
        "MistralAdaRAG": MistralLLMAdaRAG,
        "ChatGPTOpenAIAdaRAG": ChatGPTOpenAIAdaRAG,
        "LLaMA7BBertRAG": LLaMA7BLLMBertRAG,
        "MistralBertRAG": MistralLLMBertRAG,
        "FalconAdaRAG": FalconLLMAdaRAG,
        "FalconBertRAG": FalconLLMBertRAG,
        "VicunaBertRAG": VicunaLLMBertRAG,
        "VicunaAdaRAG": VicunaLLMAdaRAG,
        "MPTBertRAG": MPTLLMBertRAG,
        "MPTAdaRAG": MPTLLMAdaRAG,
        "MambaLLMAdaRAG": MambaLLMAdaRAG,
        "MambaLLMBertRAG": MambaLLMBertRAG

    },
    "retrieval": {
        "BM25Retrieval": BM25Retrieval,
        "TFIDFRetrieval": TFIDFRetrieval,
        "BERTRetrieval": BERTRetrieval,
        "SpecterBERTRetrieval": SpecterBERTRetrieval,
        "FlanT5XLRetrieval": FlanT5XLRetrieval,
        "FlanT5XXLRetrieval": FlanT5XXLRetrieval,
        "SVMBERTRetrieval": SVMBERTRetrieval,
        "AdaRetrieval": AdaRetrieval,
    },
    "icv": {
        "LLaMA7BAdaICV": LLaMA7BLLMAdaICV,
        "LLaMA7BBertICV": LLaMA7BLLMBertICV,
        "FalconAdaICV": FalconLLMAdaICV,
        "FalconBertICV": FalconLLMBertICV,
        "VicunaBertICV": VicunaLLMBertICV,
        "VicunaAdaICV": VicunaLLMAdaICV,
        "MPTBertICV": MPTLLMBertICV,
        "MPTAdaICV": MPTLLMAdaICV,
    },
    "fewshot": {
        "LLaMA7BAdaFewShot": LLaMA7BLLMAdaFewShot,
        "MistralAdaFewShot": MistralLLMAdaFewShot,
        "ChatGPTOpenAIAdaFewShot": ChatGPTOpenAIAdaFewShot,
        "LLaMA7BBertFewShot": LLaMA7BLLMBertFewShot,
        "MistralBertFewShot": MistralLLMBertFewShot,
        "FalconAdaFewShot": FalconLLMAdaFewShot,
        "FalconBertFewShot": FalconLLMBertFewShot,
        "VicunaBertFewShot": VicunaLLMBertFewShot,
        "VicunaAdaFewShot": VicunaLLMAdaFewShot,
        "MPTBertFewShot": MPTLLMBertFewShot,
        "MPTAdaFewShot": MPTLLMAdaFewShot,
        "MambaLLMAdaFewShot": MambaLLMAdaFewShot,
        "MambaLLMBertFewShot": MambaLLMBertFewShot
    },
}

__all__ = ["MatcherCatalog"]
