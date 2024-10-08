# -*- coding: utf-8 -*-
"""
    BaseConfig: Data Configuration of models
"""
import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import openai
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())


class BaseConfig:
    def __init__(self, approach: Optional[str] = "none"):
        self.batch_size = None
        self.device = None
        self.root_dataset_dir = Path(__file__).parents[2]
        self.parser = argparse.ArgumentParser()
        self.approach = approach

    def mkdir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def __str__(self):
        return "config."

    def llm(self) -> Dict:
        config = {
            "max_token_length": 1,
            "tokenizer_max_length": 500,
            "num_beams": 1,
            "device": self.device,
            "truncation": True,
            "top_p": 0.95,
            "temperature": 0.8,
            "batch_size": self.batch_size,
            "padding": "max_length",
        }
        if self.approach == "naiv-conv-oaei":
            config["tokenizer_max_length"] = 4096
            config["max_token_length"] = 5000
        return config

    def gpt(self) -> Dict:
        openai.api_key = os.environ["OPENAI_KEY"]
        # due to the privacy I had to set key here,
        # but the correct way is to set inside OPENAILLMARCH class
        config = {
            "sleep": 5,
            "top_p": 0.95,
            "temperature": 0,
            "batch_size": self.batch_size,
            "max_token_length": 2,
        }
        if self.approach == "naiv-conv-oaei":
            config["max_token_length"] = 5000
        return config

    def fuzzy(self) -> Dict:
        config = {"fuzzy_sm_threshold": 0.8}
        return config

    def retrieval(self) -> Dict:
        config = {"top_k": 5, "device": self.device}
        return config

    def get_args(self, device="cpu", batch_size: int = None, nshots: int = None) -> Dict:
        self.device = device
        self.batch_size = batch_size

        # General configurations
        self.parser.add_argument("--root_dir", type=str, default=os.path.join(self.root_dataset_dir, "datasets"))
        self.parser.add_argument("--experiments_dir", type=str, default=os.path.join(self.root_dataset_dir, "experiments"))
        self.parser.add_argument("--output_dir", type=str, default=os.path.join(self.root_dataset_dir, "experiments", "outputs"))
        self.parser.add_argument("--stats_dir", type=str, default=os.path.join(self.root_dataset_dir, "experiments", "stats"))
        self.parser.add_argument("--openai_embedding_dir", type=str, default=os.path.join(self.root_dataset_dir, "assets", "openai-embedding"))

        # LLM configurations
        llm_config = self.llm()
        llm_models = ["FlanT5", "LLaMA7B", "Wizard13B", "LLaMA13B", "Mistral7B"]
        for llm_model in llm_models:
            self.parser.add_argument("--" + llm_model, type=dict, default=llm_config)
        self.parser.add_argument("--ChatGPT", type=dict, default=self.gpt())
        self.parser.add_argument("--GPT4", type=dict, default=self.gpt())

        # Lightweight configurations
        fuzzy_models = ['SimpleFuzzySM', 'WeightedFuzzySM', 'TokenSetFuzzySM']
        for fuzzy_model in fuzzy_models:
            self.parser.add_argument("--" + fuzzy_model, type=dict, default=self.fuzzy())

        # Retrieval Configurations
        retriever_config = self.retrieval()
        retriever_models = ["BM25Retrieval", "TFIDFRetrieval", "BERTRetrieval", "SpecterBERTRetrieval",
                            "FlanT5XLRetrieval", "FlanT5XXLRetrieval", "SVMBERTRetrieval", "AdaRetrieval"]
        for retriever_model in retriever_models:
            self.parser.add_argument("--" + retriever_model, type=dict, default=retriever_config)

        # RAG + ICV + FewShot configurations
        llama_rag_config = {"retriever-config": retriever_config, "llm-config": llm_config, "nshots": nshots}
        rag_icv_models = ["LLaMA7BAdaRAG", "MistralAdaRAG", "FalconAdaRAG", "VicunaAdaRAG", "MPTAdaRAG",
                          "LLaMA7BBertRAG", "MistralBertRAG", "FalconBertRAG", "VicunaBertRAG", "MPTBertRAG",
                          "LLaMA7BAdaICV", "FalconAdaICV", "VicunaAdaICV", "MPTAdaICV",
                          "LLaMA7BBertICV", "FalconBertICV", "VicunaBertICV", "MPTBertICV",
                          "LLaMA7BAdaFewShot", "MistralAdaFewShot", "FalconAdaFewShot", "VicunaAdaFewShot", "MPTAdaFewShot",
                          "LLaMA7BBertFewShot", "MistralBertFewShot", "FalconBertFewShot", "VicunaBertFewShot", "MPTBertFewShot",
                          "MambaLLMAdaFewShot", "MambaLLMBertFewShot", "MambaLLMAdaRAG", "MambaLLMBertRAG"]

        for rag_icv_model in rag_icv_models:
            self.parser.add_argument("--" + rag_icv_model, type=dict, default=llama_rag_config)

        openai_rag_config = {"retriever-config": retriever_config, "llm-config": self.gpt(), "nshots": nshots}
        self.parser.add_argument("--ChatGPTOpenAIAdaRAG", type=dict, default=openai_rag_config)
        self.parser.add_argument("--ChatGPTOpenAIAdaFewShot", type=dict, default=openai_rag_config)

        self.parser.add_argument("-f")
        return self.parser.parse_args()
