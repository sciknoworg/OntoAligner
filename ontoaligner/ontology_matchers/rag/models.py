# -*- coding: utf-8 -*-

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    MistralForCausalLM,
)

from .rag import RAG, RAGBasedDecoderLLMArch, RAGBasedOpenAILLMArch
from ..retrieval.models import AdaRetrieval, BERTRetrieval

from typing import Any
import torch


class LLaMA7BDecoderLM(RAGBasedDecoderLLMArch):
    tokenizer = LlamaTokenizer
    model = LlamaForCausalLM
    path = "meta-llama/Llama-2-7b-hf"

    def __str__(self):
        return super().__str__() + "-LLaMA-2-7B"


class Mistral7BDecoderLM(RAGBasedDecoderLLMArch):
    tokenizer = AutoTokenizer
    model = MistralForCausalLM
    path = "mistralai/Mistral-7B-v0.1"

    def __str__(self):
        return super().__str__() + "-MistralLM-7B-v0.1"


class Falcon7BDecoderLM(RAGBasedDecoderLLMArch):
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM
    path = "tiiuae/falcon-7b"

    def __str__(self):
        return super().__str__() + "-falcon-7b"

    def get_probas_yes_no(self, outputs):
        probas_yes_no = (
            outputs.scores[0][:, self.answer_sets_token_id["yes"] + self.answer_sets_token_id["no"]]
            .float()
            .softmax(-1)
        )
        return probas_yes_no

    def check_answer_set_tokenizer(self, answer: str) -> bool:
        return len(self.tokenizer(answer).input_ids) == 1


class Vicuna7BDecoderLM(RAGBasedDecoderLLMArch):
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM
    path = "lmsys/vicuna-7b-v1.5"

    def __str__(self):
        return super().__str__() + "-vicuna-7b-v1.5"


class MPT7BDecoderLM(RAGBasedDecoderLLMArch):
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM
    path = "mosaicml/mpt-7b"

    def __str__(self):
        return super().__str__() + "-mpt-7b"

    def get_probas_yes_no(self, outputs):
        probas_yes_no = (
            outputs.scores[0][:, self.answer_sets_token_id["yes"] + self.answer_sets_token_id["no"]]
            .float()
            .softmax(-1)
        )
        return probas_yes_no

    def check_answer_set_tokenizer(self, answer: str) -> bool:
        return len(self.tokenizer(answer).input_ids) == 1


class ChatGPTOpenAILLM(RAGBasedOpenAILLMArch):
    path = "gpt-3.5-turbo-1106"

    def __str__(self):
        return super().__str__() + "-GPT-3.5"


class LLaMA7BLLMAdaRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = LLaMA7BDecoderLM

    def __str__(self):
        return super().__str__() + "-LLaMA7BAdaRAG"


class MistralLLMAdaRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = Mistral7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MistralLMAdaRAG"


class LLaMA7BLLMBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = LLaMA7BDecoderLM

    def __str__(self):
        return super().__str__() + "-LLaMA7BLLMBertRAG"


class MistralLLMBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = Mistral7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MistralLLMBertRAG"


class ChatGPTOpenAIAdaRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = ChatGPTOpenAILLM

    def __str__(self):
        return super().__str__() + "-ChatGPTOpenAIAdaRAG"


class FalconLLMAdaRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = Falcon7BDecoderLM

    def __str__(self):
        return super().__str__() + "-FalconLLMAdaRAG"


class FalconLLMBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = Falcon7BDecoderLM

    def __str__(self):
        return super().__str__() + "-FalconLLMBertaRAG"


class VicunaLLMAdaRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = Vicuna7BDecoderLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMAdaRAG"


class VicunaLLMBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = Vicuna7BDecoderLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMBertRAG"


class MPTLLMAdaRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = MPT7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MPTLLMAdaRAG"


class MPTLLMBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = MPT7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MPTLLMBertRAG"


class Mamba3BSSMLLM(RAGBasedDecoderLLMArch):
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM
    path = "Q-bert/Mamba-3B"

    def __str__(self):
        return super().__str__() + "-mamba-3B"

    def load_model(self) -> None:
        if self.kwargs["device"] != "cpu":
            self.model = self.model.from_pretrained(
                self.path,
                load_in_8bit=True,
                device_map="balanced",
                trust_remote_code=True,
            )
        else:
            self.model = self.model.from_pretrained(self.path, trust_remote_code=True)
            self.model.to(self.kwargs["device"])

    def generate_for_llm(self, tokenized_input_data: Any) -> Any:
        with torch.cuda.amp.autocast():
            outputs = self.model.generate(
                tokenized_input_data.input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.kwargs["max_token_length"],
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )
        return outputs

    def get_probas_yes_no(self, outputs):
        probas_yes_no = (
            outputs.scores[0][:, self.answer_sets_token_id["yes"] + self.answer_sets_token_id["no"]]
            .float()
            .softmax(-1)
        )
        return probas_yes_no

    def check_answer_set_tokenizer(self, answer: str) -> bool:
        return len(self.tokenizer(answer).input_ids) == 1


class MambaLLMAdaRAG(RAG):
    Retrieval = AdaRetrieval
    LLM = Mamba3BSSMLLM

    def __str__(self):
        return super().__str__() + "-MambaLLMAdaRAG"


class MambaLLMBertRAG(RAG):
    Retrieval = BERTRetrieval
    LLM = Mamba3BSSMLLM

    def __str__(self):
        return super().__str__() + "-MambaLLMBertRAG"
