# -*- coding: utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from .icv import ICV, ICVBasedDecoderLLMArch
from ..retrieval.models import AdaRetrieval, BERTRetrieval


class LLaMA7BDecoderLM(ICVBasedDecoderLLMArch):
    tokenizer = LlamaTokenizer
    model = LlamaForCausalLM
    path = "meta-llama/Llama-2-7b-hf"

    def __str__(self):
        return super().__str__() + "-LLaMA-2-7B"


class Falcon7BDecoderLM(ICVBasedDecoderLLMArch):
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

    def check_answer_set_tokenizer(self, answer):
        return len(self.tokenizer(answer).input_ids) == 1


class Vicuna7BDecoderLM(ICVBasedDecoderLLMArch):
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM
    path = "lmsys/vicuna-7b-v1.5"

    def __str__(self):
        return super().__str__() + "-vicuna-7b-v1.5"


class MPT7BDecoderLM(ICVBasedDecoderLLMArch):
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

    def check_answer_set_tokenizer(self, answer):
        return len(self.tokenizer(answer).input_ids) == 1


class LLaMA7BLLMAdaICV(ICV):
    Retrieval = AdaRetrieval
    LLM = LLaMA7BDecoderLM

    def __str__(self):
        return super().__str__() + "-LLaMA7BLLMAdaICV"


class LLaMA7BLLMBertICV(ICV):
    Retrieval = BERTRetrieval
    LLM = LLaMA7BDecoderLM

    def __str__(self):
        return super().__str__() + "-LLaMA7BLLMBertICV"


class FalconLLMAdaICV(ICV):
    Retrieval = AdaRetrieval
    LLM = Falcon7BDecoderLM

    def __str__(self):
        return super().__str__() + "-FalconLLMAdaICV"


class FalconLLMBertICV(ICV):
    Retrieval = BERTRetrieval
    LLM = Falcon7BDecoderLM

    def __str__(self):
        return super().__str__() + "-FalconLLMBertICV"


class VicunaLLMAdaICV(ICV):
    Retrieval = AdaRetrieval
    LLM = Vicuna7BDecoderLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMAdaICV"


class VicunaLLMBertICV(ICV):
    Retrieval = BERTRetrieval
    LLM = Vicuna7BDecoderLM

    def __str__(self):
        return super().__str__() + "-VicunaLLMBertICV"


class MPTLLMAdaICV(ICV):
    Retrieval = AdaRetrieval
    LLM = MPT7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MPTLLMAdaICV"


class MPTLLMBertICV(ICV):
    Retrieval = BERTRetrieval
    LLM = MPT7BDecoderLM

    def __str__(self):
        return super().__str__() + "-MPTLLMBertICV"
