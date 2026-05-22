# Copyright 2025 Scientific Knowledge Organization (SciKnowOrg) Research Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script defines an OLaLa LLM aligner for ontology matching.

The aligner verifies SBERT-generated candidate correspondences using binary
yes/no token probabilities from a decoder language model.
"""
from typing import Any, Dict, List, Set
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import AutoTokenizer, AutoModelForCausalLM

from .dataset import OLaLaLLMDataset
from ..llm import DecoderLLMArch


class OLaLaStopOnWords(StoppingCriteria):
    """
    A stopping criterion for OLaLa yes/no token detection.
    """

    def __init__(self, tokenizer: Any, words_to_detect: List[Set[str]]) -> None:
        """
        Initializes the stopping criterion.

        Parameters:
            tokenizer (Any): The tokenizer used by the language model.
            words_to_detect (List[Set[str]]): The word groups to detect.
        """
        self.word_id_groups = []
        self.word_ids = set()

        for word_variations in words_to_detect:
            variation_ids = set()
            for text in word_variations:
                word_id = tokenizer(text, return_tensors="pt").input_ids[0][-1].item()
                variation_ids.add(word_id)
                self.word_ids.add(word_id)
            self.word_id_groups.append(variation_ids)

    def __call__(self, input_ids: Any, scores: Any, **kwargs) -> bool:
        """
        Checks whether the last generated token is a stop word.

        Parameters:
            input_ids (Any): The generated input ids.
            scores (Any): The generation scores.

        Returns:
            bool: True if generation should stop.
        """
        return input_ids[0][-1].item() in self.word_ids

    def get_position(self, sequences: Any, scores: Any) -> int:
        """
        Finds the first generated stop-word position.

        Parameters:
            sequences (Any): The generated token sequences.
            scores (Any): The generation scores.

        Returns:
            int: The generated-token position, or -1 if none was found.
        """
        generated_tokens = sequences[0][-len(scores):]

        for index, token in enumerate(generated_tokens):
            if token.item() in self.word_ids:
                return index

        return -1

    def get_confidence_for_position(self, scores: Any, position: int) -> List[float]:
        """
        Extracts word-group probabilities at a generated-token position.

        Parameters:
            scores (Any): The generation scores.
            position (int): The generated-token position.

        Returns:
            List[float]: The maximum probability for each word group.
        """
        normalized_scores = torch.nn.functional.softmax(scores[position], dim=1)
        confidences = []

        for word_ids in self.word_id_groups:
            maximum_value = 0.0
            for word_id in word_ids:
                value = normalized_scores[0][word_id].item()
                if value > maximum_value:
                    maximum_value = value
            confidences.append(maximum_value)

        return confidences

    def get_confidence_at_found_token(self, sequences: Any, scores: Any) -> List[float]:
        """
        Extracts confidence at the detected token or the first generated token.

        Parameters:
            sequences (Any): The generated token sequences.
            scores (Any): The generation scores.

        Returns:
            List[float]: The detected word-group confidences.
        """
        if len(scores) == 0:
            return [0.0 for _ in self.word_id_groups]

        position = self.get_position(sequences=sequences, scores=scores)
        if position == -1:
            position = 0

        return self.get_confidence_for_position(scores=scores, position=position)


class OLaLaLLMAligner(DecoderLLMArch):
    """
    An OLaLa LLM aligner for binary candidate verification.
    """
    tokenizer = AutoTokenizer
    model = AutoModelForCausalLM

    def __init__(
        self,
        device: str = "cpu",
        max_new_tokens: int = 10,
        temperature: float = 0.0,
        word_stopper: bool = True,
        loading_arguments: Dict[str, Any] = None,
        system_prompt_template: str = "{user_prompt}",
        dataset_class: Any = OLaLaLLMDataset,
        **kwargs,
    ) -> None:
        """
        Initializes the OLaLa LLM aligner.

        Parameters:
            device (str): The device used by the language model.
            max_new_tokens (int): The maximum number of generated tokens.
            temperature (float): The generation temperature.
            word_stopper (bool): Whether to stop when a yes/no token is generated.
            loading_arguments (Dict[str, Any]): Additional model loading arguments.
            system_prompt_template (str): Optional prompt wrapper.
            dataset_class (Any): The dataset class used to build prompts.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            word_stopper=word_stopper,
            loading_arguments=loading_arguments or {},
            **kwargs,
        )
        self.system_prompt_template = system_prompt_template
        self.dataset_class = dataset_class

    def load(self, path: str) -> None:
        """
        Loads the tokenizer and decoder language model.

        Parameters:
            path (str): The HuggingFace model path.

        Returns:
            None
        """
        self.path = path
        self.load_tokenizer(path=path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        loading_arguments = dict(self.kwargs.get("loading_arguments", {}))
        dtype_value = loading_arguments.get("torch_dtype")
        if isinstance(dtype_value, str):
            loading_arguments["torch_dtype"] = getattr(
                torch,
                dtype_value.strip().replace("torch.", ""),
            )

        model_class = type(self).model
        self.model = model_class.from_pretrained(path, **loading_arguments)

        if not loading_arguments.get("device_map"):
            self.model.to(self.kwargs["device"])

        self.model.eval()

    def include_more_variations(self, *words: str) -> Set[str]:
        """
        Adds MELT-style word variations.

        Parameters:
            *words (str): The words to expand.

        Returns:
            Set[str]: The expanded word variations.
        """
        intermediate = set()

        for word in words:
            intermediate.add(word)
            intermediate.add("*" + word)
            intermediate.add(" " + word)

        final_words = set()
        for word in intermediate:
            final_words.add(word)
            final_words.add(word.lower())
            final_words.add(word.upper())
            final_words.add(word.title())

        return final_words

    def get_words_to_detect(self) -> List[Set[str]]:
        """
        Returns positive and negative word groups.

        Returns:
            List[Set[str]]: The positive and negative word groups.
        """
        return [
            self.include_more_variations("yes", "true"),
            self.include_more_variations("no", "false"),
        ]

    def predict_confidence(self, prompt: str) -> float:
        """
        Predicts the binary LLM confidence for one prompt.

        Parameters:
            prompt (str): The filled LLM prompt.

        Returns:
            float: The LLM binary confidence.
        """
        stopper = OLaLaStopOnWords(
            tokenizer=self.tokenizer,
            words_to_detect=self.get_words_to_detect(),
        )

        model_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=self.kwargs["truncation"],
            max_length=self.kwargs["max_length"],
            padding=self.kwargs["padding"],
        )

        device = getattr(self.model, "device", None)
        if device is None:
            device = next(self.model.parameters()).device

        model_inputs.to(device)

        generation_arguments = {
            "max_new_tokens": self.kwargs["max_new_tokens"],
            "output_scores": True,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if self.kwargs.get("temperature", 0.0) > 0:
            generation_arguments["temperature"] = self.kwargs["temperature"]

        if self.kwargs.get("word_stopper", True):
            generation_arguments["stopping_criteria"] = StoppingCriteriaList([stopper])

        with torch.no_grad():
            generated_sequence = self.model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs.get("attention_mask", None),
                **generation_arguments,
            )

        confidences = stopper.get_confidence_at_found_token(
            sequences=generated_sequence.sequences,
            scores=generated_sequence.scores,
        )

        yes_confidence = confidences[0]
        no_confidence = confidences[1]
        denominator = yes_confidence + no_confidence

        if denominator == 0:
            return 0.0

        return yes_confidence / denominator

    def group_predictions(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Groups pair-level LLM predictions by source IRI.

        Parameters:
            pairs (List[Dict[str, Any]]): The pair-level predictions.

        Returns:
            List[Dict[str, Any]]: The grouped predictions.
        """
        grouped = {}

        for pair in pairs:
            source_iri = pair["source"]
            if source_iri not in grouped:
                grouped[source_iri] = []

            grouped[source_iri].append(pair)

        outputs = []
        for source_iri, source_pairs in grouped.items():
            source_pairs = sorted(
                source_pairs,
                key=lambda item: item["llm_score"],
                reverse=True,
            )

            outputs.append({
                "source": source_iri,
                "target-cands": [pair["target"] for pair in source_pairs],
                "score-cands": [pair["llm_score"] for pair in source_pairs],
            })

        return outputs

    def generate(self, input_data: List) -> List:
        """
        Generates LLM-verified candidate correspondences.

        Parameters:
            input_data (List): The source ontology, target ontology, and candidates.

        Returns:
            List: The LLM-verified candidate correspondences.
        """
        dataset = self.dataset_class(
            source_onto=input_data[0],
            target_onto=input_data[1],
            candidates=input_data[2],
            system_prompt_template=self.system_prompt_template,
        )

        scored_pairs = []

        for index in range(len(dataset)):
            sample = dataset[index]
            source_iri, target_iri = sample["iris"]

            if not sample["is_valid"]:
                llm_score = 0.0
            else:
                llm_score = self.predict_confidence(prompt=sample["prompts"])

            scored_pairs.append({
                "source": source_iri,
                "target": target_iri,
                "llm_score": llm_score,
            })

        return self.group_predictions(scored_pairs)

    def __str__(self):
        """
        Returns the string representation of the aligner.

        Returns:
            str: The string representation of the aligner.
        """
        return super().__str__() + "-OLaLaLLMAligner"
