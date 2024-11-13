# -*- coding: utf-8 -*-
"""
This script defines a custom Dataset class `TokenizedForStyleRightPad` for processing and tokenizing
text data for style-based tasks, typically used for prompt-based language models. It processes input data,
tokenizes it with the provided tokenizer, and prepares it for training or evaluation.

Classes:
    TokenizedForStyleRightPad: A subclass of PyTorch's Dataset used to tokenize and process text data
                               for style-based prompt learning tasks.

"""

# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer


class TokenizedForStyleRightPad(Dataset):
    """
    A PyTorch Dataset for tokenizing and processing text data with style-based prompts for either
    evaluation or fine-tuning tasks. It supports padding the input text to a maximum length.

    Attributes:
        tok (PreTrainedTokenizer): The tokenizer used to process the input text.
        prompt_fn (function): A function that generates prompts for the input queries.
        references (list): A list of references associated with the input queries (used for evaluation).
        max_length (int): The maximum tokenized length of the data.
        data (list): A list of processed input data for either training or evaluation.
    """

    def __init__(
        self,
        data,
        tok: PreTrainedTokenizer,
        prompt_fn,
        mode="eval",
        no_padding=False,
        prefix="",
    ):
        """
        Initializes the TokenizedForStyleRightPad dataset with tokenization, prompt generation, and data
        preprocessing. It prepares the data based on the specified mode (either fine-tuning or evaluation).

        Args:
            data (list): List of dictionaries containing 'query' (str) and 'choices' (list of str).
            tok (PreTrainedTokenizer): The tokenizer for tokenizing the input queries.
            prompt_fn (function): The function to generate prompts from the input queries.
            mode (str, optional): Specifies the mode ('eval' for evaluation or 'ft' for fine-tuning).
                                  Default is 'eval'.
            no_padding (bool, optional): Whether to disable padding for tokenized text. Default is False.
            prefix (str, optional): A prefix to add to the prompt. Default is an empty string.
        """
        # data: [query: str, choices: list(str)]
        self.tok = tok
        self.prompt_fn = prompt_fn
        self.references = None
        self.max_length = self._find_max_length(data, mode=mode)
        if mode == "ft":
            self.data = self._build_ft_data(data)
        elif mode == "eval":
            self.data, self.references = self._build_eval_data(
                data, no_padding=no_padding, prefix=prefix
            )
        else:
            raise NotImplementedError
        print(f"Tokenization finished: {len(self.data)}, max_length={self.max_length}")

    def _find_max_length(self, data, mode=eval):
        """
        Calculates the maximum tokenized length across the dataset based on the query length and
        prompt function output.

        Args:
            data (list): The dataset to calculate the maximum length from.
            mode (str, optional): The mode of operation ('eval' for evaluation or 'ft' for fine-tuning).
                                  Default is 'eval'.

        Returns:
            int: The maximum length of tokenized input in the dataset.
        """
        max_len = 0

        def tok_len(t):
            return len(self.tok.encode(t))

        for ex in tqdm(data, desc="Data preprocessing(1/2)"):
            query = ex["query"]
            if mode == "eval":
                len_query = len(self.prompt_fn(query)[0])
            elif mode == "ft":
                len_query = len(self.prompt_fn(query)[1])
            else:
                raise NotImplementedError
            max_len = max(max_len, len_query)
        return max_len

    def _build_eval_data(self, data, no_padding=False, prefix=""):
        """
        Processes the input data for evaluation mode. It applies the prompt function to the query and
        tokenizes the data, optionally including padding.

        Args:
            data (list): The dataset to be processed.
            no_padding (bool, optional): Whether to disable padding in the tokenized output. Default is False.
            prefix (str, optional): A prefix to prepend to each prompt. Default is an empty string.

        Returns:
            tuple: A tuple containing a list of tokenized data and a list of references.
        """
        processed = []
        references = []
        for ex in tqdm(data, desc="Data preprocessing(2/2)"):
            query = ex["query"]
            processed_input = self.prompt_fn(
                query, return_reference=True, Instruction=prefix
            )
            t_query, t_full, t_reference = processed_input
            processed_input = self.tokenize(t_full, t_query, no_padding=no_padding)
            processed.append(processed_input)
            references.append(t_reference)

        print("Style dataset: finish!")
        return processed, references

    def _build_ft_data(self, data):
        """
        Processes the input data for fine-tuning mode. It applies the prompt function to the query and
        tokenizes the data.

        Args:
            data (list): The dataset to be processed.

        Returns:
            list: A list of tokenized data for fine-tuning.
        """
        processed = []
        for ex in tqdm(data, desc="Data preprocessing(2/2)"):
            query = ex["query"]
            processed_input = self.prompt_fn(query)
            t_query, t_full = processed_input
            processed_input = self.tokenize(t_query, t_full)
            processed.append(processed_input)

        print("Finetuning dataset: finish!")
        return processed

    def tokenize_demonstration(self, demonstration):
        """
        Tokenizes a single demonstration by converting the text into input IDs and attention masks.

        Args:
            demonstration (str): The demonstration text to be tokenized.

        Returns:
            tuple: A tuple containing the tokenized input IDs and attention mask as LongTensors.
        """
        e = self.tok(demonstration)
        return torch.LongTensor(e["input_ids"]), torch.LongTensor(
            e["attention_mask"]
        )  # no padding

    def tokenize_each_demonstration(self, demonstration_list, dataset_name=None):
        """
        Tokenizes a list of demonstrations by converting each pair of text into tokenized input.

        Args:
            demonstration_list (list of tuples): List of demonstration pairs (original, rewritten).
            dataset_name (str, optional): The name of the dataset (unused in this method).

        Returns:
            list: A list of tokenized demonstrations.
        """
        tokenized_demonstration_list = []
        for exp_id in range(len(demonstration_list)):
            demonstration_list[exp_id] = (
                demonstration_list[exp_id][0].strip(" .").strip("."),
                demonstration_list[exp_id][1].strip(" .").strip("."),
            )

            e_original = self.tok(demonstration_list[exp_id][0])
            e_rewrite = self.tok(demonstration_list[exp_id][1])
            tokenized_demonstration_list.append((e_original, e_rewrite))
        return tokenized_demonstration_list

    def tokenize(self, only_query, full_text, no_padding=False):
        """
        Tokenizes the input text with optional padding to the maximum length.

        Args:
            only_query (str): The query part of the input text.
            full_text (str): The full input text including the query.
            no_padding (bool, optional): Whether to disable padding. Default is False.

        Returns:
            dict: A dictionary containing the tokenized input IDs and attention mask.
        """
        # tok_only_query = self.tok(only_query, add_special_tokens=False)
        tok_full_no_padding = self.tok(full_text, add_special_tokens=False)
        tok_full = self.tok(
            full_text,
            padding="max_length",
            max_length=self.max_length,
            add_special_tokens=False,
        )  # <pad> is not a special token

        if no_padding:
            e = {
                "input_ids": tok_full_no_padding.input_ids,
                "attention_mask": tok_full_no_padding.attention_mask,
            }
        else:
            e = {
                "input_ids": tok_full.input_ids,
                "attention_mask": tok_full.attention_mask,
            }

        return e

    def __len__(self):
        """
        Returns the size of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset by index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the tokenized input IDs, attention mask, and (optionally) reference.
        """
        es = self.data[idx]

        if self.references:
            return (
                torch.LongTensor(es["input_ids"]),
                torch.LongTensor(es["attention_mask"]),
                self.references[idx],
            )
        else:
            return es
