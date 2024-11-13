# -*- coding: utf-8 -*-
"""
This script provides utility functions for working with Transformer-based models in PyTorch.
It includes functions for accessing various layers within the model, such as the embedding layer,
attention layers, and MLP layers. The script also includes functions for recursively traversing
the model to find the longest `nn.ModuleList`, and for setting and getting nested attributes within the model.

Functions:
    - get_nested_attr: Retrieves a nested attribute from an object using a string path.
    - set_nested_attr: Sets a nested attribute on an object using a string path.
    - find_longest_modulelist: Recursively searches a PyTorch model to find the longest `nn.ModuleList`.
    - find_module: Finds a module in a PyTorch model based on a list of keyword matches.
    - get_embedding_layer: Finds and returns the embedding layer in a transformer model.
    - get_lm_head: Finds and returns the language model head of a transformer model.
    - get_lm_pipeline: Returns the complete language model pipeline (normalization and head) for different model types.
    - get_layers_path: Finds the path to the longest `nn.ModuleList` in a model.
    - get_layers: Retrieves all the layers from the model, based on the longest `nn.ModuleList`.
    - get_attention_layers: Finds and returns the attention layers in a transformer model.
    - get_mlp_layers: Finds and returns the MLP (feedforward) layers in a transformer model.
"""
from torch import nn
from transformers import PreTrainedModel


def get_nested_attr(obj, attr_path):
    """
    Retrieves a nested attribute from an object using a string path.

    Args:
        obj (object): The object from which to retrieve the attribute.
        attr_path (str): A dot-separated string representing the path to the attribute.

    Returns:
        The value of the nested attribute.
    """
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def set_nested_attr(obj, attr_path, value):
    """
    Sets a nested attribute on an object using a string path.

    Args:
        obj (object): The object on which to set the attribute.
        attr_path (str): A dot-separated string representing the path to the attribute.
        value (any): The value to set for the nested attribute.
    """
    attrs = attr_path.split(".")
    parent = get_nested_attr(obj, ".".join(attrs[:-1]))
    setattr(parent, attrs[-1], value)


def find_longest_modulelist(model, path=""):
    """
    Recursively find the longest nn.ModuleList in a PyTorch model.
    Args:
        model: PyTorch model.
        path: Current path in the model (used for recursion).
    Returns:
        Tuple with path and length of the longest nn.ModuleList found.
    """
    longest_path = path
    longest_len = 0

    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name

        # Recursively check the child's children
        child_path, child_len = find_longest_modulelist(
            child, f"{path}.{name}" if path else name
        )
        if child_len > longest_len:
            longest_len = child_len
            longest_path = child_path

    return longest_path, longest_len


def find_module(block, keywords):
    """
    Try to find a module in a transformer block.
    Args:
        block: Transformer block (nn.Module).
        keywords: List of possible module names (str).
    Returns:
        The found module if found, else None.
    """
    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")


def get_embedding_layer(model: PreTrainedModel):
    """
    Finds and returns the embedding layer of a transformer model.

    Args:
        model (PreTrainedModel): The transformer model from which to retrieve the embedding layer.

    Returns:
        nn.Module: The embedding layer of the model.
    """
    # model_type = model.__class__.__name__
    # if model_type == "LlamaForCausalLM":
    #     return model.model.embed_tokens
    # elif model_type == "RWForCausalLM":
    #     return model.transformer.word_embeddings

    keywords = ["emb", "wte"]
    return find_module(model, keywords)


def get_lm_head(model: PreTrainedModel):
    """
    Finds and returns the language model head of a transformer model.

    Args:
        model (PreTrainedModel): The transformer model from which to retrieve the LM head.

    Returns:
        nn.Module: The LM head module of the model.
    """
    keywords = ["lm_head", "embed_out"]
    return find_module(model, keywords)


def get_lm_pipeline(model: PreTrainedModel):
    """
    Returns the complete language model pipeline (normalization and head) for different model types.

    Args:
        model (PreTrainedModel): The transformer model for which the LM pipeline is generated.

    Returns:
        nn.Sequential: A sequential model containing the normalization and LM head.
    """
    model_class = model.__class__.__name__

    if model_class == "LlamaForCausalLM":
        return nn.Sequential(model.model.norm, model.lm_head)
    elif model_class == "RWForCausalLM":
        return nn.Sequential(model.transformer.ln_f, model.lm_head)
    elif model_class == "GPTNeoForCausalLM":
        return nn.Sequential(model.transformer.ln_f, model.lm_head)
    elif model_class == "GPTNeoXForCausalLM":
        return nn.Sequential(model.gpt_neox.final_layer_norm, model.embed_out)

    # TODO: make the default case more robust
    return get_lm_head(model)


def get_layers_path(model: PreTrainedModel):
    """
    Finds the path to the longest `nn.ModuleList` in a model.

    Args:
        model (PreTrainedModel): The model to search within.

    Returns:
        str: The path to the longest `nn.ModuleList`.
    """
    longest_path, longest_len = find_longest_modulelist(model)
    return longest_path


def get_layers(model: PreTrainedModel):
    """
    Retrieves all the layers from a model, based on the longest `nn.ModuleList`.

    Args:
        model (PreTrainedModel): The model from which to retrieve layers.

    Returns:
        list: A list of layers from the model.
    """
    # model_type = model.__class__.__name__
    # if model_type == "LlamaForCausalLM":
    #     return model.model.layers
    # elif model_type == "RWForCausalLM":
    #     return model.transformer.h

    longest_path = get_layers_path(model)
    return get_nested_attr(model, longest_path)


def get_attention_layers(model: PreTrainedModel):
    """
    Finds and returns the attention layers in a transformer model.

    Args:
        model (PreTrainedModel): The transformer model to retrieve attention layers from.

    Returns:
        list: A list of attention layers in the model.
    """
    # model_type = model.__class__.__name__
    # if model_type == "LlamaForCausalLM":
    #     return [layer.self_attn for layer in layers]
    # elif model_type == "RWForCausalLM":
    #     return [layer.self_attention for layer in layers]

    layers = get_layers(model)
    keywords = ["attention", "attn"]
    attention_layers = [find_module(layer, keywords) for layer in layers]
    return attention_layers


def get_mlp_layers(model: PreTrainedModel):
    """
    Finds and returns the MLP (feedforward) layers in a transformer model.

    Args:
        model (PreTrainedModel): The transformer model to retrieve MLP layers from.

    Returns:
        list: A list of MLP layers in the model.
    """
    # model_type = model.__class__.__name__
    # if model_type == "LlamaForCausalLM":
    #     return [layer.mlp for layer in layers]
    # elif model_type == "RWForCausalLM":
    #     return [layer.mlp for layer in layers]

    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    mlp_layers = [find_module(layer, mlp_keywords) for layer in layers]
    return mlp_layers
