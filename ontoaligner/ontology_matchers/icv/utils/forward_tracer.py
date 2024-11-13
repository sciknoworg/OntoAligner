# -*- coding: utf-8 -*-
"""
This script defines classes and functions for tracing the forward pass of a model, specifically for capturing
the activations of different layers (embedding, attention, and MLP layers). The `ForwardTracer` class allows
the activation values to be stored during the forward pass and subsequently accessed for analysis.

Classes:
    - ResidualStream: A data class that holds the activations for different parts of the model's forward pass,
                      including hidden states, attention weights, and MLP outputs.
    - ForwardTrace: A class that holds the forward trace of the model, including the residual stream and attention weights.
    - ForwardTracer: A context manager that registers hooks on a model's layers to trace activations during the
                     forward pass.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedModel

from .llm_layers import get_attention_layers, get_embedding_layer, get_layers, get_mlp_layers


@dataclass
class ResidualStream:
    """
    A data class that holds the activations (or residuals) at different stages of the model's forward pass.

    Attributes:
        hidden (torch.Tensor): The activations from the hidden layers of the model.
        attn (torch.Tensor): The activations from the attention layers of the model.
        mlp (torch.Tensor): The activations from the MLP layers of the model.
    """
    hidden: torch.Tensor
    attn: torch.Tensor
    mlp: torch.Tensor


class ForwardTrace:
    """
    A class that holds the forward trace for a model, including the residual stream of activations
    and attention weights collected during the forward pass.

    Attributes:
        residual_stream (ResidualStream): The activations from different parts of the model during the forward pass.
        attentions (torch.Tensor, optional): The attention maps collected from the attention layers of the model.
    """
    def __init__(self):
        """
        Initializes a `ForwardTrace` object, setting up an empty `ResidualStream` and initializing the
        `attentions` attribute to None.
        """
        self.residual_stream: Optional[ResidualStream] = ResidualStream(
            hidden=[],
            attn=[],
            mlp=[],
        )
        self.attentions: Optional[torch.Tensor] = None


class ForwardTracer:
    """
    A context manager for tracing the forward pass of a model. It registers hooks on the model's layers to collect
    activations from the hidden layers, attention layers, and MLP layers during the forward pass.

    Attributes:
        _model (PreTrainedModel): The model whose forward pass is being traced.
        _forward_trace (ForwardTrace): The object holding the trace information for the forward pass.
        _with_submodules (bool): A flag indicating whether to include submodule activations in the trace.
        _layers (list): The list of layers in the model (excluding embeddings, attention, and MLP).
        _attn_layers (list): The list of attention layers in the model.
        _mlp_layers (list): The list of MLP layers in the model.
        _hooks (list): The list of registered hooks for collecting activations.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        forward_trace: ForwardTrace,
        with_submodules: bool = False,
    ):
        """
        Initializes the `ForwardTracer` with a model, forward trace object, and a flag for submodules.

        Args:
            model (PreTrainedModel): The model to trace.
            forward_trace (ForwardTrace): The forward trace object to store activations.
            with_submodules (bool, optional): Flag indicating whether to trace submodule activations. Defaults to False.
        """
        self._model = model
        self._forward_trace = forward_trace
        self._with_submodules = with_submodules

        self._layers = get_layers(model)
        self._attn_layers = get_attention_layers(model)
        self._mlp_layers = get_mlp_layers(model)

        self._hooks = []

    def __enter__(self):
        """
        Enters the context manager and registers the forward hooks for tracing activations during the forward pass.
        """
        self._register_forward_hooks()

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits the context manager, removing all registered hooks and processing the collected activations.

        Args:
            exc_type (type): The exception type (if any).
            exc_value (Exception): The exception instance (if any).
            traceback (traceback): The traceback object (if any).
        """
        for hook in self._hooks:
            hook.remove()

        if exc_type is None:
            residual_stream = self._forward_trace.residual_stream

            if residual_stream.hidden[0] == []:
                residual_stream.hidden.pop(0)

            for key in residual_stream.__dataclass_fields__.keys():
                acts = getattr(residual_stream, key)
                # TODO: this is a hack, fix it
                if key != "hidden" and not self._with_submodules:
                    continue

                nonempty_layer_acts = [
                    layer_acts for layer_acts in acts if layer_acts != []
                ][0]
                final_shape = torch.cat(nonempty_layer_acts, dim=0).shape

                for i, layer_acts in enumerate(acts):
                    if layer_acts == []:
                        acts[i] = torch.zeros(final_shape)
                    else:
                        acts[i] = torch.cat(layer_acts, dim=0)
                acts = torch.stack(acts).transpose(0, 1)
                setattr(residual_stream, key, acts)

            # if self._with_submodules:
            #     self._forward_trace.attentions = torch.stack(self._forward_trace.attentions).transpose(0, 1)
            # else:
            self._forward_trace.attentions = None

    def _register_forward_hooks(self):
        """
        Registers forward hooks on the model layers to capture activations during the forward pass.
        This includes hooks for the embedding layer, hidden layers, attention layers, and MLP layers.
        """
        # model = self._model
        hooks = self._hooks
        residual_stream = self._forward_trace.residual_stream

        def store_activations(
            residual_stream: ResidualStream, acts_type: str, layer_num: int
        ):
            """
            Returns a hook function that stores activations in the residual stream.

            Args:
                residual_stream (ResidualStream): The residual stream object to store activations.
                acts_type (str): The type of activations to store ("hidden", "attn", or "mlp").
                layer_num (int): The layer number to store the activations for.

            Returns:
                function: The hook function to be used in the forward pass.
            """
            def hook(model, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                out = out.float().to("cpu", non_blocking=True)

                acts = getattr(residual_stream, acts_type)
                while len(acts) < layer_num + 1:
                    acts.append([])
                try:
                    acts[layer_num].append(out)
                except IndexError:
                    print(len(acts), layer_num)

            return hook

        def store_attentions(layer_num):
            """
            Returns a hook function that stores attention maps in the forward trace.

            Args:
                layer_num (int): The layer number to store the attention maps for.

            Returns:
                function: The hook function to be used in the forward pass.
            """
            def hook(model, inp, out):
                attention_maps = out[1]
                attention_maps = attention_maps.to("cpu", non_blocking=True).float()
                self._forward_trace.attentions[layer_num] = attention_maps

            return hook

        embedding_hook = get_embedding_layer(self._model).register_forward_hook(
            store_activations(residual_stream, "hidden", 0)
        )
        hooks.append(embedding_hook)

        for i, layer in enumerate(self._layers):
            hidden_states_hook = layer.register_forward_hook(
                store_activations(residual_stream, "hidden", i + 1)
            )
            hooks.append(hidden_states_hook)

        if self._with_submodules:
            for i, mlp_layer in enumerate(self._mlp_layers):
                mlp_res_hook = mlp_layer.register_forward_hook(
                    store_activations(residual_stream, "mlp", i)
                )
                hooks.append(mlp_res_hook)

            for i, attn_layer in enumerate(self._attn_layers):
                attn_res_hook = attn_layer.register_forward_hook(
                    store_activations(residual_stream, "attn", i)
                )
                hooks.append(attn_res_hook)
                # attn_attentions_hook = attn_layer.register_forward_hook(store_attentions(i))
                # hooks.append(attn_attentions_hook)
