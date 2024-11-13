# -*- coding: utf-8 -*-
"""
This script defines context managers for managing and modifying forward passes in a model.
It provides utility for handling multiple context managers at once, and for tracing the forward pass
of a model, optionally including submodules.

Classes:
    - CombinedContextManager: A context manager that allows the use of multiple context managers
                              in a single context.

Functions:
    - modified_forward_context_manager: Returns a context manager that applies a set of forward
                                         modifiers during the forward pass of a model.
    - traced_forward_context_manager: Returns a context manager and a forward trace object
                                      to trace the forward pass of a model, optionally including
                                      submodules.
"""

from contextlib import AbstractContextManager, ExitStack

from .forward_tracer import ForwardTrace, ForwardTracer


class CombinedContextManager(AbstractContextManager):
    """
    A context manager that allows the use of multiple context managers simultaneously.
    It ensures that all provided context managers are entered and exited in the correct order.

    Attributes:
        context_managers (list): A list of context managers to be managed together.
        stack (ExitStack, optional): The ExitStack used to manage the nested context managers.
    """

    def __init__(self, context_managers):
        """
        Initializes the CombinedContextManager with a list of context managers to be used together.

        Args:
            context_managers (list): A list of context managers to be managed together.
        """
        self.context_managers = context_managers
        self.stack = None

    def __enter__(self):
        """
        Enters the context, entering all the provided context managers.

        Returns:
            ExitStack: The ExitStack that manages the nested context managers.
        """
        self.stack = ExitStack()
        for cm in self.context_managers:
            self.stack.enter_context(cm)
        return self.stack

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the context, ensuring all context managers are properly exited.

        Args:
            exc_type (type): The exception type (if any).
            exc_val (Exception): The exception instance (if any).
            exc_tb (traceback): The traceback object (if any).
        """
        if self.stack is not None:
            self.stack.__exit__(exc_type, exc_val, exc_tb)


def modified_forward_context_manager(model, forward_modifiers=()):
    """
    Creates a context manager that applies a set of forward modifiers during the forward pass
    of a model. This context manager ensures that the forward pass is modified according to
    the provided modifiers.

    Args:
        model (nn.Module): The model whose forward pass is to be modified.
        forward_modifiers (tuple, optional): A tuple of context managers or functions to modify
                                              the forward pass. Default is an empty tuple.

    Returns:
        CombinedContextManager: A context manager that applies the provided forward modifiers.
    """
    context_manager = CombinedContextManager([*forward_modifiers])
    return context_manager


def traced_forward_context_manager(model, with_submodules=False):
    """
    Creates a context manager and a forward trace object that traces the forward pass of a
    model. This context manager captures the forward pass, optionally including submodules
    of the model.

    Args:
        model (nn.Module): The model whose forward pass is to be traced.
        with_submodules (bool, optional): Whether to include submodules in the trace. Default is False.

    Returns:
        tuple: A tuple containing:
            - ForwardTracer: A context manager that traces the forward pass of the model.
            - ForwardTrace: The forward trace object that holds the trace data.
    """
    forward_trace = ForwardTrace()
    context_manager = ForwardTracer(
        model, forward_trace, with_submodules=with_submodules
    )
    return context_manager, forward_trace
