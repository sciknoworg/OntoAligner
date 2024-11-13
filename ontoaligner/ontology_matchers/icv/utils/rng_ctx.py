# -*- coding: utf-8 -*-
"""
This script defines utilities for managing and controlling the random number generator (RNG) states
of Python's `random`, NumPy, and PyTorch. It includes classes to save, restore, and control the RNG states
to ensure reproducibility across various computations involving these libraries.

Classes:
    - RandomState: A utility class for storing and restoring the state of Python's `random`, NumPy, and PyTorch RNGs.
    - RandomContext: A context manager that saves and restores RNG states when entering and exiting a block of code.
    - EmptyContext: A simple context manager that does nothing, used as a placeholder.

"""

import random
import sys

import numpy as np
import torch


class RandomState:
    """
    A class to store and restore the state of Python's `random`, NumPy, and PyTorch RNGs.

    This class is used to capture the RNG state of multiple libraries, so that it can be restored later,
    ensuring that random number generation is reproducible across sessions or experiments.
    """

    def __init__(self):
        """
        Initializes and saves the current state of Python's `random`, NumPy, and PyTorch RNGs.

        - Saves the RNG state for the Python `random` module.
        - Saves the RNG state for NumPy.
        - Saves the RNG state for the CPU and all available GPU devices in PyTorch.
        """
        self.random_mod_state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_cpu_state = torch.get_rng_state()
        self.torch_gpu_states = [
            torch.cuda.get_rng_state(d) for d in range(torch.cuda.device_count())
        ]

    def restore(self):
        """
        Restores the RNG state for Python's `random`, NumPy, and PyTorch (CPU and GPU) from the saved states.

        This method ensures that the RNG states are set back to their values at the time of the `RandomState` object creation.
        """
        random.setstate(self.random_mod_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.torch_cpu_state)
        for d, state in enumerate(self.torch_gpu_states):
            torch.cuda.set_rng_state(state, d)


class RandomContext:
    """
    Save and restore state of PyTorch, NumPy, Python RNGs.
    A context manager that saves and restores the RNG states of Python's `random`, NumPy, and PyTorch.

    This class ensures that RNG states are restored to their initial values upon exiting the context,
    even if an exception occurs within the context.
    """

    def __init__(self, seed=None):
        """
        Initializes the context manager by saving the current RNG states and setting new ones based on the provided seed.

        Args:
            seed (int, optional): The seed to initialize the RNGs. If None, a random seed is used.
        """
        outside_state = RandomState()

        random.seed(seed)
        np.random.seed(seed)
        if seed is None:
            torch.manual_seed(random.randint(-sys.maxsize - 1, sys.maxsize))
        else:
            torch.manual_seed(seed)
        # torch.cuda.manual_seed_all is called by torch.manual_seed
        self.inside_state = RandomState()

        outside_state.restore()

        self._active = False

    def __enter__(self):
        """
        Enters the context and restores the RNG state to the values saved before entering the context.

        This method ensures that the RNG state inside the context block is consistent with the state before the context started.

        Raises:
            Exception: If the context manager is entered more than once without exiting.
        """
        if self._active:
            raise Exception("RandomContext can be active only once")

        self.outside_state = RandomState()
        self.inside_state.restore()
        self._active = True

    def __exit__(self, exception_type, exception_value, traceback):
        """
        Exits the context, restoring the RNG state to the saved state from before the context was entered.

        This method is called when the block inside the context manager exits, ensuring that RNG states are
        restored to their pre-context values, even if an exception occurred within the block.
        """
        self.inside_state = RandomState()
        self.outside_state.restore()
        self.outside_state = None

        self._active = False


class EmptyContext:
    """
    A context manager that does nothing.

    This is a placeholder context manager used when no state management is needed. It allows for a consistent API
    without performing any actual actions.
    """

    def __enter__(self):
        """
        Enters the context. This method does nothing.
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the context. This method does nothing.
        """
        pass
