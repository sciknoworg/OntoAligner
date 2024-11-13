# -*- coding: utf-8 -*-
"""
This script defines different variants of the `FuzzySMLightweight` class, each implementing
a different string similarity ratio estimation method using the RapidFuzz library.

The `SimpleFuzzySMLightweight`, `WeightedFuzzySMLightweight`, and `TokenSetFuzzySMLightweight`
classes each override the `ratio_estimate` method to use different string comparison techniques
from RapidFuzz for fuzzy string matching.

Classes:
    - SimpleFuzzySMLightweight: Inherits from `FuzzySMLightweight` and uses the basic string ratio.
    - WeightedFuzzySMLightweight: Inherits from `FuzzySMLightweight` and uses weighted string ratio.
    - TokenSetFuzzySMLightweight: Inherits from `FuzzySMLightweight` and uses token set ratio for fuzzy matching.
"""

from typing import Any

import rapidfuzz

from .lightweight import FuzzySMLightweight


class SimpleFuzzySMLightweight(FuzzySMLightweight):
    """
    A subclass of `FuzzySMLightweight` that uses the basic string similarity ratio from RapidFuzz.
    """

    def __str__(self):
        """
        Returns a string representation of the `SimpleFuzzySMLightweight` class.

        Returns:
            str: A string that indicates the class name with the suffix "-SimpleFuzzySMLightweight".
        """
        return super().__str__() + "-SimpleFuzzySMLightweight"

    def ratio_estimate(self) -> Any:
        """
        Returns the string matching ratio function from RapidFuzz.

        This method overrides the parent method to return the `ratio` function from RapidFuzz,
        which is used to calculate the basic fuzzy string matching score.

        Returns:
            Any: The `rapidfuzz.fuzz.ratio` function used for basic string similarity.
        """
        return rapidfuzz.fuzz.ratio


class WeightedFuzzySMLightweight(FuzzySMLightweight):
    """
    A subclass of `FuzzySMLightweight` that uses a weighted string similarity ratio from RapidFuzz.
    """

    def __str__(self):
        """
        Returns a string representation of the `WeightedFuzzySMLightweight` class.

        Returns:
            str: A string that indicates the class name with the suffix "-WeightedFuzzySMLightweight".
        """
        return super().__str__() + "-WeightedFuzzySMLightweight"

    def ratio_estimate(self) -> Any:
        """
        Returns the weighted string matching ratio function from RapidFuzz.

        This method overrides the parent method to return the `WRatio` function from RapidFuzz,
        which calculates a weighted fuzzy matching score between two strings.

        Returns:
            Any: The `rapidfuzz.fuzz.WRatio` function used for weighted string similarity.
        """
        return rapidfuzz.fuzz.WRatio


class TokenSetFuzzySMLightweight(FuzzySMLightweight):
    """
    A subclass of `FuzzySMLightweight` that uses the token set ratio for string similarity from RapidFuzz.
    """

    def __str__(self):
        """
        Returns a string representation of the `TokenSetFuzzySMLightweight` class.

        Returns:
            str: A string that indicates the class name with the suffix "-TokenSetFuzzySMLightweight".
        """
        return super().__str__() + "-TokenSetFuzzySMLightweight"

    def ratio_estimate(self) -> Any:
        """
        Returns the token set string matching ratio function from RapidFuzz.

        This method overrides the parent method to return the `token_set_ratio` function from RapidFuzz,
        which calculates similarity by comparing sets of tokens rather than the full string.

        Returns:
            Any: The `rapidfuzz.fuzz.token_set_ratio` function used for token set similarity.
        """
        return rapidfuzz.fuzz.token_set_ratio
