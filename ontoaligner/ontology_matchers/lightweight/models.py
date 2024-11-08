# -*- coding: utf-8 -*-
from typing import Any

import rapidfuzz

from .lightweight import FuzzySMLightweight


class SimpleFuzzySMLightweight(FuzzySMLightweight):
    def __str__(self):
        return super().__str__() + "-SimpleFuzzySMLightweight"

    def ratio_estimate(self) -> Any:
        return rapidfuzz.fuzz.ratio


class WeightedFuzzySMLightweight(FuzzySMLightweight):
    def __str__(self):
        return super().__str__() + "-WeightedFuzzySMLightweight"

    def ratio_estimate(self) -> Any:
        return rapidfuzz.fuzz.WRatio


class TokenSetFuzzySMLightweight(FuzzySMLightweight):
    def __str__(self):
        return super().__str__() + "-TokenSetFuzzySMLightweight"

    def ratio_estimate(self) -> Any:
        return rapidfuzz.fuzz.token_set_ratio
