# -*- coding: utf-8 -*-
from typing import Any

import rapidfuzz

from .lightweight import FuzzySMLightweight


class SimpleFuzzySMLightweight(FuzzySMLightweight):
    def __str__(self):
        return super().__str__() + "+SimpleFuzzySM"

    def ratio_estimate(self) -> Any:
        return rapidfuzz.fuzz.ratio


class WeightedFuzzySMLightweight(FuzzySMLightweight):
    def __str__(self):
        return super().__str__() + "+WeightedFuzzySM"

    def ratio_estimate(self) -> Any:
        return rapidfuzz.fuzz.WRatio


class TokenSetFuzzySMLightweight(FuzzySMLightweight):
    def __str__(self):
        return super().__str__() + "+TokenSetFuzzySM"

    def ratio_estimate(self) -> Any:
        return rapidfuzz.fuzz.token_set_ratio
