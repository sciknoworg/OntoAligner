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
from . graph import GraphEmbeddingAligner

class CovEAligner(GraphEmbeddingAligner):
    model = "ConvE"


class TransDAligner(GraphEmbeddingAligner):
    model = "TransD"


class TransEAligner(GraphEmbeddingAligner):
    model = "TransE"


class TransFAligner(GraphEmbeddingAligner):
    model = "TransF"


class TransHAligner(GraphEmbeddingAligner):
    model = "TransH"


class TransRAligner(GraphEmbeddingAligner):
    model = "TransR"


class DistMultAligner(GraphEmbeddingAligner):
    model = "DistMult"


class ComplExAligner(GraphEmbeddingAligner):
    model = "ComplEx"


class HolEAligner(GraphEmbeddingAligner):
    model = "HolE"


class RotatEAligner(GraphEmbeddingAligner):
    model = "RotatE"


class SimplEAligner(GraphEmbeddingAligner):
    model = "SimplE"


class CrossEAligner(GraphEmbeddingAligner):
    model = "CrossE"


class BoxEAligner(GraphEmbeddingAligner):
    model = "BoxE"


class CompGCNAligner(GraphEmbeddingAligner):
    model = "CompGCN"


class MuREAligner(GraphEmbeddingAligner):
    model = "MuRE"


class QuatEAligner(GraphEmbeddingAligner):
    model = "QuatE"


class RGCNAligner(GraphEmbeddingAligner):
    model = "RGCN"

class SEAligner(GraphEmbeddingAligner):
    model = "SE"
