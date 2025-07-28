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
import torch
from .graph import GraphEmbeddingAligner

class ConvEAligner(GraphEmbeddingAligner):
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

    def _similarity_matrix(self, source_onto_tensor, target_onto_tensor):
        return (source_onto_tensor @ target_onto_tensor.T).real

class HolEAligner(GraphEmbeddingAligner):
    model = "HolE"


class RotatEAligner(GraphEmbeddingAligner):
    model = "RotatE"

    def _similarity_matrix(self, source_onto_tensor, target_onto_tensor):
        return (source_onto_tensor @ target_onto_tensor.T).real


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

    def _similarity_matrix(self, source_onto_tensor, target_onto_tensor):
        return self.quat_similarity_normalized(source_onto_tensor, target_onto_tensor)

    def quat_mul(self, q, r):
        # q, r shape: (..., 4) where last dim is quaternion (w,x,y,z)
        # Returns quaternion product q * r
        w1, x1, y1, z1 = q.unbind(-1)
        w2, x2, y2, z2 = r.unbind(-1)
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)

    def quat_conj(self, q):
        # conjugate of quaternion
        w, x, y, z = q.unbind(-1)
        return torch.stack([w, -x, -y, -z], dim=-1)

    def quat_similarity(self, source, target):
        # source shape: [n_source, d, 4]
        # target shape: [n_target, d, 4]
        n_source, d, _ = source.shape
        # Normalize if needed
        source = source / source.norm(dim=-1, keepdim=True)
        target = target / target.norm(dim=-1, keepdim=True)
        # Expand dims for broadcasting
        source_exp = source.unsqueeze(1)  # [n_source, 1, d, 4]
        target_exp = target.unsqueeze(0)  # [1, n_target, d, 4] # n_target = target.shape[0]
        # Compute quaternion product: q * conj(r)
        prod = self.quat_mul(source_exp, self.quat_conj(target_exp))  # shape [n_source, n_target, d, 4]
        # Sum over embedding dim and take real part (w component)
        scores = prod[..., 0].sum(dim=-1)  # [n_source, n_target]
        return scores

    def quat_similarity_normalized(self, source, target):
        # source and target: [n_entities, embedding_dim, 4]
        # Normalize quaternions to unit length
        source = source / source.norm(dim=-1, keepdim=True)
        target = target / target.norm(dim=-1, keepdim=True)
        # Compute raw similarity scores
        raw_scores = self.quat_similarity(source, target)
        # Option 1: Min-max normalize
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        norm_scores = (raw_scores - min_score) / (max_score - min_score + 1e-8)
        return norm_scores

class SEAligner(GraphEmbeddingAligner):
    model = "SE"
