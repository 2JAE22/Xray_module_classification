# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from timm import create_model

class SparseDispatcher(object): 
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1),
                            requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):
    def __init__(self, num_experts, output_size, noisy_gating=False, k=1, input_size = None):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        self.k = k
        assert(self.k <= self.num_experts)

        # 이미지 크기와 동일하게 맞춤
        # 이미지는 [3, 384, 384], 이를 flatten하면 3*384*384 = 442368
        self.image_flat_dim = 3 * 384 * 384 

        # 전문가 네트워크(ResNet18)
        self.experts = nn.ModuleList([
            create_model('resnet18', pretrained=True, num_classes=output_size)
            for _ in range(num_experts)
        ])

        # 게이팅 메커니즘 파라미터
        self.w_gate = nn.Parameter(torch.zeros(self.image_flat_dim, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.image_flat_dim, num_experts), requires_grad=True)
        
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        # x shape: [batch, image_flat_dim]
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        k = min(self.k, self.num_experts)
        top_count = min(k + 1, self.num_experts)
        top_logits, top_indices = logits.topk(top_count, dim=1)

        top_k_logits = top_logits[:, :k]
        top_k_indices = top_indices[:, :k]

        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)

        return gates, load

    def forward(self, x):
        # x: [batch, 3, 384, 384]

        # 이미지를 펼쳐서 게이팅 수행
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # [batch, 442368]

        # 게이팅
        gates, _ = self.noisy_top_k_gating(x_flat, self.training)
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_flat)

        # 전문가 입력을 다시 이미지 형태로 변환
        # expert_inputs[i]: [expert_batch_i, 442368]
        # -> [expert_batch_i, 3, 384, 384]
        expert_inputs = [ei.view(ei.size(0), 3, 384, 384) for ei in expert_inputs]

        # 전문가에 입력
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]

        # 출력 결합
        y = dispatcher.combine(expert_outputs)
        return y
