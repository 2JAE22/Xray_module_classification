import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from timm import create_model
import torch.nn.functional as F

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
                            device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):
    def __init__(self, num_experts, output_size, noisy_gating=False, k=1):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        self.k = k
        assert self.k <= self.num_experts

        # 공유 레이어 (ResNet18 일부) 
        base_model = create_model('resnet18', pretrained=True)
        self.shared_layers = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.act1,
            base_model.maxpool,
            base_model.layer1,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # 필요 시 shared layer 동결로 연산량 감소
        for param in self.shared_layers.parameters():
            param.requires_grad = False

        gate_input_size = 64
        self.w_gate = nn.Linear(gate_input_size, num_experts)
        self.w_noise = nn.Linear(gate_input_size, num_experts) if self.noisy_gating else None

        self.experts = nn.ModuleList([
            nn.Linear(gate_input_size, output_size) for _ in range(num_experts)
        ])

        self.softmax = nn.Softmax(dim=1)
        self.softplus = nn.Softplus()
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("std", torch.tensor(1.0))

    def forward(self, x):
        # 입력 리사이즈
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # 공유 레이어 통과 (freeze했으므로 grad 필요 X)
        with torch.no_grad():
            shared_features = self.shared_layers(x_resized) # [B, 64]

        # 게이팅 로지츠
        clean_logits = self.w_gate(shared_features)
        if self.noisy_gating and self.training:
            raw_noise_stddev = self.w_noise(shared_features)
            noise_stddev = self.softplus(raw_noise_stddev) + 1e-2
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits

        logits = self.softmax(logits)

        # 상위 k개 추출
        k = min(self.k, self.num_experts)
        top_logits, top_indices = logits.topk(k, dim=1)
        top_k_gates = top_logits / (top_logits.sum(1, keepdim=True) + 1e-6)
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_indices, top_k_gates)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(shared_features)

        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]

        y = dispatcher.combine(expert_outputs)
        return y
