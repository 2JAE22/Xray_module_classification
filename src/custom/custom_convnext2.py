from timm import create_model
import torch.nn as nn
from src.moe_github.mixture_of_experts.moe import MoE  # Import the MoE class

class ConvNeXtCustom(nn.Module):
    """
    필요한 stage만 포함한 커스텀 ConvNeXt 모델
    """
    def __init__(self, pretrained=True, num_experts=4, hidden_size=256, k=2):
        super(ConvNeXtCustom, self).__init__()

        # ConvNeXt 전체 모델 생성
        full_model = create_model(
            "convnext_xxlarge",
            pretrained=pretrained,
            features_only=False
        )

        # 필요한 부분만 유지
        self.stem = full_model.stem  # 입력 계층
        self.stage0 = full_model.stages[0]  # stage[0]
        self.stage1 = full_model.stages[1]  # stage[1]
        self.head = full_model.head  # 기존 Head

        # Add MoE layer
        self.moe = MoE(
            input_size=full_model.num_features,  # Assuming this is the output size of stage1
            output_size=full_model.num_features,  # Adjust as needed
            num_experts=num_experts,
            hidden_size=hidden_size,
            k=k
        )

    def forward(self, x):
        # 필요한 stage만 호출
        x = self.stem(x)
        x = self.stage0(x)
        x = self.stage1(x)

        # Pass through MoE layer
        x, _ = self.moe(x)

        return x