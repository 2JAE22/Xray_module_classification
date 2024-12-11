from timm import create_model
import torch.nn as nn
from src.custom2.moe import MoE
import torch

class ConvNeXtCustom(nn.Module):
    def __init__(self, pretrained=True, hidden_size=128, k=3, num_classes=474, num_experts=None):
        super(ConvNeXtCustom, self).__init__()
        # ConvNeXt 모델 생성 (구조 변경 없이 그대로 불러옴)
        full_model = create_model("convnext_xxlarge", pretrained=pretrained, features_only=False)

        # convnext_xxlarge의 특정 stage까지만 사용 (구조를 변경하지 않고, forward에서 사용 부분만 조정)
        self.stem = full_model.stem
        self.stage0 = full_model.stages[0]
        self.stage1 = full_model.stages[1]

        # feature_dim 추출
        self.feature_dim = full_model.stages[1].blocks[-1].conv_dw.weight.shape[0]

        # 여기서 full_model의 나머지 부분을 삭제하지 않고 참조하지 않음으로써 원본 베이스 모델 변경 최소화
        # 필요하다면 forward 함수에서 stage2 이후를 호출하지 않으면 됨.
        # del full_model.stages[2:]
        # del full_model.norm_pre
        # del full_model.head

        # MoE 초기화
        if num_experts is None:
            raise ValueError("num_experts must be provided")

        self.moe = MoE(
            num_experts=num_experts,
            output_size=hidden_size,
            noisy_gating=True,
            k=k
        )

        # ConvNeXt의 features를 Global Average Pooling으로 축소
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 최종 classifier 입력 크기: feature_dim + hidden_size
        combined_feature_size = self.feature_dim + hidden_size
        self.classifier = nn.Linear(combined_feature_size, num_classes)

    def forward(self, x):
        # ConvNeXt 특징 추출
        features = self.stem(x)
        features = self.stage0(features)
        features = self.stage1(features)

        # Global Average Pooling 적용
        features = self.avgpool(features)  # [B, C, 1, 1]
        features = features.view(features.size(0), -1)  # [B, feature_dim]

        # MoE 출력을 얻음
        moe_output = self.moe(x)  # [B, hidden_size]

        # MoE와 ConvNeXt feature concat
        combined = torch.cat([features, moe_output], dim=1)

        # 분류기 통과
        output = self.classifier(combined)
        return output
