from timm import create_model
import torch.nn as nn

from timm.models.convnext import ConvNeXt
from timm.models.layers import SelectAdaptivePool2d

### custom ConvNeXtCustom 모델
class ConvNeXtCustom(nn.Module):
    """
    필요한 stage만 포함한 커스텀 ConvNeXt 모델
    """
    def __init__(self, pretrained=True):
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

    def forward(self, x):
        # 필요한 stage만 호출
        x = self.stem(x)
        x = self.stage0(x)
        x = self.stage1(x)

        return x
