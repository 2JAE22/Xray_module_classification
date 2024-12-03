import torch
import torch.nn as nn
from custom.softmax import gumbel_softmax
from custom.basemodel import BaseModel

class Router(nn.Module):
    def __init__(self, num_classes=474, backbone_output_size=768, shared_model=True):
        """
        Router: 입력 특징에 따라 적합한 하위 모델을 선택
        Args:
            num_classes: 분류 클래스 수
            backbone_output_size: ConvNeXt의 출력 크기
            shared_model: 하위 모델 간 파라미터를 공유할지 여부
        """
        super(Router, self).__init__()
        print(f"Router initializing with num_classes: {num_classes}, backbone_output_size: {backbone_output_size}")

        self.shared_model = shared_model

        # Selector: 각 모델의 선택 확률 계산
        self.selector = nn.Sequential(
            nn.Linear(backbone_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 3개의 모델 중 선택
        )

        # 모델 정의
        if shared_model:
            # 수정: BaseModel에 전달하는 인자 수정
            self.shared_base_model = BaseModel(
                input_features=backbone_output_size,
                output_features=num_classes,  # num_classes를 output_features로 전달
                dropout_rate=0.1  # 기본 드롭아웃 비율 설정
            )
        else:
            self.model1 = BaseModel(input_features=backbone_output_size, output_features=num_classes, dropout_rate=0.1)
            self.model2 = BaseModel(input_features=backbone_output_size, output_features=num_classes, dropout_rate=0.2)
            self.model3 = BaseModel(input_features=backbone_output_size, output_features=num_classes, dropout_rate=0.3)

    def forward(self, x, tau=1.0):
        logits = self.selector(x)  # [batch_size, 3]
        selection = gumbel_softmax(logits, tau=tau, hard=True)  # [batch_size, 3]

        if self.shared_model:
            # 공유된 모델 사용
            model_output = self.shared_base_model(x)
            outputs = torch.stack([model_output, model_output, model_output], dim=1)  # [batch_size, num_models, num_classes]
        else:
            # 독립된 모델 사용
            outputs = torch.stack([
                self.model1(x),
                self.model2(x),
                self.model3(x)
            ], dim=1)  # [batch_size, num_models, num_classes]

        # 선택된 모델만 활성화
        selected_output = torch.sum(outputs * selection.unsqueeze(-1), dim=1)
        return selected_output  # [batch_size, num_classes]


