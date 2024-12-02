import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from timm.models.convnext import ConvNeXt
from timm.models.layers import SelectAdaptivePool2d


def gumbel_softmax(logits, tau=1.0, hard=True):
    """
    Gumbel-Softmax 샘플링 함수
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = logits + gumbel_noise
    y = F.softmax(y / tau, dim=-1)
    if hard:
        y_hard = torch.zeros_like(y).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y  # Gradient는 softmax를 따라가지만, 결과는 one-hot
    return y

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


### router이후에 적용될 모델들
class BaseModel(nn.Module):
    """
    Base model class with common functionality for Model1, Model2, Model3.
    """
    def __init__(self, input_features, output_features, dropout_rate=0.0):
        super(BaseModel, self).__init__()
        self.head = self._build_head(input_features, output_features, dropout_rate)

    def _build_head(self, input_features, output_features, dropout_rate):
        return nn.Sequential(
            nn.LayerNorm(input_features),
            nn.Linear(input_features, output_features), 
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        return self.head(x)


class Model1(BaseModel):
    def __init__(self, input_features):
        super(Model1, self).__init__(input_features, output_features=474, dropout_rate=0.1)


class Model2(BaseModel):
    def __init__(self, input_features):
        super(Model2, self).__init__(input_features, output_features=474, dropout_rate=0.2)


class Model3(BaseModel):
    def __init__(self, input_features):
        super(Model3, self).__init__(input_features, output_features=474, dropout_rate=0.3)


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

### 이 XrayModel2가 작동하는 것.
class XrayModel2(pl.LightningModule):
    def __init__(self, config):
        super(XrayModel2, self).__init__()
        self.config = config

        # 커스텀 ConvNeXt 백본 사용
        self.backbone = ConvNeXtCustom(pretrained=config.model.pretrained)

        # Router 초기화
        self.router = Router(num_classes=474, backbone_output_size=768, shared_model=True)

        # Metrics 정의
        self.precision = MulticlassPrecision(num_classes=474, average="macro")
        self.recall = MulticlassRecall(num_classes=474, average="macro")
        self.f1_score = MulticlassF1Score(num_classes=474, average="macro")
        self.wandb_logger = WandbLogger(project="Xray", name="Xray_TEST")

    def forward(self, x):
        features = self.backbone(x)  # ConvNeXt의 필요한 stage 출력만 사용
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1))  # Global Pooling
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # Flatten
        return self.router(pooled_features)  # Router로 전달


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        # Cross-Entropy Loss 계산
        loss = F.cross_entropy(y_hat, y)

        # Metrics 계산
        precision = self.precision(y_hat, y)
        recall = self.recall(y_hat, y)
        f1_score = self.f1_score(y_hat, y)

        # Logging
        self.log("train_loss", loss, sync_dist=False)
        self.log("train_precision", precision, sync_dist=False)
        self.log("train_recall", recall, sync_dist=False)
        self.log("train_f1_score", f1_score, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        # Cross-Entropy Loss 계산
        loss = F.cross_entropy(y_hat, y)

        # Metrics 계산
        precision = self.precision(y_hat, y)
        recall = self.recall(y_hat, y)
        f1_score = self.f1_score(y_hat, y)

        # Logging
        self.log("val_loss", loss, sync_dist=False)
        self.log("val_precision", precision, sync_dist=False)
        self.log("val_recall", recall, sync_dist=False)
        self.log("val_f1_score", f1_score, sync_dist=False)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.get("learning_rate", 2e-4)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if hasattr(self.config, "scheduler"):
            scheduler_class = getattr(torch.optim.lr_scheduler, self.config.scheduler.name)
            scheduler = scheduler_class(optimizer, **self.config.scheduler.params)
            return [optimizer], [{'scheduler': scheduler}]
        else:
            return optimizer
