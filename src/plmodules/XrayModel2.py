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

from custom.router import Router
from custom.custom_convnext import ConvNeXtCustom


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
