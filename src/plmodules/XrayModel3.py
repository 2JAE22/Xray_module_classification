import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm import create_model
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from src.custom2.moe import MoE  # MoE 클래스 임포트

class XrayModel3(pl.LightningModule):
    def __init__(self, config):
        super(XrayModel3, self).__init__()
        self.config = config

        # MoE 모델 초기화
        self.moe = MoE(
            input_size=config.model.input_size,
            output_size=config.model.output_size,
            num_experts=config.model.num_experts,
            noisy_gating=config.model.noisy_gating,
            k=config.model.k
        )

        # Metrics 정의
        self.precision = MulticlassPrecision(num_classes=config.model.output_size, average="macro")
        self.recall = MulticlassRecall(num_classes=config.model.output_size, average="macro")
        self.f1_score = MulticlassF1Score(num_classes=config.model.output_size, average="macro")
        self.wandb_logger = WandbLogger(project="Xray", name="Xray_TEST")

    def forward(self, x):
        
        # MoE 모델을 통해 예측
        y = self.moe(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("train_loss", loss, sync_dist=False)
        self.log("train_precision", self.precision(y_hat, y), sync_dist=False)
        self.log("train_recall", self.recall(y_hat, y), sync_dist=False)
        self.log("train_f1_score", self.f1_score(y_hat, y), sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        logits = F.softmax(y_hat, dim=1)
        preds = logits.argmax(dim=1)
        acc = torch.tensor(torch.sum(preds == y).item() / len(preds))

        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_precision", self.precision(y_hat, y))
        self.log("val_recall", self.recall(y_hat, y))
        self.log("val_f1_score", self.f1_score(y_hat, y))
        return loss

    def configure_optimizers(self):
        lr = self.hparams.get("learning_rate", 2e-4)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer


