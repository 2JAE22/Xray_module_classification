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

from src.custom.router import Router
from src.custom.custom_convnext import ConvNeXtCustom


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
        loss = F.cross_entropy(y_hat, y)

        self.log("train_loss", loss,sync_dist = False)
        self.log("train_precision", self.precision(y_hat, y),sync_dist = False)
        self.log("train_recall", self.recall(y_hat, y),sync_dist = False)
        self.log("train_f1_score", self.f1_score(y_hat, y),sync_dist = False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        
        # 예측 결과 저장
        logits = F.softmax(y_hat, dim=1)
        preds = logits.argmax(dim=1)
        acc = torch.tensor(torch.sum(preds == y).item() / len(preds))
        # print("[Validation_acc]:", acc)

        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_precision", self.precision(y_hat, y))
        self.log("val_recall", self.recall(y_hat, y))
        self.log("val_f1_score", self.f1_score(y_hat, y))
        return loss

    def on_test_epoch_start(self):
        self.test_step_outputs = []  # 테스트 에포크 시작 시 초기화

    def test_step(self, batch, batch_idx):
        x = batch
        y_hat = self.forward(x)
        logits = F.softmax(y_hat, dim=1)
        preds = logits.argmax(dim=1)
        output = {"preds": preds.cpu().detach().numpy()} # 우리의 데이터셋에는 target 없음!
        self.test_step_outputs.append(output)  # 결과를 리스트에 추가
        return output

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        # preds = torch.cat([output["preds"] for output in outputs])
        preds = np.concatenate([output["preds"] for output in outputs])
        self.test_results["predictions"] = preds

        # 모든 test data에 대한 예측 한 리스트로 합치기!
        self.test_predictions.extend(preds) # csv 파일에 저장하기 위함!
        self.test_step_outputs.clear()  # 메모리 정리

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        lr = self.hparams.get("learning_rate", 2e-4) # sweep으로 lr 조정. / default: 2e-4
        optimizer_class = torch.optim.Adam

        # 옵티마이저 생성
        optimizer = optimizer_class(self.parameters(), lr=lr)

        if self.config.use_sweep is True: # sweep 사용 시
            # 스케줄러 설정
            if self.hparams.get("lr_scheduler") == 'StepLR':
                step_size = self.hparams.get("step_size", 1)  # Sweep에서 step_size 받기
                gamma = self.hparams.get("gamma", 0.7)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=step_size,
                    gamma=gamma
                )
            elif self.hparams.get("lr_scheduler") == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=self.trainer.max_epochs  # CosineAnnealing 스케줄러의 최대 에폭 설정
                )
            elif self.hparams.get("lr_scheduler") == 'ReduceLROnPlateau':  #scheduler_type은 정의되지 않았음 -> 따라서 self.hparams.get()으로 직접 인자 받아야함..
                patience = self.hparams.get("patience", 10)
                factor = self.hparams.get("factor", 0.1)
                scheduler = {
                    'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 
                        patience=patience, 
                        factor=factor
                    ),
                    'monitor': 'val_acc',
                    'interval': 'epoch',
                    'frequency': 1
                }
            else:
                scheduler = None
        else: # sweep 사용 X
            if hasattr(self.config, "scheduler"):
                scheduler_class = getattr(
                    torch.optim.lr_scheduler, self.config.scheduler.name
                )
                if self.config.scheduler.name == "ReduceLROnPlateau":
                    scheduler = {
                        'scheduler': scheduler_class(optimizer, **self.config.scheduler.params),
                        'monitor': 'val_acc'
                    }
                elif self.config.scheduler.name == "CosineAnnealingLR":
                    scheduler = scheduler_class(
                        optimizer, 
                        T_max=self.config.scheduler.params.get("T_max", self.trainer.max_epochs),
                        eta_min=self.config.scheduler.params.get("eta_min", 1e-6)
                    )
                else:
                    scheduler = scheduler_class(optimizer, **self.config.scheduler.params)

        if scheduler:
            if isinstance(scheduler, dict):  # ReduceLROnPlateau는 딕셔너리 형태로 반환
                return [optimizer], [scheduler]
            else:
                return [optimizer], [{'scheduler': scheduler}]
        else:
            return optimizer 
    # def training_step(se0lf, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.forward(x)

    #     # Cross-Entropy Loss 계산
    #     loss = F.cross_entropy(y_hat, y)

    #     # Metrics 계산
    #     precision = self.precision(y_hat, y)
    #     recall = self.recall(y_hat, y)
    #     f1_score = self.f1_score(y_hat, y)

    #     # Logging
    #     self.log("train_loss", loss, sync_dist=False)
    #     self.log("train_precision", precision, sync_dist=False)
    #     self.log("train_recall", recall, sync_dist=False)
    #     self.log("train_f1_score", f1_score, sync_dist=False)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.forward(x)

    #     # Cross-Entropy Loss 계산
    #     loss = F.cross_entropy(y_hat, y)

    #     # Metrics 계산
    #     precision = self.precision(y_hat, y)
    #     recall = self.recall(y_hat, y)
    #     f1_score = self.f1_score(y_hat, y)

    #     # Logging
    #     self.log("val_loss", loss, sync_dist=False)
    #     self.log("val_precision", precision, sync_dist=False)
    #     self.log("val_recall", recall, sync_dist=False)
    #     self.log("val_f1_score", f1_score, sync_dist=False)
    #     return loss

    # def configure_optimizers(self):
    #     lr = self.hparams.get("learning_rate", 2e-4)
    #     optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    #     if hasattr(self.config, "scheduler"):
    #         scheduler_class = getattr(torch.optim.lr_scheduler, self.config.scheduler.name)
    #         scheduler = scheduler_class(optimizer, **self.config.scheduler.params)
    #         return [optimizer], [{'scheduler': scheduler}]
    #     else:
    #         return optimizer


