import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from src.custom2.custom_convnext2 import ConvNeXtCustom

class XrayModel3(pl.LightningModule):
    def __init__(self, config):
        super(XrayModel3, self).__init__()
        self.config = config
        self.backbone = ConvNeXtCustom(
            pretrained=config.model.pretrained,
            num_experts=config.model.num_experts,
            hidden_size=config.model.hidden_size,
            k=config.model.k,
            num_classes=config.model.output_size
        )

        self.precision = MulticlassPrecision(num_classes=config.model.output_size, average="macro")
        self.recall = MulticlassRecall(num_classes=config.model.output_size, average="macro")
        self.f1_score = MulticlassF1Score(num_classes=config.model.output_size, average="macro")

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # 필요시 sync_dist는 True로 해도 됨. 단 single gpu면 의미 없음.
        self.log("train_loss", loss)
        self.log("train_precision", self.precision(y_hat, y))
        self.log("train_recall", self.recall(y_hat, y))
        self.log("train_f1_score", self.f1_score(y_hat, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_precision", self.precision(y_hat, y))
        self.log("val_recall", self.recall(y_hat, y))
        self.log("val_f1_score", self.f1_score(y_hat, y))
        return loss

    def configure_optimizers(self):
        lr = self.config.optimizer.params.learning_rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
