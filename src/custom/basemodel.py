import torch.nn as nn

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
