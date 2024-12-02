import matplotlib.pyplot as plt
import torch
import timm
from src.models.custiom_model import Router
import torch.nn as nn

class ConvNeXtWithRouter(nn.Module):
    def __init__(self, pretrained=True):
        super(ConvNeXtWithRouter, self).__init__()
        self.convnext = timm.create_model('convnext_base', pretrained=pretrained)
        self.stem = self.convnext.stem
        self.stages = self.convnext.stages
        self.router = Router()  # Router 추가

    def forward(self, x):
        x = self.stem(x)
        stage1_out = self.stages[0](x)
        
        # 특징 맵 시각화
        print("Stage 1 Output Shape:", stage1_out.shape)
        visualize_features(stage1_out, num_channels=6)
        
        # Router로 전달
        routing_probs = self.router(stage1_out)
        return routing_probs

    
def visualize_features(features, num_channels=4):
    """
    특징 맵 시각화 함수.
    - features: Stage 1 출력 (Batch, Channels, Height, Width)
    - num_channels: 시각화할 채널 수
    """
    batch_size, channels, height, width = features.shape
    features = features[0]  # 첫 번째 배치 선택 (Channels, Height, Width)
    
    # 시각화할 채널 수 제한
    num_channels = min(channels, num_channels)
    
    plt.figure(figsize=(15, 5))
    for i in range(num_channels):
        plt.subplot(1, num_channels, i + 1)
        plt.imshow(features[i].detach().cpu().numpy(), cmap='viridis')  # 채널 i 시각화
        plt.title(f"Channel {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 모델 생성
model = ConvNeXtWithRouter(pretrained=True)

# 임의의 입력 데이터 생성
x = torch.randn(1, 3, 224, 224)  # (Batch=1, Channels=3, Height=224, Width=224)

# Stage 1 출력
stage1_out = model(x)

# 시각화
print("Stage 1 Output Shape:", stage1_out.shape)  # (1, 384, 56, 56)
visualize_features(stage1_out, num_channels=6)  # 상위 6개 채널 시각화
