import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# 그레이스케일 로딩을 보장하는 커스텀 로더
def grayscale_loader(path):
    return Image.open(path).convert('L')

# 데이터 로더 정의
transform = transforms.Compose([
    transforms.ToTensor()  # [0, 255]를 [0, 1]로 변환
])

dataset = datasets.ImageFolder(
    root="/home/vilab/data/Training_Gray/정보저장매체",
    transform=transform,
    loader=grayscale_loader  # 커스텀 로더 사용
)
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=32)

# 평균과 표준편차 계산
mean = 0.0
var = 0.0
pixel_count = 0

for data, _ in tqdm(loader, desc="평균과 표준편차 계산 중"):
    batch_samples = data.size(0)
    data = data.view(batch_samples, -1)
    pixel_count += data.shape[1]
    mean += data.mean(1).sum()
    var += data.var(1).sum()

mean /= len(dataset)
var /= len(dataset)
std = torch.sqrt(var)

print(f"평균: {mean.item():.4f}")
print(f"표준편차: {std.item():.4f}")