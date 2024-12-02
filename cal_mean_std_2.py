from PIL import Image
import numpy as np

img_path = '/home/vilab/data/Training_Gray/위해물품/Arrow_tip_화살촉(151)/E3S690_20221109_03122026_S_Arrow tip_151-009_1.png'
img = Image.open(img_path).convert('L')  # 'L'은 그레이스케일 모드
img_array = np.array(img)

print(f"이미지 형태: {img_array.shape}")
print(f"최소값: {img_array.min()}, 최대값: {img_array.max()}")



import os
from PIL import Image
import numpy as np

dataset_path = "/home/vilab/data/Training_Gray/정보저장매체"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, file)
            img = Image.open(img_path).convert('L')
            img_array = np.array(img)
            print(f"파일: {file}")
            print(f"형태: {img_array.shape}")
            print(f"최소값: {img_array.min()}, 최대값: {img_array.max()}")
            print("---")
            
            # 처음 10개 이미지만 확인
            if files.index(file) == 9:
                break