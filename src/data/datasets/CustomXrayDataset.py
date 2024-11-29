import gzip
import os
from typing import Tuple, Any, Callable, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# class CustomXrayDataset(Dataset):
#     def __init__(
#         self,
#         data_dir: str,
#         info_df: pd.DataFrame,
#         transform: Callable,
#         is_inference: bool = False
#     ):
#         # 데이터셋의 기본 경로, 이미지 변환 방법, 이미지 경로 및 레이블을 초기화합니다.
#         self.data_dir = data_dir  # 이미지 파일들이 저장된 기본 디렉토리
#         self.transform = transform  # 이미지에 적용될 변환 처리
#         self.is_inference = is_inference # 추론인지 확인
#         self.image_paths = info_df['image_path'].tolist()  # 이미지 파일 경로 목록

#         if not self.is_inference:
#             self.targets = info_df['target'].tolist()  # 각 이미지에 대한 레이블 목록

#     def __len__(self) -> int:
#         # 데이터셋의 총 이미지 수를 반환합니다.
#         return len(self.image_paths)

#     def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
#         # 주어진 인덱스에 해당하는 이미지를 로드하고 변환을 적용한 후, 이미지와 레이블을 반환합니다.
#         img_path = os.path.join(self.data_dir, self.image_paths[index])  # 이미지 경로 조합
#          # 파일 존재 여부 확인
#         if not os.path.exists(img_path):
#             print(f"경고: 이미지 파일을 찾을 수 없습니다: {img_path}")
#             # 대체 이미지를 반환하거나 이 항목을 건너뜁니다.
#             return None  # 또는 대체 이미지 반환
#         image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 이미지를 BGR 컬러 포맷의 numpy array로 읽어옵니다.
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR 포맷을 RGB 포맷으로 변환합니다.
#         #image = Image.fromarray(image)
#         image = np.array(image)
#         augmentations = self.transform(image=image)
#         image = augmentations["image"]  # 설정된 이미지 변환을 적용합니다.

#         if self.is_inference:
#             return image
#         else:
#             target = self.targets[index]  # 해당 이미지의 레이블
#             return image, target  # 변환된 이미지와 레이블을 튜플 형태로 반환합니다.


class CustomXrayDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        info_df: pd.DataFrame,
        transform: Callable,
        is_inference: bool = False
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.is_inference = is_inference

        # 이미지 경로와 레이블을 초기화
        image_paths = info_df['image_path'].tolist()
        if not self.is_inference:
            targets = info_df['target'].tolist()

        # 유효한 파일만 필터링
        self.image_paths = []
        if not self.is_inference:
            self.targets = []
            for img_path, target in zip(image_paths, targets):
                full_img_path = os.path.join(self.data_dir, img_path)
                if os.path.exists(full_img_path):
                    self.image_paths.append(img_path)
                    self.targets.append(target)
                # else:
                #     print(f"경고: 이미지 파일을 찾을 수 없습니다: {full_img_path}")
        else:
            for img_path in image_paths:
                full_img_path = os.path.join(self.data_dir, img_path)
                if os.path.exists(full_img_path):
                    self.image_paths.append(img_path)
                # else:
                #     print(f"경고: 이미지 파일을 찾을 수 없습니다: {full_img_path}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        img_path = os.path.join(self.data_dir, self.image_paths[index])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmentations = self.transform(image=image)
        image = augmentations["image"]

        if self.is_inference:
            return image
        else:
            target = self.targets[index]
            return image, target
