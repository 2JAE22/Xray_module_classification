import os
import json
import pandas as pd

# 결과를 저장할 DataFrame 초기화
data_list = []

# JSON 파일이 있는 기본 디렉토리
base_dir = '/home/vilab/data/Training_Color/02.라벨링데이터'

# 디렉토리를 재귀적으로 탐색
for root, dirs, files in os.walk(base_dir):
    for file in files:
        # JSON 파일만 처리
        if file.endswith('.json'):
            json_file_path = os.path.join(root, file)  # JSON 파일의 전체 경로
            with open(json_file_path, 'r') as json_file:
                try:
                    data = json.load(json_file)  # JSON 데이터 로드

                    # 필요한 정보 추출
                    class_name = data['categories'][0]['name']  # 클래스 이름 추출
                    image_file_name = data['images'][0]['file_name']  # 이미지 파일 이름 추출

                    # 현재 폴더 이름 추출
                    folder_name = os.path.basename(root)  # 현재 JSON 파일이 있는 폴더 이름 추출

                    # 앞부분의 접두사 제거 ('위해물품_' 또는 '일반물품_' 등)
                    if '_' in folder_name:
                        folder_name = folder_name.split('_', 1)[-1]  # 첫 번째 밑줄 이후의 문자열만 사용

                    # 부모 폴더 이름 추출
                    parent_dir = os.path.basename(os.path.dirname(root))  # 부모 폴더 이름 추출

                    # 부모 폴더 이름에서 'TL_' 접두사 제거
                    if parent_dir.startswith('TL_'):
                        parent_dir = parent_dir[3:]  # 'TL_' 접두사 제거

                    # 이미지 파일의 상대 경로 생성
                    image_path = os.path.join(parent_dir, folder_name, image_file_name).replace('\\', '/')

                    target = data['categories'][0]['id']  # 타겟 ID 추출

                    # DataFrame에 추가할 데이터 생성
                    data_list.append({
                        'class_name': class_name,
                        'image_path': image_path,
                        'target': target
                    })
                except (json.JSONDecodeError, IndexError) as e:
                    print(f"Error processing {json_file_path}: {e}")  # 오류 발생 시 메시지 출력

# DataFrame 생성
df = pd.DataFrame(data_list)

# CSV 파일로 저장
df.to_csv('train.csv', index=False)
