import os
import json
import pandas as pd

# 결과를 저장할 DataFrame 초기화
data_list = []

# JSON 파일이 있는 기본 디렉토리
base_dir = 'data/02.라벨링데이터/'

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
                    
                    # 이미지 파일의 경로 생성
                    # JSON 파일이 있는 디렉토리와 이미지 파일 이름을 결합하여 전체 경로 생성
                    image_path = os.path.join(root, image_file_name).replace('\\', '/')  # 경로 조정

                    target = data['categories'][0]['id']  # 타겟 ID 추출

                    # DataFrame에 추가할 데이터 생성
                    data_list.append({
                        'class_name': class_name,
                        'image_path': image_path,  # 이제 .json 대신 이미지 파일 경로가 들어감
                        'target': target
                    })
                except (json.JSONDecodeError, IndexError) as e:
                    print(f"Error processing {json_file_path}: {e}")  # 오류 발생 시 메시지 출력

# DataFrame 생성
df = pd.DataFrame(data_list)

# CSV 파일로 저장
df.to_csv('train.csv', index=False)