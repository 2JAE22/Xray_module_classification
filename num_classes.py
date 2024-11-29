import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('data/train.csv')
max_classnumber = df["target"].max()
print("max_num_classnumber: ",max_classnumber)
# 고유한 클래스 이름 추출
unique_classes = df['class_name'].unique()

# num_classes 계산
num_classes = len(unique_classes)

print(f"Unique Number of classes: {num_classes}")