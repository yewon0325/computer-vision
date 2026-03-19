import cv2 as cv                  # OpenCV 기능을 사용하기 위해 불러옴 → 이미지 처리와 직선 그리기가 가능해짐
import matplotlib.pyplot as plt   # Matplotlib을 사용하기 위해 불러옴 → 결과 이미지를 화면에 출력할 수 있음
import numpy as np                # NumPy를 사용하기 위해 불러옴 → 허프 변환 각도 값을 설정할 수 있음

# 1. 이미지 불러오기
img = cv.imread("images/dabo.jpg")  # 다보탑 이미지를 읽어옴 → 원본 컬러 이미지가 저장됨

# 2. 원본 이미지를 복사
line_img = img.copy()  # 원본을 복사함 → 직선을 그려도 원본 이미지는 유지됨

# 3. 그레이스케일 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 컬러 이미지를 흑백으로 변환함 → 에지 검출이 쉬워짐

# 4. Canny 에지 검출
# 힌트에 따라 threshold1=100, threshold2=200 사용
edges = cv.Canny(gray, 100, 200)  # 밝기 변화가 큰 경계를 검출함 → 흰색 에지 맵이 생성됨

# 5. HoughLinesP를 사용하여 직선 검출
# rho, theta, threshold, minLineLength, maxLineGap 값을 조절
lines = cv.HoughLinesP(            # 에지 맵에서 직선 후보를 찾음 → 직선 좌표 정보가 반환됨
    edges,                         # 캐니 에지 결과를 입력함 → 경계선 기반으로 직선을 검출함
    rho=1,                         # 거리 해상도를 1픽셀로 설정함 → 세밀하게 직선을 탐색함
    theta=np.pi / 180,             # 각도 해상도를 1도로 설정함 → 다양한 방향의 직선을 검출함
    threshold=120,                 # 직선으로 인정할 최소 투표 수를 설정함 → 강한 직선만 남게 됨
    minLineLength=80,              # 최소 직선 길이를 설정함 → 너무 짧은 선은 제외됨
    maxLineGap=10                  # 선 사이 최대 간격을 설정함 → 끊긴 선을 하나로 연결할 수 있음
)

# 6. 검출된 직선을 원본 이미지에 그리기
if lines is not None:                          # 검출된 직선이 있는지 확인함 → 직선이 있을 때만 그림
    for line in lines:                        # 검출된 직선을 하나씩 꺼냄 → 모든 직선을 순서대로 처리함
        x1, y1, x2, y2 = line[0]             # 직선의 시작점과 끝점 좌표를 저장함 → 그릴 위치가 정해짐
        # 힌트에 따라 빨간색 (0, 0, 255), 두께 2 사용
        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 직선을 빨간색으로 그림 → 검출 결과가 원본 위에 표시됨

# 7. Matplotlib으로 원본 이미지와 직선 검출 결과 시각화
plt.figure(figsize=(14, 6))  # 출력 창 크기를 설정함 → 두 이미지를 넓게 비교할 수 있음

# 원본 이미지
plt.subplot(1, 2, 1)                           # 1행 2열 중 첫 번째 영역을 선택함 → 원본 이미지가 왼쪽에 배치됨
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) # 원본 이미지를 RGB로 변환해 출력함 → 색상이 올바르게 보임
plt.title("Original Image")                    # 첫 번째 이미지 제목을 설정함 → 원본 이미지임을 알 수 있음
plt.axis("off")                                # 축을 숨김 → 이미지가 더 깔끔하게 보임

# 직선 검출 결과 이미지
plt.subplot(1, 2, 2)                                # 1행 2열 중 두 번째 영역을 선택함 → 결과 이미지가 오른쪽에 배치됨
plt.imshow(cv.cvtColor(line_img, cv.COLOR_BGR2RGB)) # 직선이 그려진 이미지를 RGB로 변환해 출력함 → 검출 결과가 정상 색상으로 보임
plt.title("Canny + Hough Line Detection")           # 두 번째 이미지 제목을 설정함 → 직선 검출 결과임을 알 수 있음
plt.axis("off")                                     # 축을 숨김 → 결과를 보기 쉽게 정리함

plt.tight_layout()  # 이미지와 제목 간격을 자동으로 조정함 → 화면 구성이 겹치지 않게 됨
plt.show()          # 최종 결과를 화면에 출력함 → 원본과 직선 검출 결과가 함께 나타남