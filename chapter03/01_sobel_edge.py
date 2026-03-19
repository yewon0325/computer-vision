import cv2 as cv                  # OpenCV 기능을 사용하기 위해 불러옴 → 이미지 처리 가능
import matplotlib.pyplot as plt   # Matplotlib을 사용하기 위해 불러옴 → 결과 화면 출력 가능

# 1. 이미지 불러오기
img = cv.imread("images/edgeDetectionImage.jpg")  # 파일에서 이미지를 읽음 → 컬러 원본 이미지가 저장됨                                      # 프로그램을 종료함 → 이후 코드 실행을 막음

# 2. 그레이스케일 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 컬러 이미지를 흑백으로 변환함 → 에지 검출이 쉬워짐

# 3. Sobel 필터로 x축, y축 방향 에지 검출
# 힌트에 따라 ksize=3 사용
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)  # x방향 변화량을 계산함 → 세로 경계가 강조됨
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)  # y방향 변화량을 계산함 → 가로 경계가 강조됨

# 4. 에지 강도(magnitude) 계산
magnitude = cv.magnitude(sobel_x, sobel_y)  # x,y 에지를 합쳐 강도를 계산함 → 전체 에지 세기가 구해짐

# 5. 시각화를 위해 uint8 형식으로 변환
edge_magnitude = cv.convertScaleAbs(magnitude)  # 결과를 8비트 영상으로 변환함 → 화면에 보기 쉽게 바뀜

# 6. Matplotlib으로 원본 이미지와 에지 강도 이미지 시각화
plt.figure(figsize=(12, 5))  # 출력 창 크기를 설정함 → 두 이미지를 넓게 비교 가능

# 원본 이미지 (OpenCV는 BGR이므로 RGB로 변환해서 출력)
plt.subplot(1, 2, 1)                           # 1행 2열 중 첫 번째 영역을 선택함 → 원본 이미지 위치가 정해짐
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) # 원본을 RGB로 변환해 출력함 → 색상이 정상적으로 보임
plt.title("Original Image")                    # 첫 번째 이미지 제목을 설정함 → 원본임을 알 수 있음
plt.axis("off")                                # 축을 숨김 → 이미지에만 집중할 수 있음

# 에지 강도 이미지
plt.subplot(1, 2, 2)                      # 1행 2열 중 두 번째 영역을 선택함 → 에지 결과 위치가 정해짐
plt.imshow(edge_magnitude, cmap='gray')   # 에지 강도 영상을 흑백으로 출력함 → 경계가 선명하게 보임
plt.title("Sobel Edge Magnitude")         # 두 번째 이미지 제목을 설정함 → Sobel 결과임을 알 수 있음
plt.axis("off")                           # 축을 숨김 → 결과 이미지가 깔끔하게 보임

plt.tight_layout()  # 그래프 간격을 자동 조정함 → 제목과 이미지가 겹치지 않음
plt.show()          # 최종 결과 창을 화면에 표시함 → 원본과 에지 결과가 함께 나타남