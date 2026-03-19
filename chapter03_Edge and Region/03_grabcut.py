import cv2 as cv                           # OpenCV 기능을 사용하기 위해 불러옴 → 이미지 읽기와 GrabCut 수행 가능
import numpy as np                         # NumPy 기능을 사용하기 위해 불러옴 → 마스크 계산과 배열 연산 가능
import matplotlib.pyplot as plt            # Matplotlib을 사용하기 위해 불러옴 → 결과 이미지를 화면에 출력 가능

img = cv.imread("images/coffee cup.JPG")   # images 폴더에서 커피컵 이미지를 읽어옴 → 원본 컬러 이미지가 저장됨

mask = np.zeros(img.shape[:2], np.uint8)   # 이미지 크기와 같은 2차원 마스크를 생성함 → 초기 분할용 빈 마스크가 만들어짐
bgdModel = np.zeros((1, 65), np.float64)   # 배경 모델 배열을 0으로 초기화함 → GrabCut이 배경 정보를 저장할 준비를 함
fgdModel = np.zeros((1, 65), np.float64)   # 전경 모델 배열을 0으로 초기화함 → GrabCut이 객체 정보를 저장할 준비를 함

rect = (120, 120, 1040, 750)               # 초기 사각형 영역을 지정함 → 컵과 접시가 포함된 전경 후보 영역이 설정됨

cv.grabCut(                                # GrabCut 알고리즘을 실행함 → 사각형 기준으로 전경/배경이 분할됨
    img,                                   # 원본 이미지를 입력함 → 분할 기준이 되는 실제 영상이 사용됨
    mask,                                  # 초기 마스크를 입력함 → 분할 결과가 이 마스크에 저장됨
    rect,                                  # 초기 사각형 영역을 입력함 → 이 영역을 중심으로 객체를 추출함
    bgdModel,                              # 배경 모델을 입력함 → 배경 통계 정보가 갱신됨
    fgdModel,                              # 전경 모델을 입력함 → 객체 통계 정보가 갱신됨
    5,                                     # 반복 횟수를 5로 설정함 → 분할 결과가 더 안정적으로 계산됨
    cv.GC_INIT_WITH_RECT                   # 사각형 기반 초기화를 사용함 → 지정한 영역으로 GrabCut을 시작함
)

mask2 = np.where(                          # GrabCut 마스크 값을 0과 1로 바꿈 → 배경 제거용 이진 마스크가 생성됨
    (mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),  # 확실한 배경과 배경일 가능성이 큰 영역을 찾음 → 제거할 부분이 선택됨
    0,                                     # 배경은 0으로 설정함 → 결과 이미지에서 사라지게 됨
    1                                      # 전경은 1로 설정함 → 결과 이미지에서 남게 됨
).astype("uint8")                          # 마스크를 uint8 형식으로 변환함 → 이미지 연산에 바로 사용 가능해짐

result = img * mask2[:, :, np.newaxis]     # 이진 마스크를 원본 이미지에 곱함 → 배경이 제거되고 객체만 남음

plt.figure(figsize=(15, 5))                # 출력 창 크기를 설정함 → 세 개의 이미지를 나란히 보기 좋게 만듦

plt.subplot(1, 3, 1)                       # 1행 3열 중 첫 번째 영역을 선택함 → 원본 이미지가 왼쪽에 표시됨
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))  # 원본 이미지를 RGB로 변환해 출력함 → 색상이 정상적으로 보임
plt.title("Original Image")                # 첫 번째 이미지 제목을 설정함 → 원본 이미지임을 알 수 있음
plt.axis("off")                            # 축을 숨김 → 이미지가 더 깔끔하게 보임

plt.subplot(1, 3, 2)                       # 1행 3열 중 두 번째 영역을 선택함 → 마스크 이미지가 가운데에 표시됨
plt.imshow(mask2 * 255, cmap="gray")       # 이진 마스크를 흑백으로 출력함 → 객체 영역은 흰색, 배경은 검은색으로 보임
plt.title("GrabCut Mask")                  # 두 번째 이미지 제목을 설정함 → GrabCut 마스크 결과임을 알 수 있음
plt.axis("off")                            # 축을 숨김 → 마스크 결과를 보기 쉽게 만듦

plt.subplot(1, 3, 3)                       # 1행 3열 중 세 번째 영역을 선택함 → 배경 제거 결과가 오른쪽에 표시됨
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))  # 배경 제거 결과를 RGB로 변환해 출력함 → 객체만 남은 이미지가 보임
plt.title("Foreground Extraction")         # 세 번째 이미지 제목을 설정함 → 객체 추출 결과임을 알 수 있음
plt.axis("off")                            # 축을 숨김 → 최종 결과가 깔끔하게 보임

plt.tight_layout()                         # 이미지와 제목 간격을 자동 조정함 → 화면 구성이 겹치지 않음
plt.show()                                 # 최종 결과를 화면에 출력함 → 원본, 마스크, 객체 추출 결과가 함께 나타남