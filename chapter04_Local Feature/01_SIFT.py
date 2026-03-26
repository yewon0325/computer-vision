import cv2 as cv  # OpenCV를 불러오는 이유 -> SIFT 생성, 특징점 검출, 특징점 시각화를 수행할 수 있음
import matplotlib.pyplot as plt  # matplotlib를 불러오는 이유 -> 원본 이미지와 결과 이미지를 나란히 출력할 수 있음
from pathlib import Path  # Path를 불러오는 이유 -> 실행 위치가 달라도 이미지 경로를 안정적으로 찾을 수 있음

img_path = Path("images/mot_color70.jpg")  # 기본 이미지 경로를 만드는 이유 -> 과제 이미지 mot_color70.jpg를 읽기 위함

img = cv.imread(str(img_path))  # 컬러 이미지를 읽는 이유 -> 원본 장면에서 SIFT 특징점을 검출하기 위함

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 그레이스케일로 변환하는 이유 -> SIFT가 밝기 정보를 기준으로 특징점을 검출하기 때문임

sift_default = cv.SIFT_create()  # 기본 SIFT 객체를 만드는 이유 -> 기준이 되는 기본 검출 결과를 확인할 수 있음
kp_default, des_default = sift_default.detectAndCompute(gray, None)  # 기본 설정으로 특징점과 기술자를 구하는 이유 -> 기본 SIFT 결과가 계산됨

sift_tuned = cv.SIFT_create(nfeatures=250, contrastThreshold=0.03)  # 매개변수를 조정하는 이유 -> 특징점 개수를 제한하고 검출 결과를 비교할 수 있음
kp_tuned, des_tuned = sift_tuned.detectAndCompute(gray, None)  # 조정된 설정으로 특징점과 기술자를 구하는 이유 -> 더 정돈된 특징점 결과가 계산됨

img_default_kp = cv.drawKeypoints(img, kp_default, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 기본 특징점을 그리는 이유 -> 특징점의 위치, 크기, 방향이 시각화된 이미지가 생성됨
img_tuned_kp = cv.drawKeypoints(img, kp_tuned, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 조정된 특징점을 그리는 이유 -> 제한된 특징점의 위치, 크기, 방향이 시각화된 이미지가 생성됨

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 원본 이미지를 RGB로 바꾸는 이유 -> matplotlib에서 색이 올바르게 표시됨
img_default_kp_rgb = cv.cvtColor(img_default_kp, cv.COLOR_BGR2RGB)  # 기본 결과 이미지를 RGB로 바꾸는 이유 -> matplotlib에서 색이 정상적으로 보임
img_tuned_kp_rgb = cv.cvtColor(img_tuned_kp, cv.COLOR_BGR2RGB)  # 조정 결과 이미지를 RGB로 바꾸는 이유 -> matplotlib에서 색이 정상적으로 보임

plt.figure(figsize=(18, 6))  # 큰 출력 창을 만드는 이유 -> 원본과 두 결과를 한 화면에서 보기 좋게 표시할 수 있음

plt.subplot(1, 3, 1)  # 첫 번째 영역을 만드는 이유 -> 원본 이미지를 배치할 위치가 만들어짐
plt.imshow(img_rgb)  # 원본 이미지를 출력하는 이유 -> 특징점 검출 전 장면을 확인할 수 있음
plt.title("Original Image")  # 제목을 붙이는 이유 -> 어떤 이미지인지 바로 구분할 수 있음
plt.axis("off")  # 축을 숨기는 이유 -> 이미지 자체에 집중해서 볼 수 있음

plt.subplot(1, 3, 2)  # 두 번째 영역을 만드는 이유 -> 기본 SIFT 결과를 배치할 위치가 만들어짐
plt.imshow(img_default_kp_rgb)  # 기본 SIFT 결과를 출력하는 이유 -> 기본 설정으로 검출된 특징점을 확인할 수 있음
plt.title(f"Default SIFT ({len(kp_default)} keypoints)")  # 특징점 개수를 제목에 넣는 이유 -> 기본 결과의 검출량을 바로 비교할 수 있음
plt.axis("off")  # 축을 숨기는 이유 -> 특징점 시각화 결과가 더 잘 보이게 함

plt.subplot(1, 3, 3)  # 세 번째 영역을 만드는 이유 -> 조정된 SIFT 결과를 배치할 위치가 만들어짐
plt.imshow(img_tuned_kp_rgb)  # 조정된 SIFT 결과를 출력하는 이유 -> nfeatures를 제한한 검출 결과를 확인할 수 있음
plt.title(f"Tuned SIFT ({len(kp_tuned)} keypoints)")  # 특징점 개수를 제목에 넣는 이유 -> 조정 결과의 검출량을 바로 비교할 수 있음
plt.axis("off")  # 축을 숨기는 이유 -> 특징점 결과를 더 깔끔하게 볼 수 있음

plt.tight_layout()  # 여백을 자동 정리하는 이유 -> 제목과 이미지가 겹치지 않고 정돈되어 보임
plt.show()  # 최종 결과를 화면에 띄우는 이유 -> 원본과 두 SIFT 결과를 한 번에 확인할 수 있음

print(f"기본 SIFT 특징점 개수: {len(kp_default)}")  # 기본 특징점 수를 출력하는 이유 -> 텍스트로도 결과를 확인할 수 있음
print(f"조정 SIFT 특징점 개수: {len(kp_tuned)}")  # 조정 특징점 수를 출력하는 이유 -> 파라미터 변경 효과를 수치로 비교할 수 있음