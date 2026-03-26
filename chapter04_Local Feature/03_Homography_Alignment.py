import cv2 as cv  # OpenCV를 불러오는 이유 -> 이미지 읽기, SIFT 특징점 검출, 매칭, 호모그래피 계산, 원근 변환을 수행할 수 있음
import numpy as np  # NumPy를 불러오는 이유 -> 좌표 배열을 만들어 호모그래피 계산에 사용할 수 있음
import matplotlib.pyplot as plt  # matplotlib를 불러오는 이유 -> 매칭 결과와 정합 결과를 나란히 화면에 출력할 수 있음
from pathlib import Path  # Path를 불러오는 이유 -> 이미지 파일 경로를 안정적으로 지정할 수 있음

img1_path = Path("images/img1.jpg")  # 첫 번째 이미지 경로를 지정하는 이유 -> 기준이 되는 이미지를 불러오기 위함
img2_path = Path("images/img2.jpg")  # 두 번째 이미지 경로를 지정하는 이유 -> 첫 번째 이미지와 정합할 대상을 불러오기 위함

img1 = cv.imread(str(img1_path))  # 첫 번째 이미지를 읽는 이유 -> 기준 영상의 특징점을 검출하고 정합 기준으로 사용하기 위함
img2 = cv.imread(str(img2_path))  # 두 번째 이미지를 읽는 이유 -> 변환할 영상의 특징점을 검출하고 정합 대상으로 사용하기 위함

if img1 is None or img2 is None:  # 이미지가 정상적으로 읽혔는지 확인하는 이유 -> 경로 오류가 있으면 바로 확인할 수 있음
    raise FileNotFoundError("입력 이미지 파일(img1.jpg, img2.jpg)을 찾을 수 없습니다.")  # 오류를 발생시키는 이유 -> 이미지가 없을 때 잘못된 결과 없이 즉시 문제를 알 수 있음

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)  # 첫 번째 이미지를 그레이스케일로 바꾸는 이유 -> SIFT가 밝기 정보 기반으로 특징점을 검출하기 때문임
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)  # 두 번째 이미지를 그레이스케일로 바꾸는 이유 -> 두 이미지에서 같은 기준으로 특징점을 검출하기 위함

sift = cv.SIFT_create()  # SIFT 객체를 생성하는 이유 -> 두 이미지의 특징점과 descriptor를 추출할 수 있음

kp1, des1 = sift.detectAndCompute(gray1, None)  # 첫 번째 이미지의 특징점과 descriptor를 구하는 이유 -> 기준 이미지의 특징 정보가 계산됨
kp2, des2 = sift.detectAndCompute(gray2, None)  # 두 번째 이미지의 특징점과 descriptor를 구하는 이유 -> 정합 대상 이미지의 특징 정보가 계산됨

bf = cv.BFMatcher(cv.NORM_L2)  # BFMatcher를 생성하는 이유 -> SIFT descriptor 사이의 거리 기반 매칭을 수행할 수 있음
knn_matches = bf.knnMatch(des2, des1, k=2)  # 각 특징점마다 최근접 이웃 2개를 찾는 이유 -> ratio test를 적용해 좋은 매칭만 선별할 수 있음

good_matches = []  # 좋은 매칭만 저장할 리스트를 만드는 이유 -> 신뢰도 높은 대응점만 따로 관리할 수 있음

for pair in knn_matches:  # knn 매칭 결과를 하나씩 확인하는 이유 -> 각 특징점 쌍에 대해 ratio test를 적용할 수 있음
    if len(pair) == 2:  # 최근접 이웃이 2개인 경우만 사용하는 이유 -> ratio test를 정상적으로 적용할 수 있음
        m, n = pair  # 두 개의 최근접 이웃을 꺼내는 이유 -> 가장 가까운 매칭과 두 번째 매칭을 비교할 수 있음
        if m.distance < 0.7 * n.distance:  # 거리 비율 임계값 0.7을 적용하는 이유 -> 더 정확한 좋은 매칭만 남길 수 있음
            good_matches.append(m)  # 좋은 매칭만 저장하는 이유 -> 호모그래피 계산에 사용할 신뢰도 높은 대응점이 모임

if len(good_matches) < 4:  # 좋은 매칭이 4개 미만인지 확인하는 이유 -> 호모그래피 계산에는 최소 4개의 대응점이 필요함
    raise ValueError(f"호모그래피 계산에 필요한 좋은 매칭점이 부족합니다. 현재 개수: {len(good_matches)}")  # 오류를 발생시키는 이유 -> 대응점이 부족할 때 잘못된 정합을 막을 수 있음

src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 두 번째 이미지 좌표를 모으는 이유 -> 변환할 영상의 점들을 호모그래피 입력으로 사용하기 위함
dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 첫 번째 이미지 좌표를 모으는 이유 -> 기준 영상의 대응점들을 호모그래피 목표점으로 사용하기 위함

H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)  # RANSAC으로 호모그래피를 계산하는 이유 -> 이상점의 영향을 줄여 더 안정적인 변환 행렬을 얻을 수 있음

h1, w1 = img1.shape[:2]  # 첫 번째 이미지 크기를 구하는 이유 -> 정합 결과 캔버스의 기준 크기를 정할 수 있음
h2, w2 = img2.shape[:2]  # 두 번째 이미지 크기를 구하는 이유 -> 정합 결과 캔버스의 전체 크기를 정할 수 있음

panorama_width = w1 + w2  # 파노라마 가로 크기를 정하는 이유 -> 두 이미지를 넉넉하게 포함할 수 있는 출력 공간이 만들어짐
panorama_height = max(h1, h2)  # 파노라마 세로 크기를 정하는 이유 -> 두 이미지 중 더 큰 높이를 기준으로 출력 공간이 만들어짐

warped_img2 = cv.warpPerspective(img2, H, (panorama_width, panorama_height))  # 두 번째 이미지를 변환하는 이유 -> 첫 번째 이미지 기준 좌표계로 정렬된 결과가 생성됨
warped_img2[0:h1, 0:w1] = img1  # 첫 번째 이미지를 왼쪽 위에 복사하는 이유 -> 정렬된 두 이미지를 한 장의 결과 이미지로 확인할 수 있음

good_matches = sorted(good_matches, key=lambda x: x.distance)  # 좋은 매칭을 거리 기준으로 정렬하는 이유 -> 더 유사한 매칭이 앞쪽에 오도록 정리할 수 있음
top_matches = good_matches[:50]  # 상위 50개 매칭만 선택하는 이유 -> 너무 많은 선을 줄여서 매칭 결과를 보기 쉽게 만들 수 있음

match_img = cv.drawMatches(  # 매칭 결과 이미지를 만드는 이유 -> 두 이미지 사이의 대응 특징점을 선으로 연결해 시각화할 수 있음
    img2, kp2, img1, kp1, top_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # 좋은 매칭만 표시하는 이유 -> 신뢰도 높은 대응점만 깔끔하게 확인할 수 있음
)  # drawMatches 호출을 마무리하는 이유 -> 최종 매칭 시각화 이미지가 생성됨

match_img_rgb = cv.cvtColor(match_img, cv.COLOR_BGR2RGB)  # 매칭 결과를 RGB로 바꾸는 이유 -> matplotlib에서 색이 올바르게 보이게 할 수 있음
warped_img2_rgb = cv.cvtColor(warped_img2, cv.COLOR_BGR2RGB)  # 정합 결과를 RGB로 바꾸는 이유 -> matplotlib에서 색이 올바르게 보이게 할 수 있음

plt.figure(figsize=(18, 8))  # 큰 출력 창을 만드는 이유 -> 매칭 결과와 정합 결과를 한 화면에서 보기 좋게 표시할 수 있음

plt.subplot(1, 2, 1)  # 첫 번째 영역을 만드는 이유 -> 특징점 매칭 결과를 배치할 위치가 만들어짐
plt.imshow(match_img_rgb)  # 매칭 결과를 출력하는 이유 -> 두 이미지 간 대응 특징점 연결 상태를 확인할 수 있음
plt.title(f"Matching Result (Top 50 / Good Matches: {len(good_matches)})")  # 제목을 붙이는 이유 -> 몇 개의 좋은 매칭이 있었는지 바로 알 수 있음
plt.axis("off")  # 축을 숨기는 이유 -> 매칭 선과 이미지 자체에 집중해서 볼 수 있음

plt.subplot(1, 2, 2)  # 두 번째 영역을 만드는 이유 -> 정합 결과를 배치할 위치가 만들어짐
plt.imshow(warped_img2_rgb)  # 정합 결과를 출력하는 이유 -> 두 번째 이미지가 첫 번째 이미지 기준으로 정렬된 결과를 확인할 수 있음
plt.title("Warped / Aligned Image")  # 제목을 붙이는 이유 -> 이 이미지가 호모그래피로 정렬된 결과임을 바로 알 수 있음
plt.axis("off")  # 축을 숨기는 이유 -> 정합된 장면 자체를 더 깔끔하게 볼 수 있음

plt.tight_layout()  # 레이아웃을 정리하는 이유 -> 제목과 이미지가 겹치지 않고 정돈되어 보이게 함
plt.show()  # 결과 창을 띄우는 이유 -> 매칭 결과와 정합 결과를 최종적으로 확인할 수 있음

print(f"첫 번째 이미지 특징점 개수: {len(kp1)}")  # 첫 번째 이미지 특징점 수를 출력하는 이유 -> 기준 이미지의 특징점 양을 수치로 확인할 수 있음
print(f"두 번째 이미지 특징점 개수: {len(kp2)}")  # 두 번째 이미지 특징점 수를 출력하는 이유 -> 변환 이미지의 특징점 양을 수치로 확인할 수 있음
print(f"좋은 매칭 개수: {len(good_matches)}")  # 좋은 매칭 수를 출력하는 이유 -> ratio test 후 남은 신뢰도 높은 대응점 수를 확인할 수 있음
print("호모그래피 행렬:\n", H)  # 호모그래피 행렬을 출력하는 이유 -> 두 이미지 사이의 변환 관계를 수치로 확인할 수 있음