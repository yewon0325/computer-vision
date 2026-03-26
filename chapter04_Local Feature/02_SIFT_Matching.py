import cv2 as cv  # OpenCV를 불러오는 이유 -> 이미지 읽기, SIFT 특징점 추출, 특징점 매칭, 결과 시각화를 수행할 수 있음
import matplotlib.pyplot as plt  # matplotlib를 불러오는 이유 -> 두 이미지의 매칭 결과를 화면에 출력할 수 있음
from pathlib import Path  # Path를 불러오는 이유 -> 이미지 파일 경로를 안정적으로 지정할 수 있음

img1_path = Path("images/mot_color70.jpg")  # 첫 번째 이미지 경로를 지정하는 이유 -> 기준 이미지 mot_color70.jpg를 불러오기 위함
img2_path = Path("images/mot_color83.jpg")  # 두 번째 이미지 경로를 지정하는 이유 -> 비교 이미지 mot_color83.jpg를 불러오기 위함

img1 = cv.imread(str(img1_path))  # 첫 번째 이미지를 읽는 이유 -> 첫 번째 영상에서 특징점을 검출하기 위함
img2 = cv.imread(str(img2_path))  # 두 번째 이미지를 읽는 이유 -> 두 번째 영상에서 특징점을 검출하기 위함

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)  # 첫 번째 이미지를 그레이스케일로 변환하는 이유 -> SIFT가 밝기 정보 기반으로 특징점을 검출하기 때문임
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)  # 두 번째 이미지를 그레이스케일로 변환하는 이유 -> 두 영상에서 같은 방식으로 특징점을 검출하기 위함

sift = cv.SIFT_create()  # SIFT 객체를 생성하는 이유 -> 두 이미지에서 SIFT 특징점과 descriptor를 추출할 수 있음

kp1, des1 = sift.detectAndCompute(gray1, None)  # 첫 번째 이미지의 특징점과 descriptor를 구하는 이유 -> 매칭에 사용할 기준 정보가 계산됨
kp2, des2 = sift.detectAndCompute(gray2, None)  # 두 번째 이미지의 특징점과 descriptor를 구하는 이유 -> 비교 대상 정보가 계산됨

index_params = dict(algorithm=1, trees=5)  # FLANN 인덱스 파라미터를 설정하는 이유 -> KD-Tree 기반으로 SIFT descriptor를 빠르게 탐색할 수 있음
search_params = dict(checks=50)  # FLANN 검색 파라미터를 설정하는 이유 -> 최근접 이웃 탐색 정확도를 높일 수 있음

flann = cv.FlannBasedMatcher(index_params, search_params)  # FLANN 매처를 생성하는 이유 -> 두 이미지의 descriptor를 효율적으로 매칭할 수 있음
knn_matches = flann.knnMatch(des1, des2, k=2)  # 각 특징점마다 최근접 이웃 2개를 찾는 이유 -> ratio test를 적용할 수 있는 매칭 결과가 생성됨

good_matches = []  # 좋은 매칭만 저장할 리스트를 만드는 이유 -> 부정확한 매칭을 걸러낸 결과를 따로 관리할 수 있음

for m, n in knn_matches:  # 최근접 이웃 2개를 순회하는 이유 -> 각 특징점 쌍에 대해 ratio test를 적용할 수 있음
    if m.distance < 0.75 * n.distance:  # ratio test를 적용하는 이유 -> 가장 가까운 매칭이 두 번째보다 충분히 좋을 때만 신뢰할 수 있음
        good_matches.append(m)  # 좋은 매칭만 추가하는 이유 -> 더 정확한 특징점 매칭 결과가 저장됨

good_matches = sorted(good_matches, key=lambda x: x.distance)  # 거리 기준으로 정렬하는 이유 -> 더 유사한 매칭이 앞쪽에 오도록 정리할 수 있음
top_matches = good_matches[:50]  # 상위 매칭 일부만 선택하는 이유 -> 너무 많은 선이 그려지지 않아 결과를 보기 쉽게 만들 수 있음

match_img = cv.drawMatches(  # 매칭 결과 이미지를 생성하는 이유 -> 두 이미지 사이의 대응 특징점을 선으로 연결해 시각화할 수 있음
    img1, kp1, img2, kp2, top_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # 좋은 매칭만 표시하는 이유 -> 신뢰도 높은 대응점만 깔끔하게 확인할 수 있음
)

match_img_rgb = cv.cvtColor(match_img, cv.COLOR_BGR2RGB)  # 결과 이미지를 RGB로 변환하는 이유 -> matplotlib에서 색이 올바르게 보이게 할 수 있음

plt.figure(figsize=(18, 8))  # 큰 출력 창을 만드는 이유 -> 두 이미지와 매칭 선들을 넓게 보기 좋게 표시할 수 있음
plt.imshow(match_img_rgb)  # 매칭 결과를 출력하는 이유 -> 두 영상 간 특징점 매칭 결과를 시각적으로 확인할 수 있음
plt.title(f"SIFT Feature Matching (Top 50 / Good Matches: {len(good_matches)})")  # 제목을 붙이는 이유 -> 좋은 매칭 개수와 출력 내용이 무엇인지 바로 알 수 있음
plt.axis("off")  # 축을 숨기는 이유 -> 이미지와 매칭 선 자체에 집중해서 볼 수 있음
plt.tight_layout()  # 레이아웃을 정리하는 이유 -> 제목과 이미지가 겹치지 않고 깔끔하게 출력됨
plt.show()  # 결과 창을 띄우는 이유 -> 최종 매칭 결과를 화면에서 확인할 수 있음

print(f"첫 번째 이미지 특징점 개수: {len(kp1)}")  # 첫 번째 이미지 특징점 수를 출력하는 이유 -> 검출된 특징점 양을 수치로 확인할 수 있음
print(f"두 번째 이미지 특징점 개수: {len(kp2)}")  # 두 번째 이미지 특징점 수를 출력하는 이유 -> 비교 이미지의 특징점 양을 수치로 확인할 수 있음
print(f"좋은 매칭 개수: {len(good_matches)}")  # 좋은 매칭 수를 출력하는 이유 -> ratio test 후 남은 신뢰도 높은 매칭 수를 확인할 수 있음