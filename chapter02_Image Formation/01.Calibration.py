import cv2  # OpenCV 함수(코너 검출/캘리브레이션/왜곡보정)를 쓰기 위해 임포트 → 이후 calibrate/undistort 가능
import numpy as np  # 수치연산/좌표배열 생성에 사용 → objp, 슬라이싱, hstack 등에 사용
from pathlib import Path  # 경로를 OS에 안전하게 다루기 위해 사용 → 한글 경로 포함 파일 탐색 가능

# (A) 한글/특수문자 경로에서도 안전한 imread: cv2.imread가 한글 경로에서 실패할 수 있어 우회 로더를 사용
def imread_unicode(path, flags=cv2.IMREAD_COLOR):  # 한글 경로에서도 이미지를 읽기 위한 함수 정의 
    data = np.fromfile(str(path), dtype=np.uint8)  # 파일을 바이트로 읽음 → 경로 인코딩 문제를 피함
    img = cv2.imdecode(data, flags)  # 바이트를 이미지로 디코딩 → 실제 이미지 배열(BGR 등) 반환
    return img  # 디코딩된 이미지 반환 → 이후 처리(cv2.cvtColor 등)에 사용

# (0) 설정값: 체커보드 크기/스케일/정밀화 조건을 지정해서 코너 검출과 보정을 안정화
CHECKERBOARD = (9, 6)      # 내부 코너 개수 (가로, 세로) → findChessboardCorners가 이 개수로 코너를 찾음
square_size = 25.0         # 한 칸 크기(mm) → 실제(3D) 좌표 스케일을 mm로 맞춤
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 서브픽셀 반복 조건 → 코너 좌표가 더 정확해짐

# (1) 실제(3D) 좌표 준비  # 체크보드의 실제 격자 좌표를 만들고, 이미지 코너(2D)와 매칭하기 위함
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)  # (N,3) 3D 점 배열 생성 → (x,y,z) 형태
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)  # (x,y) 격자 좌표 채움 → z=0 평면 가정
objp *= square_size  # 실제 단위(mm)로 스케일 적용 → depth/왜곡 계수 추정에 실제 크기 반영

objpoints = []  # 각 이미지의 3D 실제 좌표 목록 → calibrateCamera 입력으로 사용
imgpoints = []  # 각 이미지의 2D 코너 좌표 목록 → calibrateCamera 입력으로 사용

# (2) 이미지 목록 수집: 캘리브레이션에 사용할 여러 장의 체크보드 이미지를 모으기 위함
img_dir = Path("L02 실습/images/calibration_images")  # 체크보드 이미지 폴더 경로 지정 
images = sorted(img_dir.glob("left*.jpg"))  # left01~left13 같은 파일 목록 수집 → 캘리브레이션 입력 이미지들

if not images:  # 이미지가 하나도 없으면 진행 불가 → 파일 경로 문제를 바로 알림
    raise FileNotFoundError(f"이미지 없음: {img_dir} / left*.jpg 경로 확인")  # 실행 즉시 중단 → 경로 수정 필요

print(f"[INFO] Found {len(images)} images")  # 몇 장을 찾았는지 출력 → 정상적으로 파일을 찾았는지 확인

img_size = None  # 이미지 크기(W,H)를 저장할 변수 → calibrateCamera에 필요

# (3) 코너 검출 + 대응점 수집  # 각 이미지에서 코너를 찾아 (3D-2D) 대응점을 쌓기 위함
for fname in images:  # 이미지 목록을 하나씩 처리 → 성공한 것만 objpoints/imgpoints에 추가
    img = imread_unicode(fname)  # 한글 경로 안전 로더로 이미지 읽기 → img가 None이면 디코딩 실패
    if img is None:  # 읽기에 실패하면 다음 이미지로 넘어감 → 캘리브레이션 데이터 부족 방지
        print(f"[skip] cannot read: {fname}")  # 어떤 파일이 실패했는지 출력 → 경로/파일 손상 확인
        continue  # 실패 파일은 제외 → 다음 이미지 처리

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 코너 검출은 보통 그레이에서 수행 → 코너 탐지가 안정적
    img_size = gray.shape[::-1]  # (W, H) 형태로 저장 → calibrateCamera의 imageSize로 사용

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE  # 코너 검출 보조 옵션 → 다양한 조명에서 성공률 증가
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)  # 체크보드 코너 검출 → 성공하면 corners에 좌표 저장

    if not ret:  # 코너를 못 찾았으면 해당 이미지는 제외 → 잘못된 대응점 방지
        print(f"[skip] corners not found: {fname}")  # 실패한 파일 출력 → CHECKERBOARD 설정/이미지 품질 점검
        continue  # 실패 이미지는 건너뜀 → 다음 이미지 처리

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # 코너를 서브픽셀로 정밀화 → 캘리브레이션 정확도 향상

    objpoints.append(objp)  # 이 이미지에 대응하는 3D 점 추가 → N개 코너의 실제 좌표
    imgpoints.append(corners2)  # 이 이미지에서 찾은 2D 점 추가 → N개 코너의 이미지 좌표

    # 코너 시각화: 코너가 잘 잡혔는지 눈으로 확인하기 위한 디버깅용 표시
    vis = img.copy()  # 원본을 복사해서 그리기용 이미지 생성 → 원본 손상 방지
    cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, ret)  # 코너 표시(점/선) → 검출 성공 여부를 시각 확인
    cv2.imshow("Corners", vis)  # 코너가 그려진 이미지를 창에 띄움 → 진행 중 상태 확인
    cv2.waitKey(80)  # 80ms 보여줌 → 여러 장이 빠르게 넘어가며 확인 가능

cv2.destroyAllWindows()  # 코너 확인 창 닫기 → 다음 결과(undistort) 출력 전에 창 정리

# (4) 캘리브레이션 (K, dist): 카메라 내부행렬 K와 왜곡계수 dist를 추정해서 왜곡보정/깊이추정에 사용
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)  # K/dist 계산 → 보정 파라미터 출력 가능

print("\n=== Calibration Result ===")  # 결과 구분 출력 → 콘솔에서 보기 쉽게
print("Reprojection error (RMS):", ret)  # RMS 재투영 오차 출력 → 값이 작을수록 캘리브레이션이 잘 됨
print("\nCamera Matrix K:")  # 내부행렬 헤더 출력 → fx, fy, cx, cy 포함
print(K)  # K 행렬 출력 → 이후 undistort/3D 계산에 사용 가능
print("\nDistortion Coefficients (k1, k2, p1, p2, k3 ...):")  # 왜곡계수 헤더 출력 → 방사/접선 왜곡 파라미터
print(dist.ravel())  # dist를 1차원으로 출력 → k1,k2,p1,p2,k3 형태 확인

# (5) 왜곡 보정 시각화 (undistort)  # 추정된 K/dist로 원본을 보정해 직선이 더 곧게 보이게 만들기 위함
sample_path = images[0]  # 첫 번째 이미지를 샘플로 선택 → undistort 결과를 대표로 확인
sample = imread_unicode(sample_path)   # cv2.imread 대신! → 한글 경로에서도 샘플을 정상 로드

undist = cv2.undistort(sample, K, dist)  # 왜곡 보정 수행 → 보정된 이미지(undist) 생성

combined = np.hstack((sample, undist))  # 원본과 보정을 좌우로 붙임 → 비교가 한눈에 가능

# 화면에 너무 크게 뜨는 걸 방지하기 위해 리사이즈로 보기 편하게 만듦
max_w = 1400  # 최대 가로 폭 제한 → 모니터에 맞게 표시
h, w = combined.shape[:2]  # 현재 결합 이미지의 높이/너비 → 축소 비율 계산에 사용
if w > max_w:  # 가로가 너무 크면 축소 실행 → 창이 화면 밖으로 나가는 걸 방지
    scale = max_w / w  # 축소 비율 계산 → 비율 유지
    combined = cv2.resize(combined, (int(w * scale), int(h * scale)))  # 비율대로 축소 → 화면에 잘 보이게 출력

cv2.imshow("Original | Undistorted", combined)  # 원본/보정 비교창 표시 → 보정 효과(직선/왜곡 감소) 확인
print(f"\n[INFO] Showing undistortion result for: {Path(sample_path).name}")  # 어떤 파일을 보여주는지 출력 → 디버깅/기록용
print("[INFO] Press any key to exit")  # 종료 안내 출력 
cv2.waitKey(0)  # 아무 키 입력까지 대기 
cv2.destroyAllWindows()  # 모든 창 닫기