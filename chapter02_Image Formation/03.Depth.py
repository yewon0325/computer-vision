import cv2  # OpenCV 함수 사용 → 이미지 읽기/변환/시각화 가능
import numpy as np  # 수치 연산용 배열 사용 → disparity/depth 계산 가능
from pathlib import Path  # 경로를 안전하게 처리 → 폴더/파일 찾기 편리

# -----------------------------
# (0) 출력 폴더 생성
# -----------------------------
output_dir = Path("./outputs")  # 결과 저장 폴더 경로 지정 → outputs 폴더에 결과를 모음
output_dir.mkdir(parents=True, exist_ok=True)  # 출력 폴더 생성 → 저장 시 폴더 없어서 실패하는 것 방지

# -----------------------------
# (1) 한글/특수문자 경로에서도 안전하게 이미지 읽기
# -----------------------------
def imread_unicode(path, flags=cv2.IMREAD_COLOR):  # 한글/특수문자 경로에서도 안전하게 이미지 읽기 위한 함수
    data = np.fromfile(str(path), dtype=np.uint8)  # 파일을 바이트로 읽음 → 경로 인코딩 문제를 줄임
    return cv2.imdecode(data, flags)  # 바이트를 이미지로 디코딩 → 정상적인 이미지 배열 반환

# -----------------------------
# (2) 좌/우 이미지 불러오기
# -----------------------------
left_color = cv2.imread("chapter02_Image Formation/L02 실습/images/left.png")  # 왼쪽 이미지 읽기 → stereo 입력용 원본 확보
right_color = cv2.imread("chapter02_Image Formation/L02 실습/images/right.png")  # 오른쪽 이미지 읽기 → stereo 입력용 원본 확보

# -----------------------------
# (3) 기본 경로 실패 시 대체 경로에서 다시 읽기
# -----------------------------
if left_color is None or right_color is None:  # 기본 경로 읽기가 실패했는지 확인 → 한글 경로 문제 대비
    base = Path(__file__).resolve().parent  # 현재 파이썬 파일 기준 폴더 위치 구하기 → 상대경로 보정
    left_path = base / "L02 실습" / "images" / "left.png"  # 대체 왼쪽 이미지 경로 생성 → 다른 폴더 구조 대응
    right_path = base / "L02 실습" / "images" / "right.png"  # 대체 오른쪽 이미지 경로 생성 → 다른 폴더 구조 대응
    if left_path.exists() and right_path.exists():  # 대체 경로에 파일이 실제로 있는지 확인 → 안전하게 로드
        left_color = imread_unicode(left_path)  # 한글 경로 안전 로더로 왼쪽 이미지 읽기 → 이미지 확보
        right_color = imread_unicode(right_path)  # 한글 경로 안전 로더로 오른쪽 이미지 읽기 → 이미지 확보

# -----------------------------
# (4) 이미지 로드 실패 검사
# -----------------------------
if left_color is None or right_color is None:  # 끝까지 이미지 로드에 실패했는지 확인 → 이후 계산 불가 판단
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다. (left.png / right.png 경로 확인)")  # 실행 중단 → 경로 오류를 바로 알림

# -----------------------------
# (5) 카메라 파라미터 설정
# -----------------------------
f = 700.0  # 초점거리 설정 → depth 계산식 Z = fB / d 에 사용
B = 0.12  # 베이스라인 설정 → 두 카메라 사이 거리로 depth 계산에 사용

# -----------------------------
# (6) ROI 설정
# -----------------------------
rois = {  # 관심영역(ROI) 정의 → 객체별 평균 disparity/depth를 구하기 위함
    "Painting": (55, 50, 130, 110),  # 그림 영역 지정 → 이 구역의 평균 거리 계산
    "Frog": (90, 265, 230, 95),  # 개구리 영역 지정 → 이 구역의 평균 거리 계산
    "Teddy": (310, 35, 115, 90)  # 곰인형 영역 지정 → 이 구역의 평균 거리 계산
}

# -----------------------------
# (7) 그레이스케일 변환
# -----------------------------
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)  # 왼쪽 영상을 그레이로 변환 → StereoBM 입력 형식 맞춤
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)  # 오른쪽 영상을 그레이로 변환 → StereoBM 입력 형식 맞춤

# -----------------------------
# (8) Disparity 계산 (StereoBM)
# -----------------------------
stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)  # StereoBM 객체 생성 → disparity 계산 준비
disp_raw = stereo.compute(left_gray, right_gray)  # 원시 disparity 계산 → 시차 맵이 int16 형태로 생성됨
disparity = disp_raw.astype(np.float32) / 16.0  # 실수형으로 변환 후 스케일 복원 → 실제 disparity 값 사용 가능

# -----------------------------
# (9) Depth 계산
# Z = fB / d (disparity > 0만 사용)
# -----------------------------
valid_mask = disparity > 0  # 유효한 disparity 위치만 선택 → 0 이하 값은 거리 계산에서 제외
depth_map = np.zeros_like(disparity, dtype=np.float32)  # depth 결과 배열 생성 → 각 픽셀 거리 저장 준비
depth_map[valid_mask] = (f * B) / disparity[valid_mask]  # 유효 픽셀의 깊이 계산 → 거리 맵 생성

# -----------------------------
# (10) ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}  # ROI별 계산 결과를 저장할 딕셔너리 생성 → 마지막 출력에 사용
for name, (x, y, w, h) in rois.items():  # 각 ROI를 하나씩 처리 → 객체별 disparity/depth 평균 계산
    roi_disp = disparity[y:y+h, x:x+w]  # 해당 ROI의 disparity 부분만 잘라냄 → 객체 영역 시차 확보
    roi_depth = depth_map[y:y+h, x:x+w]  # 해당 ROI의 depth 부분만 잘라냄 → 객체 영역 거리 확보

    roi_valid = roi_disp > 0  # ROI 내부에서도 유효한 disparity만 선택 → 잘못된 평균 방지
    if np.any(roi_valid):  # 유효한 픽셀이 하나라도 있는지 확인 → 평균 계산 가능 여부 판단
        mean_disp = float(np.mean(roi_disp[roi_valid]))  # ROI 평균 disparity 계산 → 시차 크기 대표값 출력
        mean_depth = float(np.mean(roi_depth[roi_valid]))  # ROI 평균 depth 계산 → 거리 대표값 출력
    else:  # 유효한 disparity가 하나도 없으면 → 평균 계산 불가
        mean_disp = float("nan")  # disparity를 NaN으로 저장 → 계산 실패 표시
        mean_depth = float("nan")  # depth를 NaN으로 저장 → 계산 실패 표시

    results[name] = {  # ROI 이름별 결과 저장 → 나중에 한 번에 출력 가능
        "mean_disparity": mean_disp,  # 평균 disparity 저장 → 객체별 시차 비교 가능
        "mean_depth": mean_depth  # 평균 depth 저장 → 객체별 거리 비교 가능
    }

# -----------------------------
# (11) ROI 평균 disparity / depth 출력
# -----------------------------
print("\n=== ROI 평균 Disparity / Depth ===")  # 결과 제목 출력 → ROI별 평균값 보기 좋게 표시
for name, r in results.items():  # 저장한 ROI 결과를 순서대로 꺼냄 → 콘솔에 표시
    print(f"{name:8s} | mean disparity = {r['mean_disparity']:.4f} | mean depth = {r['mean_depth']:.4f}")  # 각 ROI 평균값 출력

# -----------------------------
# (12) 가장 가까운 / 가장 먼 ROI 판단
# -----------------------------
valid_depth_items = [(k, v["mean_depth"]) for k, v in results.items() if not np.isnan(v["mean_depth"])]  # NaN이 아닌 ROI만 추려냄 → 비교 준비

if valid_depth_items:  # 유효한 depth 결과가 하나라도 있는지 확인 → 결론 출력 가능 여부 판단
    closest = min(valid_depth_items, key=lambda t: t[1])[0]  # depth가 가장 작은 ROI 찾기 → 가장 가까운 객체 결정
    farthest = max(valid_depth_items, key=lambda t: t[1])[0]  # depth가 가장 큰 ROI 찾기 → 가장 먼 객체 결정
    print(f"\n[결론] 가장 가까운 ROI: {closest}")  # 가장 가까운 ROI 출력 → 거리 비교 결과 확인
    print(f"[결론] 가장 먼 ROI: {farthest}")  # 가장 먼 ROI 출력 → 거리 비교 결과 확인
else:  # 유효한 depth가 하나도 없으면 → 거리 비교 불가
    print("\n[WARN] 유효한 depth가 ROI에서 하나도 계산되지 않았습니다. StereoBM 파라미터/ROI를 확인하세요.")  # 경고 메시지 출력

# -----------------------------
# (13) disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
disp_tmp = disparity.copy()  # disparity 사본 생성 → 시각화용으로 안전하게 가공
disp_tmp[disp_tmp <= 0] = np.nan  # 유효하지 않은 disparity를 NaN 처리 → 시각화 범위 계산에서 제외

if np.all(np.isnan(disp_tmp)):  # 모든 disparity가 무효한지 확인 → 색상 맵 생성 가능 여부 판단
    raise ValueError("유효한 disparity 값이 없습니다.")  # 실행 중단 → disparity 계산 실패를 알림

d_min = np.nanpercentile(disp_tmp, 5)  # disparity의 하위 5% 값 계산 → 극단값 영향을 줄여 정규화
d_max = np.nanpercentile(disp_tmp, 95)  # disparity의 상위 95% 값 계산 → 극단값 영향을 줄여 정규화

if d_max <= d_min:  # 정규화 범위가 비정상인지 확인 → 0으로 나누기 방지
    d_max = d_min + 1e-6  # 아주 작은 값 추가 → 정규화 가능하게 만듦

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)  # disparity를 0~1 범위로 정규화 → 컬러맵 적용 준비
disp_scaled = np.clip(disp_scaled, 0, 1)  # 범위를 0~1로 제한 → 이상치 제거

disp_vis = np.zeros_like(disparity, dtype=np.uint8)  # disparity 시각화용 8비트 배열 생성 → 컬러맵 입력 준비
valid_disp = ~np.isnan(disp_tmp)  # 유효한 disparity 위치 계산 → 값이 있는 곳만 표시
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)  # 0~255 밝기값으로 변환 → 컬러맵 적용 가능

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)  # disparity에 JET 컬러맵 적용 → 가까운/먼 영역이 색으로 보임

# -----------------------------
# (14) depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)  # depth 시각화용 8비트 배열 생성 → 컬러맵 적용 준비

if np.any(valid_mask):  # 유효한 depth가 하나라도 있는지 확인 → depth 시각화 가능 여부 판단
    depth_valid = depth_map[valid_mask]  # 유효 depth 값만 추출 → 정규화 범위 계산에 사용

    z_min = np.percentile(depth_valid, 5)  # depth의 하위 5% 값 계산 → 극단값 영향 감소
    z_max = np.percentile(depth_valid, 95)  # depth의 상위 95% 값 계산 → 극단값 영향 감소

    if z_max <= z_min:  # 정규화 범위가 비정상인지 확인 → 나눗셈 오류 방지
        z_max = z_min + 1e-6  # 아주 작은 값 추가 → 정규화 가능하게 만듦

    depth_scaled = (depth_map - z_min) / (z_max - z_min)  # depth를 0~1 범위로 정규화 → 색상 맵 준비
    depth_scaled = np.clip(depth_scaled, 0, 1)  # 정규화 범위를 0~1로 제한 → 이상값 제거

    depth_scaled = 1.0 - depth_scaled  # 가까울수록 큰 값이 되게 반전 → 빨강이 가까움으로 보이게 함
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)  # 유효 depth만 0~255로 변환 → 컬러맵 적용 가능

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)  # depth에 JET 컬러맵 적용 → 거리 분포가 색으로 보임

# -----------------------------
# (15) Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy()  # 왼쪽 이미지 복사 → ROI 사각형을 그려도 원본 보존
right_vis = right_color.copy()  # 오른쪽 이미지 복사 → ROI 사각형을 그려도 원본 보존

for name, (x, y, w, h) in rois.items():  # 모든 ROI를 순회 → 각 객체 위치를 그림
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 왼쪽 이미지에 ROI 사각형 그림 → 객체 위치 확인 가능
    cv2.putText(left_vis, name, (x, y - 8),  # 왼쪽 이미지에 ROI 이름 표시 → 어떤 객체인지 구분
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # 글꼴/크기/색 지정 → 초록 글씨로 라벨 표시

    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 오른쪽 이미지에 ROI 사각형 그림 → 대응 위치 확인 가능
    cv2.putText(right_vis, name, (x, y - 8),  # 오른쪽 이미지에 ROI 이름 표시 → 어떤 객체인지 구분
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # 글꼴/크기/색 지정 → 초록 글씨로 라벨 표시

# -----------------------------
# (16) 결과 저장
# -----------------------------
cv2.imwrite(str(output_dir / "left_roi.png"), left_vis)  # ROI 표시된 왼쪽 이미지 저장
cv2.imwrite(str(output_dir / "right_roi.png"), right_vis)  # ROI 표시된 오른쪽 이미지 저장 
cv2.imwrite(str(output_dir / "disparity_color.png"), disparity_color)  # 컬러 disparity 저장 → 시차 분포 확인 가능
cv2.imwrite(str(output_dir / "depth_color.png"), depth_color)  # 컬러 depth 저장 → 거리 분포 확인 가능

# -----------------------------
# (17) 숫자 데이터 저장
# -----------------------------
np.save(str(output_dir / "disparity.npy"), disparity)  # disparity 수치 배열 저장 
np.save(str(output_dir / "depth.npy"), depth_map)  # depth 수치 배열 저장 
# -----------------------------
# (18) 결과 출력
# -----------------------------
cv2.imshow("Left (ROI)", left_vis)  # ROI가 표시된 왼쪽 이미지 출력 → 객체 구역 확인 가능
cv2.imshow("Right (ROI)", right_vis)  # ROI가 표시된 오른쪽 이미지 출력 → 대응 구역 확인 가능
cv2.imshow("Disparity (color)", disparity_color)  # 컬러 disparity 맵 출력 → 가까운/먼 영역을 색으로 확인
cv2.imshow("Depth (color)", depth_color)  # 컬러 depth 맵 출력 → 거리 차이를 색으로 확인
cv2.waitKey(0)  # 키 입력까지 창 유지 → 결과를 충분히 확인 가능
cv2.destroyAllWindows()  # 모든 창 닫기 → 프로그램 정상 종료
print(f"\n[INFO] Saved outputs to: {output_dir.resolve()}")  # 저장 폴더 경로 출력 → 결과 파일 위치 확인 가능