import cv2  # OpenCV 변환/출력 함수 사용 → 회전/이동 결과를 창으로 확인
import numpy as np  # 배열 처리(hstack/resize) 사용 → 원본과 결과를 나란히 표시
from pathlib import Path  # 경로를 안전하게 처리 → 이미지 파일을 정확히 찾음

# (0) 한글/특수문자 경로에서도 안전하게 이미지 읽기
def imread_unicode(path, flags=cv2.IMREAD_COLOR):  # 한글 경로에서도 이미지 로드 → imread 실패 방지
    data = np.fromfile(str(path), dtype=np.uint8)  # 파일을 바이트로 읽기 → 경로 인코딩 문제 회피
    return cv2.imdecode(data, flags)  # 바이트를 이미지로 디코딩 → img(BGR)가 반환됨

def main():  # 프로그램 실행 시작점 → 변환된 결과 창을 띄움
    # (1) 이미지 경로 설정 (스크립트 위치 기준)
    base = Path(__file__).resolve().parent  # 현재 .py 폴더 기준 잡기 → 상대경로 오류 방지
    img_path = base / "L02 실습" / "images" / "rose.png"  # rose.png 경로 구성 → 입력 이미지 지정

    # 폴더 구조가 다르면(상위 폴더에서 실행 등) 이 경로도 시도
    if not img_path.exists():  # 첫 경로에 파일이 없으면 → 다른 폴더 구조를 대비
        img_path = base / "chapter02_Image Formation" / "L02 실습" / "images" / "rose.png"  # 대체 경로로 재설정 → 파일 찾기 성공률 증가

    # (2) 이미지 로드
    img = imread_unicode(img_path)  # 이미지 읽기 수행 → 이후 회전/이동할 원본 확보
    if img is None:  # 이미지 로드 실패 시 → 이후 처리 불가
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {img_path}")  # 즉시 중단하고 원인 안내

    h, w = img.shape[:2]  # 이미지 높이/너비 가져오기 → 변환 크기와 중심 계산에 사용
    center = (w / 2, h / 2)  # 이미지 중심 좌표 계산 → 중심 기준 회전하기 위함

    # (3) 회전+스케일 행렬 생성 (힌트: getRotationMatrix2D)
    angle = 30      # +30도  # 이미지를 30도 회전 → 기울어진 결과가 생성됨
    scale = 0.8     # 0.8배  # 이미지를 0.8배 축소 → 회전과 함께 작아진 결과가 생성됨
    M = cv2.getRotationMatrix2D(center, angle, scale)  # 2x3 변환행렬 생성 → warpAffine에 넣을 준비

    # (4) 평행이동 반영 (힌트: 회전행렬 마지막 열 값 조정)
    tx, ty = 80, -40  # 이동량 설정 → 오른쪽(+80), 위쪽(-40)로 이동할 예정
    M[0, 2] += tx  # x방향 이동을 행렬에 반영 → 결과가 오른쪽으로 이동
    M[1, 2] += ty  # y방향 이동을 행렬에 반영 → 결과가 위쪽으로 이동

    # (5) 변환 적용 (힌트: warpAffine)
    out = cv2.warpAffine(img, M, (w, h))  # 회전+스케일+이동 적용 → 변환된 이미지(out) 생성

    # (6) 결과 표시 (원본 | 변환)
    combined = np.hstack((img, out))  # 원본과 결과를 가로로 붙임 → 한 창에서 비교 가능

    # 너무 크면 보기 좋게 축소
    max_w = 1400  # 출력 최대 가로폭 제한 → 화면 밖으로 나가는 것을 방지
    ch, cw = combined.shape[:2]  # 결합 이미지의 높이/너비 확인 → 축소 여부 판단
    if cw > max_w:  # 가로가 너무 크면 → 보기 편하게 축소
        s = max_w / cw  # 축소 비율 계산 → 비율 유지하며 축소
        combined = cv2.resize(combined, (int(cw * s), int(ch * s)))  # 축소 적용 → 화면에 맞게 출력

    cv2.imshow("Original | Rotated+Scaled+Translated", combined)  # 결과 창 출력 → 왼쪽 원본, 오른쪽 변환 결과가 보임
    cv2.waitKey(0)  # 키 입력 대기 → 창이 바로 꺼지지 않고 결과를 확인 가능
    cv2.destroyAllWindows()  # 모든 창 닫기 → 프로그램 정상 종료

if __name__ == "__main__":  # 이 파일을 직접 실행할 때만 → main()을 실행
    main()  # 변환 실행 → 회전/축소/이동된 결과가 화면에 표시됨