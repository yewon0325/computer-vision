import cv2 as cv
import numpy as np
import sys
import os

def main():
    img_path = "soccer.jpg"
    if not os.path.exists(img_path):
        print("파일이 존재하지 않습니다.")
        sys.exit(1)
    # 1) 이미지 로드
    img = cv.imread(img_path)
    if img is None:
        print("파일을 읽을 수 없습니다.")
        sys.exit(1)

    # 2) 그레이스케일 변환 (BGR -> GRAY)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 3) np.hstack을 위해 GRAY(1채널) -> BGR(3채널)로 맞추기
    gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    # 4) 가로로 연결
    combined = np.hstack((img, gray_bgr))

    # 5)캡처처럼 fx, fy로 축소 (예: 0.5 = 반으로)
    combined_small = cv.resize(combined, dsize=(0, 0), fx=0.5, fy=0.5)

    # 6) 출력 + 아무 키나 누르면 닫기
    cv.imshow("Original | Grayscale (small)", combined_small)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()