import cv2 as cv
import numpy as np
import sys
import os

# (0) 전역 설정: 붓 크기 / 드래그 상태 / 현재 색상
brush = 5                 # 요구사항: 초기 붓 크기 5
drawing = False           # 드래그 중인지 여부
color = (255, 0, 0)       

def on_mouse(event, x, y, flags, img):
    global drawing, color, brush

    # (4) 마우스 이벤트 처리 (힌트: setMouseCallback + circle)
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        color = (255, 0, 0)                # 좌클릭=파란색
        cv.circle(img, (x, y), brush, color, -1)

    elif event == cv.EVENT_RBUTTONDOWN:
        drawing = True
        color = (0, 0, 255)                # 우클릭=빨간색
        cv.circle(img, (x, y), brush, color, -1)

    elif event == cv.EVENT_MOUSEMOVE and drawing:
        # 드래그로 연속 그리기
        cv.circle(img, (x, y), brush, color, -1)

    elif event in (cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP):
        drawing = False

def main():
    global brush

    # (1) 이미지 로드
    img_path = "soccer.jpg"
    if not os.path.exists(img_path):
        print("파일이 존재하지 않습니다.")
        sys.exit(1)

    img = cv.imread(img_path)
    if img is None:
        print("파일을 읽을 수 없습니다.")
        sys.exit(1)

    # (2) 창 생성
    cv.namedWindow("Paint")

    # (3) 마우스 콜백 등록 (힌트: cv.setMouseCallback)
    cv.setMouseCallback("Paint", on_mouse, img)

    # (5) 키 입력 처리 루프 (힌트: cv.waitKey(1)로 +, -, q 구분)
    while True:
        cv.imshow("Paint", img)
        k = cv.waitKey(1) & 0xFF

        # (6) q 키 누르면 종료
        if k == ord('q'):
            break

        # (7) + 입력 시 붓 크기 1 증가 (최대 15)
        elif k in (ord('+'), ord('=')):    # 키보드에 따라 '+'가 '='로 들어오는 경우 대비
            brush = min(15, brush + 1)
            print("brush:", brush)

        # (8) - 입력 시 붓 크기 1 감소 (최소 1)
        elif k in (ord('-'), ord('_')):
            brush = max(1, brush - 1)
            print("brush:", brush)

    # (9) 종료 처리
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()