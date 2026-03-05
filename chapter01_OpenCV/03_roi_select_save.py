import cv2 as cv
import numpy as np
import os
import sys

# (0) 전역 상태: 드래그 시작점/현재점/드래그 중 여부/ROI
start_pt = None
end_pt = None
dragging = False
roi = None

def on_mouse(event, x, y, flags, param):
    global start_pt, end_pt, dragging, roi
    img, show = param  # img: 원본, show: 사각형 표시용

    # (3) 마우스 드래그 시작점 기록
    if event == cv.EVENT_LBUTTONDOWN:
        dragging = True
        start_pt = (x, y)
        end_pt = (x, y)

    # (4) 드래그 중이면 사각형을 계속 업데이트해서 시각화 (힌트: cv.rectangle)
    elif event == cv.EVENT_MOUSEMOVE and dragging:
        end_pt = (x, y)

    # (5) 마우스 놓으면 ROI 확정 + 별도 창에 출력 (힌트: numpy 슬라이싱)
    elif event == cv.EVENT_LBUTTONUP and dragging:
        dragging = False
        end_pt = (x, y)

        x1, y1 = start_pt
        x2, y2 = end_pt
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        # 빈 선택 방지
        if x_max - x_min > 0 and y_max - y_min > 0:
            roi = img[y_min:y_max, x_min:x_max].copy()
            cv.imshow("ROI", roi)

def main():
    global start_pt, end_pt, dragging, roi

    # (1) 이미지 로드
    img_path = "soccer.jpg"
    if not os.path.exists(img_path):
        print("파일이 존재하지 않습니다.")
        sys.exit(1)

    img = cv.imread(img_path)
    if img is None:
        print("파일을 읽을 수 없습니다.")
        sys.exit(1)

    # (2) 표시용 이미지(사각형 그리기) 준비 + 콜백 등록
    show = img.copy()
    cv.namedWindow("Image")
    cv.setMouseCallback("Image", on_mouse, (img, show))

    # (6) 키 입력 루프: r=리셋, s=저장
    while True:
        show = img.copy()

        # 드래그 중이면 현재 사각형을 화면에 그림
        if dragging and start_pt and end_pt:
            cv.rectangle(show, start_pt, end_pt, (0, 255, 0), 2)

        cv.imshow("Image", show)
        k = cv.waitKey(1) & 0xFF

        # ESC 또는 q로 종료(편의)
        if k in (27, ord('q')):
            break

        # r: 영역 선택 리셋
        elif k == ord('r'):
            start_pt = end_pt = None
            dragging = False
            roi = None
            cv.destroyWindow("ROI")

        # s: 선택 ROI 저장 (힌트: cv.imwrite)
        elif k == ord('s'):
            if roi is not None:
                cv.imwrite("roi.png", roi)
                print("Saved: roi.png")
            else:
                print("ROI가 없습니다. 마우스로 영역을 먼저 선택하세요.")

    # (7) 종료 처리
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()