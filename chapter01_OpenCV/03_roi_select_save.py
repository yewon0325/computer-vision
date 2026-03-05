import cv2 as cv                          # OpenCV 사용(창/마우스 이벤트/도형 그리기/저장)
import numpy as np                        # NumPy 사용(ROI 슬라이싱은 배열 연산 기반)
import os                                 # 파일 존재 여부 확인(os.path.exists)용
import sys                                # 오류 시 종료(sys.exit)용

# (0) 전역 상태: 드래그 시작점/현재점/드래그 중 여부/ROI
start_pt = None                           # 드래그 시작 좌표(처음 클릭한 위치)
end_pt = None                             # 드래그 현재/종료 좌표(마우스가 가리키는 위치)
dragging = False                          # 드래그 진행 중인지 여부
roi = None                                # 선택된 ROI 이미지(잘라낸 결과)

def on_mouse(event, x, y, flags, param):  # 마우스 이벤트 콜백(이벤트, 좌표, 플래그, 사용자 파라미터)
    global start_pt, end_pt, dragging, roi # 콜백에서 전역 상태를 갱신하기 위해 global 선언
    img, show = param                     # img: 원본 이미지, show: 표시용(여기서는 콜백에서 직접 수정 X)

    # (3) 마우스 드래그 시작점 기록
    if event == cv.EVENT_LBUTTONDOWN:     # 왼쪽 버튼을 누르는 순간(드래그 시작)
        dragging = True                   # 드래그 상태를 시작으로 설정
        start_pt = (x, y)                 # 시작 좌표를 현재 마우스 좌표로 저장
        end_pt = (x, y)                   # 초기 종료 좌표도 시작점으로 맞춰둠(사각형 초기화)

    # (4) 드래그 중이면 사각형을 계속 업데이트해서 시각화 (힌트: cv.rectangle)
    elif event == cv.EVENT_MOUSEMOVE and dragging: # 드래그 중 마우스가 움직일 때
        end_pt = (x, y)                   # 현재 좌표를 종료 좌표로 계속 갱신(루프에서 사각형 그림)

    # (5) 마우스 놓으면 ROI 확정 + 별도 창에 출력 (힌트: numpy 슬라이싱)
    elif event == cv.EVENT_LBUTTONUP and dragging: # 왼쪽 버튼을 떼는 순간(드래그 종료)
        dragging = False                  # 드래그 상태 종료
        end_pt = (x, y)                   # 마지막 좌표를 종료점으로 확정

        x1, y1 = start_pt                 # 시작점 좌표를 분해(x1, y1)
        x2, y2 = end_pt                   # 종료점 좌표를 분해(x2, y2)
        x_min, x_max = sorted([x1, x2])   # 드래그 방향과 무관하게 좌/우 경계를 정렬
        y_min, y_max = sorted([y1, y2])   # 드래그 방향과 무관하게 상/하 경계를 정렬

        # 빈 선택 방지
        if x_max - x_min > 0 and y_max - y_min > 0:     # 너비/높이가 0이면 ROI가 비므로 제외
            roi = img[y_min:y_max, x_min:x_max].copy()  # 원본에서 ROI를 슬라이싱하고 별도 메모리로 복사
            cv.imshow("ROI", roi)                       # 선택된 ROI를 별도 창으로 즉시 표시

def main():                             # 프로그램 실행 흐름을 담당하는 메인 함수
    global start_pt, end_pt, dragging, roi # 키 입력(r/s)에서 전역 상태를 변경하므로 global 선언

    # (1) 이미지 로드
    img_path = "soccer.jpg"             # 입력 이미지 파일 경로
    if not os.path.exists(img_path):    # 파일 존재 여부 확인
        print("파일이 존재하지 않습니다.") # 파일이 없으면 안내 메시지 출력
        sys.exit(1)                     # 에러 코드로 종료

    img = cv.imread(img_path)           # 이미지 로드(OpenCV 기본 BGR)
    if img is None:                     # 로드 실패 여부 확인
        print("파일을 읽을 수 없습니다.") # 로드 실패 메시지 출력
        sys.exit(1)                     # 에러 코드로 종료

    # (2) 표시용 이미지(사각형 그리기) 준비 + 콜백 등록
    show = img.copy()                   # 화면 표시용 이미지(원본 위에 사각형을 그릴 때 사용)
    cv.namedWindow("Image")             # "Image" 이름의 창 생성
    cv.setMouseCallback("Image", on_mouse, (img, show)) # 마우스 콜백 등록 + 원본/표시용을 파라미터로 전달

    # (6) 키 입력 루프: r=리셋, s=저장
    while True:                         # 매 프레임 화면 갱신 + 키 입력을 받는 루프
        show = img.copy()               # 매 루프마다 원본으로부터 표시용을 새로 만들어 잔상 누적 방지

        if dragging and start_pt and end_pt:            # 드래그 중이고 좌표가 준비되어 있으면
            cv.rectangle(show, start_pt, end_pt, (0, 255, 0), 2) # 표시용 이미지에 현재 사각형을 그림(BGR, 두께 2)

        cv.imshow("Image", show)        # 현재 표시용 이미지를 창에 출력
        k = cv.waitKey(1) & 0xFF        # 1ms 대기하며 키 입력을 받고 하위 8비트만 사용

        if k in (27, ord('q')):         # ESC(27) 또는 q 입력 시 종료
            break                       # 루프 탈출

        elif k == ord('r'):             # r 입력 시 선택 상태 리셋
            start_pt = end_pt = None    # 시작/끝 좌표 초기화
            dragging = False            # 드래그 상태 강제 종료
            roi = None                  # ROI 결과 초기화
            cv.destroyWindow("ROI")     # ROI 창이 열려 있으면 닫기(없으면 무시될 수 있음)

        elif k == ord('s'):             # s 입력 시 ROI를 파일로 저장
            if roi is not None:         # ROI가 실제로 선택된 상태인지 확인
                cv.imwrite("roi.png", roi) # ROI 이미지를 roi.png로 저장
                print("Saved: roi.png") # 저장 완료 메시지 출력
            else:                       # ROI가 아직 없으면
                print("ROI가 없습니다. 마우스로 영역을 먼저 선택하세요.") # 먼저 선택하라고 안내

    # (7) 종료 처리
    cv.destroyAllWindows()              # 열린 모든 OpenCV 창 닫기

if __name__ == "__main__":              # 이 파일을 직접 실행할 때만 main() 수행
    main()                               # 메인 함수 호출