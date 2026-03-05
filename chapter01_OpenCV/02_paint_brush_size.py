import cv2 as cv                      # OpenCV 기능 사용(창/마우스/그리기/키입력)
import numpy as np                    # NumPy 사용(이번 코드에서는 직접 연산은 없지만 관례적으로 포함)
import sys                            # 오류 발생 시 프로그램 종료(sys.exit)용
import os                             # 파일 존재 여부 확인(os.path.exists)용

# (0) 전역 설정: 붓 크기 / 드래그 상태 / 현재 색상
brush = 5                             # 초기 붓 크기(요구사항: 5)
drawing = False                       # 마우스를 누른 채로 드래그 중인지 상태 저장
color = (255, 0, 0)                   # 현재 그릴 색상(BGR 기준: 파랑)

def on_mouse(event, x, y, flags, img):  # 마우스 이벤트가 발생할 때 호출되는 콜백 함수
    global drawing, color, brush      # 콜백 내부에서 전역 상태(드래그/색/붓)를 수정하기 위해 선언

    # (4) 마우스 이벤트 처리 (힌트: setMouseCallback + circle)
    if event == cv.EVENT_LBUTTONDOWN: # 왼쪽 버튼을 누르는 순간(그리기 시작)
        drawing = True                # 드래그 시작 상태로 전환
        color = (255, 0, 0)           # 좌클릭은 파랑(BGR)
        cv.circle(img, (x, y), brush, color, -1)  # 현재 위치에 원을 채워서 점을 찍음(-1은 채움)

    elif event == cv.EVENT_RBUTTONDOWN: # 오른쪽 버튼을 누르는 순간(그리기 시작)
        drawing = True                # 드래그 시작 상태로 전환
        color = (0, 0, 255)           # 우클릭은 빨강(BGR)
        cv.circle(img, (x, y), brush, color, -1)  # 현재 위치에 원을 채워서 점을 찍음

    elif event == cv.EVENT_MOUSEMOVE and drawing: # 마우스 이동 중이며 버튼이 눌린 상태면
        cv.circle(img, (x, y), brush, color, -1)  # 이동 경로에 계속 원을 찍어서 선처럼 보이게 그림

    elif event in (cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP): # 버튼을 떼는 순간(그리기 종료)
        drawing = False               # 드래그 종료 상태로 전환

def main():                           # 프로그램 실행 흐름을 담당하는 메인 함수
    global brush                      # 키 입력으로 붓 크기를 변경하기 위해 전역 brush 사용

    # (1) 이미지 로드
    img_path = "soccer.jpg"           # 배경으로 사용할 이미지 파일 경로
    if not os.path.exists(img_path):  # 파일이 존재하는지 먼저 확인
        print("파일이 존재하지 않습니다.") # 파일이 없으면 안내 메시지 출력
        sys.exit(1)                   # 에러 코드로 종료

    img = cv.imread(img_path)         # 이미지 로드(OpenCV 기본 BGR)
    if img is None:                   # 로드 실패 여부 확인
        print("파일을 읽을 수 없습니다.") # 로드 실패 메시지 출력
        sys.exit(1)                   # 에러 코드로 종료

    # (2) 창 생성
    cv.namedWindow("Paint")           # "Paint"라는 이름의 창 생성

    # (3) 마우스 콜백 등록 (힌트: cv.setMouseCallback)
    cv.setMouseCallback("Paint", on_mouse, img)  # "Paint" 창에 콜백 등록 + img를 콜백으로 전달

    # (5) 키 입력 처리 루프 (힌트: cv.waitKey(1)로 +, -, q 구분)
    while True:                       # 키 입력을 계속 받기 위한 반복 루프
        cv.imshow("Paint", img)       # 현재 img(그림이 그려진 상태)를 창에 표시
        k = cv.waitKey(1) & 0xFF      # 1ms 대기하며 키 입력을 받고 하위 8비트만 사용

        # (6) q 키 누르면 종료
        if k == ord('q'):             # q 입력 시 종료 조건
            break                     # 반복 루프 탈출

        # (7) + 입력 시 붓 크기 1 증가 (최대 15)
        elif k in (ord('+'), ord('=')): # '+' 입력(키보드에 따라 '='로 들어오는 경우 포함)
            brush = min(15, brush + 1) # 붓 크기를 1 증가시키되 최대 15로 제한
            print("brush:", brush)    # 현재 붓 크기 출력(확인용)

        # (8) - 입력 시 붓 크기 1 감소 (최소 1)
        elif k in (ord('-'), ord('_')): # '-' 입력(키보드에 따라 '_'로 들어오는 경우 포함)
            brush = max(1, brush - 1) # 붓 크기를 1 감소시키되 최소 1로 제한
            print("brush:", brush)    # 현재 붓 크기 출력(확인용)

    # (9) 종료 처리
    cv.destroyAllWindows()            # 모든 OpenCV 창 닫기(종료 처리)

if __name__ == "__main__":            # 이 파일을 직접 실행할 때만 main() 수행
    main()                            # 메인 함수 호출