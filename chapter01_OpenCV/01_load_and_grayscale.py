import cv2 as cv                    # OpenCV 사용
import numpy as np               # NumPy 사용(배열 결합용)
import sys                            # 종료/에러 처리용(sys.exit)
import os                             # 파일 존재 여부 확인용

def main():                           # 메인 로직을 담는 함수 정의
    img_path = "soccer.jpg"      # 읽어올 이미지 파일 경로(현재 작업 디렉토리 기준)

    if not os.path.exists(img_path):   # img_path 위치에 파일이 실제로 존재하는지 확인
        print("파일이 존재하지 않습니다.") # 파일이 없으면 사용자에게 메시지 출력
        sys.exit(1)                    # 비정상 종료 코드(1)로 프로그램 종료

    # 1) 이미지 로드
    img = cv.imread(img_path)    # 이미지 파일을 읽어 BGR 형식의 NumPy 배열로 로드(OpenCV 기본은 BGR)

    if img is None:              # 파일을 못 읽었거나(손상/권한/경로 문제 등) 로드 실패 시 None 반환
        print("파일을 읽을 수 없습니다.") # 실패 메시지 출력
        sys.exit(1)              # 비정상 종료 코드(1)로 프로그램 종료

    # 2) 그레이스케일 변환 (BGR -> GRAY)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 컬러(BGR) 이미지를 1채널 그레이스케일로 변환

    # 3) np.hstack을 위해 GRAY(1채널) -> BGR(3채널)로 맞추기
    gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)  # 그레이(1채널)를 BGR(3채널)로 변환해 채널 수 맞춤

    # 4) 가로로 연결
    combined = np.hstack((img, gray_bgr))  # 원본(img)과 변환 이미지(gray_bgr)를 가로 방향으로 붙여 하나로 만듦

    # 5)캡처처럼 fx, fy로 축소 (예: 0.5 = 반으로)
    combined_small = cv.resize(            # 결과 이미지를 화면 표시용으로 축소/확대
        combined,                           # 리사이즈할 대상 이미지
        dsize=(0, 0),                       # dsize를 (0,0)으로 두면 fx, fy 배율을 사용
        fx=0.5,                             # 가로 크기 배율(0.5면 절반)
        fy=0.5                              # 세로 크기 배율(0.5면 절반)
    )

    # 6) 출력 + 아무 키나 누르면 닫기
    cv.imshow("Original | Grayscale (small)", combined_small)  # 창 제목과 함께 이미지 표시
    cv.waitKey(0)                                              # 키 입력을 무한 대기(아무 키나 누르면 다음 줄로)
    cv.destroyAllWindows()                                      # 열려있는 모든 OpenCV 창 닫기

if __name__ == "__main__":      # 이 파일이 '직접 실행'될 때만 아래 코드를 실행(모듈로 import될 땐 실행 X)
    main()                       # main 함수 호출