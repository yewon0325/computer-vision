import cv2                                          # 웹캠 영상 캡처 및 이미지 처리를 위해 임포트 → OpenCV 기능 사용 가능
import mediapipe as mp                              # 얼굴 랜드마크 감지 모델을 사용하기 위해 임포트 → MediaPipe 기능 사용 가능
from mediapipe.tasks import python                  # MediaPipe Tasks의 python 인터페이스 임포트 → 새 API 방식의 기반 모듈 로드
from mediapipe.tasks.python import vision           # MediaPipe vision 태스크 임포트 → FaceLandmarker 등 비전 모델 사용 가능
from mediapipe.tasks.python.vision import FaceLandmarkerOptions  # FaceLandmarker 옵션 클래스 임포트 → 검출기 설정값 구성에 사용
from mediapipe.tasks.python.vision import FaceLandmarker         # FaceLandmarker 클래스 임포트 → 얼굴 랜드마크 검출기 생성에 사용
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode  # 실행 모드 클래스 임포트 → 영상/이미지 처리 모드 설정에 사용
import urllib.request                               # URL에서 파일 다운로드를 위해 임포트 → 모델 파일 자동 다운로드에 사용
import os                                           # 파일 존재 여부 확인을 위해 임포트 → 모델 파일 중복 다운로드 방지에 사용

MODEL_PATH = "face_landmarker.task"                 # 다운로드할 모델 파일 경로(이름) 지정 → 이 경로에 모델 저장 및 로드

if not os.path.exists(MODEL_PATH):                  # 모델 파일이 이미 존재하는지 확인 → 없을 때만 다운로드 실행
    print("모델 파일 다운로드 중...")               # 다운로드 시작 안내 메시지 출력 → 사용자에게 진행 상황 알림
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"  # 공식 MediaPipe 모델 다운로드 URL 지정 → Google 서버에서 모델 파일 가져옴
    urllib.request.urlretrieve(url, MODEL_PATH)     # URL에서 파일을 MODEL_PATH에 저장 → 모델 파일 다운로드 완료
    print("다운로드 완료!")                         # 다운로드 완료 안내 메시지 출력 → 사용자에게 완료 알림

options = FaceLandmarkerOptions(                    # FaceLandmarker 옵션 객체 생성 → 검출기 동작 방식 설정
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),  # 모델 파일 경로를 기본 옵션으로 설정 → 지정한 모델 파일 로드
    running_mode=VisionTaskRunningMode.VIDEO,       # 실행 모드를 VIDEO(연속 프레임)로 설정 → 웹캠 영상 처리에 적합한 모드
    num_faces=1,                                    # 최대 감지할 얼굴 수를 1로 제한 → 1명의 얼굴만 처리
    min_face_detection_confidence=0.5,              # 얼굴 감지 최소 신뢰도 0.5 설정 → 50% 이상 확신할 때만 감지
    min_face_presence_confidence=0.5,               # 얼굴 존재 최소 신뢰도 0.5 설정 → 얼굴이 있다고 판단하는 기준값
    min_tracking_confidence=0.5                     # 랜드마크 추적 최소 신뢰도 0.5 설정 → 50% 이상 확신할 때만 추적
)

face_landmarker = FaceLandmarker.create_from_options(options)  # 설정된 옵션으로 FaceLandmarker 검출기 생성 → 랜드마크 검출기 초기화 완료

cap = cv2.VideoCapture(0)           # 기본 웹캠(인덱스 0) 열기 → 웹캠으로부터 실시간 영상 캡처 시작

frame_idx = 0                       # 프레임 번호 카운터 초기화 → VIDEO 모드에서 타임스탬프 계산에 사용

while True:                         # 무한 루프 시작 → ESC 키 입력 전까지 반복 실행
    ret, frame = cap.read()         # 웹캠에서 프레임 한 장 읽기 → ret(성공여부), frame(영상 이미지) 반환

    if not ret:                     # 프레임 읽기 실패 여부 확인 → 실패 시 루프 탈출
        break                       # 읽기 실패 시 루프 종료 → 프로그램 안전하게 종료

    frame = cv2.flip(frame, 1)      # 프레임 좌우 반전(거울 모드) → 사용자가 거울처럼 자연스럽게 보임

    h, w, _ = frame.shape           # 프레임의 높이(h), 너비(w), 채널 수 추출 → 랜드마크 좌표 변환에 사용

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB 색상 변환 → MediaPipe가 요구하는 RGB 형식으로 변환

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)  # numpy 배열을 MediaPipe Image 객체로 변환 → 새 API가 요구하는 입력 형식으로 변환

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) or (frame_idx * 33)  # 현재 프레임의 타임스탬프(ms) 계산 → VIDEO 모드에서 필수로 필요한 시간 정보 제공

    results = face_landmarker.detect_for_video(mp_image, timestamp_ms)  # MediaPipe Image와 타임스탬프로 랜드마크 검출 → 468개 랜드마크 좌표 결과 반환

    if results.face_landmarks:                          # 얼굴 랜드마크 감지 결과 존재 여부 확인 → 얼굴이 감지된 경우에만 처리
        for face_landmarks in results.face_landmarks:   # 감지된 각 얼굴의 랜드마크 순회 → 얼굴별 랜드마크 처리
            for landmark in face_landmarks:             # 얼굴의 468개 랜드마크 각각 순회 → 각 랜드마크 좌표 처리

                x = int(landmark.x * w)     # 정규화된 x 좌표(0~1)를 실제 픽셀 x 좌표로 변환 → 이미지 너비에 맞는 실제 좌표 획득
                y = int(landmark.y * h)     # 정규화된 y 좌표(0~1)를 실제 픽셀 y 좌표로 변환 → 이미지 높이에 맞는 실제 좌표 획득

                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # 랜드마크 위치에 반지름 1px 초록색 점 그리기 → 468개 랜드마크가 얼굴 위에 점으로 표시됨

    cv2.imshow("Face Landmark Detection", frame)    # 랜드마크가 그려진 프레임을 화면에 표시 → "Face Landmark Detection" 창에 실시간 영상 출력

    frame_idx += 1                  # 프레임 번호 1 증가 → 다음 프레임의 타임스탬프 계산 준비

    key = cv2.waitKey(1)            # 1ms 동안 키 입력 대기 → 눌린 키의 ASCII 코드 반환
    if key == 27:                   # ESC 키(ASCII 27) 입력 여부 확인 → ESC 누르면 루프 종료
        break                       # ESC 키 입력 시 루프 탈출 → 프로그램 종료 절차 시작

cap.release()                       # 웹캠 장치 해제 → 다른 프로그램이 웹캠 사용 가능하도록 반환
cv2.destroyAllWindows()             # 열린 모든 OpenCV 창 닫기 → 화면에 표시된 영상 창 종료
face_landmarker.close()             # FaceLandmarker 검출기 리소스 해제 → 메모리 및 모델 자원 반환
