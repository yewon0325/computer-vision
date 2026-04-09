# SORT 알고리즘을 활용한 다중 객체 추적기 구현

## 1. 문제

본 과제의 목표는 비디오에서 여러 객체를 동시에 추적하는 다중 객체 추적기(Multi-Object Tracker)를 구현하는 것이다.
이를 위해 사전 학습된 객체 검출 모델인 **YOLOv3**를 사용하여 각 프레임에서 객체를 검출하고, 검출된 객체들을 **SORT(Simple Online and Realtime Tracking)** 알고리즘으로 연속 프레임 간 연결하여 각 객체에 고유 ID를 부여한다.
최종적으로 비디오 프레임 위에 객체의 **경계 상자(Bounding Box)** 와 **추적 ID** 를 표시하여 실시간 추적 결과를 시각화한다.

---

## 2. 요구사항

과제에서 요구한 내용은 다음과 같다.

* **객체 검출기 구현**
  YOLOv3와 같은 사전 훈련된 객체 검출 모델을 사용하여 각 프레임에서 객체를 검출한다.

* **SORT 추적기 초기화**
  검출된 객체의 경계 상자를 입력으로 받아 SORT 추적기를 초기화한다.

* **객체 추적**
  각 프레임마다 새롭게 검출된 객체와 기존 추적 객체를 연관시켜 객체 추적을 유지한다.

* **결과 시각화**
  추적된 각 객체에 고유 ID를 부여하고, 해당 ID와 경계 상자를 비디오 프레임에 표시하여 출력한다.

---

## 3. 개념

### 3.1 YOLOv3 객체 검출

YOLOv3는 이미지 전체를 한 번에 입력받아 객체의 위치와 클래스를 동시에 예측하는 객체 검출 모델이다.
본 과제에서는 OpenCV의 `dnn` 모듈을 사용하여 `yolov3.cfg`와 `yolov3.weights` 파일을 불러오고, 각 프레임에서 사람, 자동차, 버스, 트럭 등 교통 영상에 필요한 객체를 검출하였다.

### 3.2 SORT 알고리즘

SORT는 다중 객체 추적을 위한 대표적인 알고리즘으로, 다음 두 가지 핵심 요소를 사용한다.

* **칼만 필터(Kalman Filter)**
  이전 상태를 바탕으로 다음 프레임에서 객체의 위치를 예측한다.

* **헝가리안 알고리즘(Hungarian Algorithm)**
  현재 프레임에서 검출된 객체와 기존 추적 객체 사이의 최적 매칭을 수행한다.

즉, YOLOv3가 각 프레임에서 객체를 찾고, SORT가 이전 프레임의 객체와 현재 프레임의 객체를 연결하여 같은 객체에 동일한 ID를 유지하도록 만든다.

### 3.3 IoU(Intersection over Union)

객체 검출 결과와 예측된 추적 결과의 유사도를 비교하기 위해 **IoU**를 사용하였다.
IoU 값이 높을수록 두 박스가 많이 겹친다는 의미이며, 이를 기준으로 객체 매칭의 신뢰도를 판단한다.

### 3.4 FPS(Frame Per Second)

FPS는 초당 처리하는 프레임 수를 의미한다.
즉, 프로그램이 얼마나 빠르게 영상을 처리하고 있는지 보여주는 지표이며, 값이 높을수록 더 부드럽게 동작한다.

---

## 4. 전체 코드

```python
import cv2 as cv  # OpenCV를 사용하기 위해 불러옴 -> 영상 읽기, YOLO 추론, 박스 그리기가 가능해짐
import numpy as np  # NumPy를 사용하기 위해 불러옴 -> 벡터/행렬 계산과 배열 처리가 가능해짐
import time  # 시간 계산을 위해 불러옴 -> FPS를 화면에 표시할 수 있음
from pathlib import Path  # 경로를 안정적으로 다루기 위해 불러옴 -> 파일 위치를 안전하게 찾을 수 있음
from scipy.optimize import linear_sum_assignment  # 헝가리안 알고리즘을 쓰기 위해 불러옴 -> 검출 결과와 추적 객체를 최적으로 매칭할 수 있음

COCO_CLASSES = [  # YOLOv3의 COCO 클래스 이름을 정의함 -> class_id를 사람이 읽을 수 있는 이름으로 바꿀 수 있음
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",  # COCO 앞부분 클래스를 저장함 -> 사람/차량/교통 관련 객체 이름을 사용할 수 있음
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",  # COCO 클래스를 계속 저장함 -> 검출된 class_id를 정확히 해석할 수 있음
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",  # COCO 클래스를 계속 저장함 -> 다양한 객체 이름을 참조할 수 있음
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",  # COCO 클래스를 계속 저장함 -> YOLO 기본 클래스 체계를 그대로 유지할 수 있음
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",  # COCO 클래스를 계속 저장함 -> 전체 인덱스 순서를 맞출 수 있음
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",  # COCO 클래스를 계속 저장함 -> 잘못된 클래스 해석을 막을 수 있음
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",  # COCO 클래스를 계속 저장함 -> class_id 매핑이 정확해짐
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"  # COCO 마지막 클래스를 저장함 -> YOLOv3의 80개 클래스 구성이 완성됨
]  # COCO 클래스 정의를 마침 -> class_id를 문자열 이름으로 바꿀 준비가 끝남

TRACK_CLASS_NAMES = {"person", "bicycle", "car", "motorbike", "bus", "truck"}  # 추적할 클래스만 따로 지정함 -> 교통 영상에서 필요한 객체만 추적해 오검출을 줄일 수 있음

def bbox_to_z(bbox):  # 박스를 칼만 필터 측정값 형태로 바꾸는 함수를 정의함 -> SORT 상태 표현으로 변환할 수 있음
    x1, y1, x2, y2 = bbox  # 입력 박스를 좌상단/우하단 좌표로 나눠 받음 -> 너비와 높이를 계산할 수 있음
    w = max(1.0, x2 - x1)  # 박스 너비를 계산함 -> 0 이하 너비로 인한 오류를 방지할 수 있음
    h = max(1.0, y2 - y1)  # 박스 높이를 계산함 -> 0 이하 높이로 인한 오류를 방지할 수 있음
    cx = x1 + w / 2.0  # 박스 중심 x좌표를 계산함 -> 칼만 필터 상태의 중심 좌표로 사용할 수 있음
    cy = y1 + h / 2.0  # 박스 중심 y좌표를 계산함 -> 칼만 필터 상태의 중심 좌표로 사용할 수 있음
    s = w * h  # 박스 면적을 계산함 -> SORT에서 크기 상태값으로 사용할 수 있음
    r = w / h  # 박스 가로세로 비율을 계산함 -> SORT에서 종횡비 상태값으로 사용할 수 있음
    return np.array([[cx], [cy], [s], [r]], dtype=np.float32)  # 측정 벡터를 반환함 -> 칼만 필터가 검출 박스를 업데이트할 수 있음

def x_to_bbox(state):  # 칼만 필터 상태를 다시 박스 좌표로 바꾸는 함수를 정의함 -> 예측 결과를 화면에 그릴 수 있음
    flat_state = np.asarray(state).reshape(-1)  # 상태 벡터를 1차원으로 펼침 -> 값을 쉽게 꺼낼 수 있음
    cx = float(flat_state[0])  # 중심 x좌표를 꺼냄 -> 박스 복원에 사용할 수 있음
    cy = float(flat_state[1])  # 중심 y좌표를 꺼냄 -> 박스 복원에 사용할 수 있음
    s = max(1.0, float(flat_state[2]))  # 면적을 꺼내고 최소값을 보정함 -> 음수/0 면적으로 인한 오류를 막을 수 있음
    r = max(0.1, float(flat_state[3]))  # 종횡비를 꺼내고 최소값을 보정함 -> 비정상적인 박스 형태를 막을 수 있음
    w = np.sqrt(s * r)  # 면적과 비율로부터 너비를 복원함 -> 원래 박스 폭을 근사할 수 있음
    h = s / w  # 면적과 너비로부터 높이를 복원함 -> 원래 박스 높이를 근사할 수 있음
    x1 = cx - w / 2.0  # 좌상단 x좌표를 계산함 -> 그릴 수 있는 박스 좌표를 얻음
    y1 = cy - h / 2.0  # 좌상단 y좌표를 계산함 -> 그릴 수 있는 박스 좌표를 얻음
    x2 = cx + w / 2.0  # 우하단 x좌표를 계산함 -> 그릴 수 있는 박스 좌표를 얻음
    y2 = cy + h / 2.0  # 우하단 y좌표를 계산함 -> 그릴 수 있는 박스 좌표를 얻음
    return np.array([x1, y1, x2, y2], dtype=np.float32)  # 복원된 박스를 반환함 -> 예측 박스를 시각화할 수 있음

def compute_iou(box_a, box_b):  # 두 박스의 IoU를 계산하는 함수를 정의함 -> 검출 결과와 추적 결과의 겹침 정도를 비교할 수 있음
    x_left = max(box_a[0], box_b[0])  # 교집합의 왼쪽 x좌표를 계산함 -> 겹치는 영역을 구할 수 있음
    y_top = max(box_a[1], box_b[1])  # 교집합의 위쪽 y좌표를 계산함 -> 겹치는 영역을 구할 수 있음
    x_right = min(box_a[2], box_b[2])  # 교집합의 오른쪽 x좌표를 계산함 -> 겹치는 영역을 구할 수 있음
    y_bottom = min(box_a[3], box_b[3])  # 교집합의 아래쪽 y좌표를 계산함 -> 겹치는 영역을 구할 수 있음
    inter_w = max(0.0, x_right - x_left)  # 교집합 너비를 계산함 -> 겹치지 않으면 0이 됨
    inter_h = max(0.0, y_bottom - y_top)  # 교집합 높이를 계산함 -> 겹치지 않으면 0이 됨
    inter_area = inter_w * inter_h  # 교집합 넓이를 계산함 -> IoU 분자 값이 됨
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])  # 첫 번째 박스 넓이를 계산함 -> IoU 분모 계산에 사용됨
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])  # 두 번째 박스 넓이를 계산함 -> IoU 분모 계산에 사용됨
    union_area = area_a + area_b - inter_area  # 합집합 넓이를 계산함 -> IoU 분모 값을 얻을 수 있음
    if union_area <= 0.0:  # 합집합이 0 이하인지 확인함 -> 0으로 나누는 오류를 막을 수 있음
        return 0.0  # 예외 상황에서는 IoU 0을 반환함 -> 잘못된 매칭을 막을 수 있음
    return inter_area / union_area  # IoU 값을 반환함 -> 겹침 정도를 수치로 비교할 수 있음

def associate_detections_to_tracks(detections, predictions, det_classes, trk_classes, iou_threshold=0.3):  # 검출 결과와 추적 객체를 매칭하는 함수를 정의함 -> SORT의 데이터 연관 단계가 가능해짐
    if len(predictions) == 0:  # 현재 활성 추적 객체가 없는지 확인함 -> 처음 프레임처럼 비교 대상이 없는 경우를 처리할 수 있음
        return np.empty((0, 2), dtype=np.int32), np.arange(len(detections)), np.empty((0,), dtype=np.int32)  # 모든 검출을 새 트랙으로 만들 수 있도록 반환함 -> 첫 프레임에서도 추적을 시작할 수 있음
    if len(detections) == 0:  # 현재 검출된 객체가 없는지 확인함 -> 추적 객체만 나이 증가시키는 경우를 처리할 수 있음
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int32), np.arange(len(predictions))  # 모든 트랙을 미매칭으로 반환함 -> 검출이 잠시 사라져도 트랙을 유지할 수 있음

    iou_matrix = np.zeros((len(detections), len(predictions)), dtype=np.float32)  # IoU 행렬을 생성함 -> 각 검출과 각 트랙의 유사도를 저장할 수 있음

    for d, det_box in enumerate(detections):  # 모든 검출 박스를 순회함 -> 각 검출을 모든 트랙과 비교할 수 있음
        for t, trk_box in enumerate(predictions):  # 모든 예측 트랙 박스를 순회함 -> 각 트랙과의 IoU를 계산할 수 있음
            if int(det_classes[d]) != int(trk_classes[t]):  # 클래스가 다른지 확인함 -> 사람과 자동차 같은 다른 종류의 객체가 섞여 매칭되는 것을 막을 수 있음
                iou_matrix[d, t] = 0.0  # 클래스가 다르면 IoU를 0으로 둠 -> 잘못된 ID 연결을 줄일 수 있음
            else:  # 클래스가 같은 경우를 처리함 -> 같은 종류끼리만 정상 비교할 수 있음
                iou_matrix[d, t] = compute_iou(det_box, trk_box)  # IoU를 계산해 저장함 -> 가장 잘 겹치는 트랙을 찾을 수 있음

    cost_matrix = 1.0 - iou_matrix  # 비용 행렬을 만듦 -> 헝가리안 알고리즘이 최소 비용 매칭을 수행할 수 있음
    row_indices, col_indices = linear_sum_assignment(cost_matrix)  # 헝가리안 알고리즘으로 최적 매칭을 수행함 -> 전체적으로 가장 좋은 연결 조합을 찾을 수 있음

    matched_pairs = []  # 유효한 매칭 쌍을 저장할 리스트를 만듦 -> 임계값을 넘는 연결만 남길 수 있음
    unmatched_detections = list(range(len(detections)))  # 처음엔 모든 검출을 미매칭으로 둠 -> 이후 매칭된 것만 제거할 수 있음
    unmatched_tracks = list(range(len(predictions)))  # 처음엔 모든 트랙을 미매칭으로 둠 -> 이후 매칭된 것만 제거할 수 있음

    for row, col in zip(row_indices, col_indices):  # 헝가리안 결과를 하나씩 확인함 -> 실제로 쓸 매칭인지 검증할 수 있음
        if iou_matrix[row, col] < iou_threshold:  # IoU가 기준보다 낮은지 확인함 -> 약한 연결은 버릴 수 있음
            continue  # 기준보다 낮으면 매칭으로 쓰지 않음 -> ID 스위치를 줄일 수 있음
        matched_pairs.append([row, col])  # 유효한 매칭을 저장함 -> 이후 트랙 업데이트에 사용할 수 있음
        unmatched_detections.remove(row)  # 매칭된 검출을 미매칭 목록에서 제거함 -> 새 트랙 생성을 막을 수 있음
        unmatched_tracks.remove(col)  # 매칭된 트랙을 미매칭 목록에서 제거함 -> 불필요한 나이 증가를 막을 수 있음

    return np.array(matched_pairs, dtype=np.int32), np.array(unmatched_detections, dtype=np.int32), np.array(unmatched_tracks, dtype=np.int32)  # 매칭 결과를 반환함 -> 업데이트/생성/삭제 단계를 진행할 수 있음

class YOLOv3Detector:  # YOLOv3 검출기를 클래스로 정의함 -> 객체 검출 기능을 재사용하기 쉬워짐
    def __init__(self, cfg_path, weights_path, conf_threshold=0.4, nms_threshold=0.4):  # 검출기 초기화 함수를 정의함 -> 모델과 임계값을 설정할 수 있음
        self.conf_threshold = conf_threshold  # 신뢰도 임계값을 저장함 -> 너무 약한 검출을 걸러낼 수 있음
        self.nms_threshold = nms_threshold  # NMS 임계값을 저장함 -> 중복 박스를 줄일 수 있음
        self.class_names = COCO_CLASSES  # COCO 클래스 이름 목록을 저장함 -> class_id를 문자열로 바꿀 수 있음
        self.net = cv.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))  # YOLOv3 네트워크를 로드함 -> 학습된 객체 검출 모델을 사용할 수 있음
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)  # OpenCV 백엔드를 사용하도록 설정함 -> 별도 환경 없이 기본 DNN 실행이 가능해짐
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)  # CPU에서 실행하도록 설정함 -> 대부분의 PC에서 바로 실행할 수 있음
        self.output_layers = self.net.getUnconnectedOutLayersNames()  # 출력 레이어 이름을 가져옴 -> forward 결과를 올바르게 받을 수 있음

    def detect(self, frame):  # 한 프레임에서 객체를 검출하는 함수를 정의함 -> 매 프레임마다 YOLO 검출을 수행할 수 있음
        frame_height, frame_width = frame.shape[:2]  # 프레임 높이와 너비를 구함 -> 정규화된 검출 결과를 실제 좌표로 바꿀 수 있음
        blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)  # 이미지를 YOLO 입력 blob으로 변환함 -> 정규화된 입력으로 안정적인 추론이 가능해짐
        self.net.setInput(blob)  # 네트워크 입력을 설정함 -> 현재 프레임으로 추론할 준비가 됨
        outputs = self.net.forward(self.output_layers)  # YOLO 추론을 수행함 -> 여러 후보 박스와 클래스 점수를 얻을 수 있음

        boxes_xywh = []  # NMS에 넣을 박스 목록을 저장할 리스트를 만듦 -> 중복 제거 준비를 할 수 있음
        confidences = []  # 각 박스의 신뢰도를 저장할 리스트를 만듦 -> 낮은 점수를 걸러낼 수 있음
        class_ids = []  # 각 박스의 클래스 번호를 저장할 리스트를 만듦 -> 어떤 객체인지 알 수 있음

        for output in outputs:  # YOLO의 각 출력 레이어를 순회함 -> 모든 후보 객체를 확인할 수 있음
            for detection in output:  # 각 후보 검출 결과를 순회함 -> 박스와 점수를 하나씩 처리할 수 있음
                scores = detection[5:]  # 클래스별 점수만 따로 꺼냄 -> 가장 높은 클래스 점수를 찾을 수 있음
                class_id = int(np.argmax(scores))  # 가장 높은 점수의 클래스 인덱스를 찾음 -> 어떤 객체인지 결정할 수 있음
                confidence = float(scores[class_id])  # 해당 클래스의 신뢰도를 가져옴 -> 검출 품질을 판단할 수 있음

                if confidence < self.conf_threshold:  # 신뢰도가 기준보다 낮은지 확인함 -> 약한 검출을 제거할 수 있음
                    continue  # 기준보다 낮으면 건너뜀 -> 오검출이 줄어듦

                class_name = self.class_names[class_id]  # class_id를 이름으로 바꿈 -> 추적할 클래스인지 확인할 수 있음

                if class_name not in TRACK_CLASS_NAMES:  # 추적 대상 클래스인지 확인함 -> 교통 영상에 필요 없는 객체를 제외할 수 있음
                    continue  # 추적 대상이 아니면 건너뜀 -> 처리량을 줄이고 결과를 깔끔하게 만들 수 있음

                center_x = int(detection[0] * frame_width)  # 박스 중심 x좌표를 실제 픽셀로 변환함 -> 화면 좌표로 바꿀 수 있음
                center_y = int(detection[1] * frame_height)  # 박스 중심 y좌표를 실제 픽셀로 변환함 -> 화면 좌표로 바꿀 수 있음
                box_w = int(detection[2] * frame_width)  # 박스 너비를 실제 픽셀로 변환함 -> 실제 크기의 박스를 만들 수 있음
                box_h = int(detection[3] * frame_height)  # 박스 높이를 실제 픽셀로 변환함 -> 실제 크기의 박스를 만들 수 있음

                x1 = max(0, center_x - box_w // 2)  # 좌상단 x좌표를 계산함 -> 박스 시작점을 얻을 수 있음
                y1 = max(0, center_y - box_h // 2)  # 좌상단 y좌표를 계산함 -> 박스 시작점을 얻을 수 있음
                x2 = min(frame_width - 1, center_x + box_w // 2)  # 우하단 x좌표를 계산함 -> 박스 끝점을 얻을 수 있음
                y2 = min(frame_height - 1, center_y + box_h // 2)  # 우하단 y좌표를 계산함 -> 박스 끝점을 얻을 수 있음

                if x2 <= x1 or y2 <= y1:  # 박스가 비정상적인지 확인함 -> 잘못된 좌표를 걸러낼 수 있음
                    continue  # 비정상 박스는 건너뜀 -> 추적기 오류를 줄일 수 있음

                boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])  # NMS용 xywh 박스를 저장함 -> 중복 제거 입력을 만들 수 있음
                confidences.append(confidence)  # 박스 신뢰도를 저장함 -> NMS와 결과 출력에 사용할 수 있음
                class_ids.append(class_id)  # 박스 클래스 번호를 저장함 -> 추적 시 클래스 정보를 유지할 수 있음

        detections = []  # 최종 검출 결과를 담을 리스트를 만듦 -> NMS 후 결과를 반환할 수 있음

        if len(boxes_xywh) == 0:  # 후보 박스가 하나도 없는지 확인함 -> 빈 결과를 바로 반환할 수 있음
            return detections  # 빈 리스트를 반환함 -> 검출이 없는 프레임도 정상 처리할 수 있음

        indices = cv.dnn.NMSBoxes(boxes_xywh, confidences, self.conf_threshold, self.nms_threshold)  # NMS를 수행함 -> 겹치는 중복 박스를 제거할 수 있음

        if len(indices) > 0:  # NMS 후 남은 박스가 있는지 확인함 -> 최종 검출만 추려낼 수 있음
            for idx in np.array(indices).flatten():  # 남은 인덱스를 하나씩 순회함 -> 최종 검출 리스트를 만들 수 있음
                x, y, w, h = boxes_xywh[idx]  # xywh 박스를 꺼냄 -> xyxy 형식으로 바꿀 수 있음
                score = confidences[idx]  # 해당 박스의 신뢰도를 꺼냄 -> 필요 시 표시할 수 있음
                class_id = class_ids[idx]  # 해당 박스의 클래스 번호를 꺼냄 -> 어떤 객체인지 유지할 수 있음
                detections.append([float(x), float(y), float(x + w), float(y + h), float(score), float(class_id)])  # SORT에 넣을 xyxy 형식으로 저장함 -> 추적기로 바로 전달할 수 있음

        return detections  # 최종 검출 결과를 반환함 -> 현재 프레임의 객체 추적을 진행할 수 있음

class KalmanBoxTracker:  # SORT의 개별 트랙 클래스를 정의함 -> 각 객체의 상태와 ID를 따로 관리할 수 있음
    count = 0  # 전역 트랙 ID 카운터를 정의함 -> 새 객체마다 고유 ID를 부여할 수 있음

    def __init__(self, bbox, class_id):  # 새 트랙을 만드는 초기화 함수를 정의함 -> 검출된 객체를 추적 객체로 바꿀 수 있음
        self.id = KalmanBoxTracker.count  # 현재 카운트를 이 트랙의 ID로 저장함 -> 각 객체가 고유 번호를 가지게 됨
        KalmanBoxTracker.count += 1  # 다음 트랙을 위해 카운트를 증가시킴 -> ID가 중복되지 않게 됨
        self.class_id = int(class_id)  # 이 트랙의 클래스 번호를 저장함 -> 차량/사람 종류를 유지할 수 있음
        self.time_since_update = 0  # 마지막 업데이트 후 경과 프레임 수를 0으로 둠 -> 방금 생성된 트랙임을 표시할 수 있음
        self.hits = 1  # 생성 시 검출 1회를 기록함 -> 트랙 신뢰도를 판단할 수 있음
        self.hit_streak = 1  # 연속 검출 횟수를 1로 시작함 -> 안정된 트랙만 출력할 수 있음
        self.age = 0  # 전체 생존 프레임 수를 0으로 둠 -> 트랙 수명을 관리할 수 있음

        self.kf = cv.KalmanFilter(7, 4)  # 7차원 상태와 4차원 측정값의 칼만 필터를 생성함 -> SORT 상태 예측과 보정이 가능해짐
        self.kf.transitionMatrix = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)  # 상태 전이 행렬을 설정함 -> 위치와 크기를 속도로 예측할 수 있음
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)  # 측정 행렬을 설정함 -> 검출 박스로 상태를 보정할 수 있음
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2  # 프로세스 노이즈를 설정함 -> 예측의 흔들림을 부드럽게 조절할 수 있음
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1  # 측정 노이즈를 설정함 -> 검출 오차를 적당히 반영할 수 있음
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)  # 초기 오차 공분산을 설정함 -> 초기 추적 불확실성을 관리할 수 있음
        self.kf.statePost = np.zeros((7, 1), dtype=np.float32)  # 초기 상태 벡터를 0으로 둠 -> 검출 박스로 초기화할 준비를 할 수 있음
        self.kf.statePost[:4] = bbox_to_z(bbox)  # 처음 검출된 박스로 상태를 초기화함 -> 첫 프레임부터 해당 객체를 추적할 수 있음

    def predict(self):  # 다음 프레임 상태를 예측하는 함수를 정의함 -> 검출이 없어도 객체 위치를 추정할 수 있음
        current_state = self.kf.statePost.copy()  # 현재 상태를 복사함 -> 예측 전 상태를 안전하게 확인할 수 있음
        if current_state[2, 0] + current_state[6, 0] <= 0:  # 다음 면적이 0 이하가 되는지 확인함 -> 잘못된 크기 예측을 막을 수 있음
            self.kf.statePost[6, 0] = 0.0  # 면적 변화 속도를 0으로 보정함 -> 박스 크기가 무너지는 현상을 줄일 수 있음
        predicted_state = self.kf.predict()  # 칼만 필터 예측을 수행함 -> 다음 위치와 크기를 얻을 수 있음
        self.age += 1  # 트랙 나이를 1 증가시킴 -> 얼마나 오래 살아있는 트랙인지 알 수 있음
        self.time_since_update += 1  # 마지막 업데이트 후 경과 프레임을 1 증가시킴 -> 오래 미검출된 트랙을 제거할 수 있음
        if self.time_since_update > 0:  # 이번 프레임에서 아직 검출로 갱신되지 않았는지 확인함 -> 연속 검출 여부를 관리할 수 있음
            self.hit_streak = 0  # 연속 검출 횟수를 끊어줌 -> 불안정한 트랙 출력을 줄일 수 있음
        return x_to_bbox(predicted_state)  # 예측 상태를 박스로 바꿔 반환함 -> 매칭 단계에서 사용할 수 있음

    def update(self, bbox):  # 검출 박스로 트랙을 갱신하는 함수를 정의함 -> 예측한 상태를 실제 검출에 맞게 보정할 수 있음
        self.time_since_update = 0  # 방금 갱신되었음을 표시함 -> 살아있는 트랙으로 유지할 수 있음
        self.hits += 1  # 누적 검출 횟수를 증가시킴 -> 트랙 신뢰도를 높일 수 있음
        self.hit_streak += 1  # 연속 검출 횟수를 증가시킴 -> 안정된 트랙 여부를 판단할 수 있음
        measurement = bbox_to_z(bbox)  # 검출 박스를 측정 벡터로 바꿈 -> 칼만 필터 보정 입력을 만들 수 있음
        self.kf.correct(measurement)  # 칼만 필터 보정을 수행함 -> 예측 위치를 실제 검출에 맞게 수정할 수 있음

    def get_state(self):  # 현재 상태를 박스로 얻는 함수를 정의함 -> 화면 표시용 좌표를 꺼낼 수 있음
        return x_to_bbox(self.kf.statePost)  # 현재 상태를 박스로 반환함 -> 최신 추적 위치를 사용할 수 있음

class SortTracker:  # 여러 개의 트랙을 관리하는 SORT 추적기 클래스를 정의함 -> 다중 객체 추적을 한 번에 처리할 수 있음
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):  # 추적기 초기화 함수를 정의함 -> 주요 파라미터를 설정할 수 있음
        self.max_age = max_age  # 미검출 허용 프레임 수를 저장함 -> 잠시 가려진 객체도 바로 지우지 않을 수 있음
        self.min_hits = min_hits  # 출력하기 위한 최소 검출 횟수를 저장함 -> 불안정한 초기 트랙을 줄일 수 있음
        self.iou_threshold = iou_threshold  # 매칭 IoU 기준값을 저장함 -> 너무 약한 연결을 막을 수 있음
        self.tracks = []  # 현재 활성 트랙 목록을 저장할 리스트를 만듦 -> 여러 객체를 동시에 관리할 수 있음
        self.frame_count = 0  # 처리한 프레임 수를 0으로 시작함 -> 초기 구간 예외 처리를 할 수 있음

    def update(self, detections):  # 현재 프레임 검출 결과로 추적기를 갱신하는 함수를 정의함 -> SORT 전체 흐름을 수행할 수 있음
        self.frame_count += 1  # 프레임 카운트를 증가시킴 -> 몇 번째 프레임인지 관리할 수 있음

        if len(detections) == 0:  # 검출 결과가 없는지 확인함 -> 빈 프레임도 안전하게 처리할 수 있음
            det_array = np.empty((0, 6), dtype=np.float32)  # 빈 검출 배열을 만듦 -> 아래 로직을 동일하게 돌릴 수 있음
        else:  # 검출 결과가 있는 경우를 처리함 -> 배열 형태로 바꿔 연산할 수 있음
            det_array = np.asarray(detections, dtype=np.float32)  # 검출 리스트를 NumPy 배열로 변환함 -> 슬라이싱과 계산이 쉬워짐

        predicted_boxes = []  # 각 트랙의 예측 박스를 저장할 리스트를 만듦 -> 매칭 단계에 사용할 수 있음
        track_classes = []  # 각 트랙의 클래스 번호를 저장할 리스트를 만듦 -> 같은 클래스끼리만 연결할 수 있음

        for track in self.tracks:  # 현재 모든 트랙을 순회함 -> 다음 프레임 위치를 먼저 예측할 수 있음
            predicted_boxes.append(track.predict())  # 각 트랙의 예측 박스를 저장함 -> 검출 결과와 비교할 수 있음
            track_classes.append(track.class_id)  # 각 트랙의 클래스 번호를 저장함 -> 클래스 일치 매칭이 가능해짐

        if len(predicted_boxes) == 0:  # 예측 박스가 없는지 확인함 -> 활성 트랙이 하나도 없는 상황을 처리할 수 있음
            predicted_boxes = np.empty((0, 4), dtype=np.float32)  # 빈 예측 배열을 만듦 -> 매칭 함수에 그대로 넣을 수 있음
            track_classes = np.empty((0,), dtype=np.int32)  # 빈 클래스 배열을 만듦 -> 매칭 함수에 그대로 넣을 수 있음
        else:  # 예측 박스가 있는 경우를 처리함 -> 배열 형태로 바꿔 계산할 수 있음
            predicted_boxes = np.asarray(predicted_boxes, dtype=np.float32)  # 예측 박스를 배열로 변환함 -> IoU 행렬 계산이 쉬워짐
            track_classes = np.asarray(track_classes, dtype=np.int32)  # 트랙 클래스 목록을 배열로 변환함 -> 클래스 비교가 쉬워짐

        det_boxes = det_array[:, :4] if len(det_array) > 0 else np.empty((0, 4), dtype=np.float32)  # 검출 박스만 따로 꺼냄 -> 매칭 계산에 사용할 수 있음
        det_classes = det_array[:, 5].astype(np.int32) if len(det_array) > 0 else np.empty((0,), dtype=np.int32)  # 검출 클래스만 따로 꺼냄 -> 같은 클래스끼리만 연결할 수 있음

        matches, unmatched_dets, unmatched_trks = associate_detections_to_tracks(det_boxes, predicted_boxes, det_classes, track_classes, self.iou_threshold)  # 검출과 트랙을 매칭함 -> 업데이트/생성/유지 대상을 나눌 수 있음

        for det_idx, trk_idx in matches:  # 매칭된 쌍을 하나씩 순회함 -> 해당 트랙을 실제 검출로 보정할 수 있음
            self.tracks[trk_idx].update(det_boxes[det_idx])  # 매칭된 트랙을 검출 박스로 업데이트함 -> ID를 유지하며 위치를 정확히 맞출 수 있음

        for det_idx in unmatched_dets:  # 매칭되지 않은 검출을 순회함 -> 새 객체가 등장한 경우를 처리할 수 있음
            self.tracks.append(KalmanBoxTracker(det_boxes[det_idx], det_classes[det_idx]))  # 새 트랙을 생성해 추가함 -> 새로운 객체에 새 ID를 부여할 수 있음

        alive_tracks = []  # 아직 유지할 트랙을 담을 리스트를 만듦 -> 오래된 트랙을 제거할 수 있음
        outputs = []  # 화면에 출력할 확정 트랙 결과를 담을 리스트를 만듦 -> 안정적인 추적 결과만 표시할 수 있음

        for track in self.tracks:  # 모든 트랙을 순회함 -> 삭제 여부와 출력 여부를 판단할 수 있음
            if track.time_since_update <= self.max_age:  # 너무 오래 미검출되지 않았는지 확인함 -> 잠깐 사라진 객체는 유지할 수 있음
                alive_tracks.append(track)  # 살아있는 트랙으로 보관함 -> 다음 프레임에서도 계속 추적할 수 있음

            if track.time_since_update == 0 and (track.hits >= self.min_hits or self.frame_count <= self.min_hits):  # 현재 프레임에 갱신되었고 충분히 안정적인지 확인함 -> 믿을 수 있는 트랙만 출력할 수 있음
                box = track.get_state()  # 현재 트랙 박스를 얻음 -> 화면에 그릴 좌표를 준비할 수 있음
                outputs.append([box[0], box[1], box[2], box[3], track.id, track.class_id])  # 박스와 ID와 클래스 정보를 저장함 -> 시각화 단계에서 사용할 수 있음

        self.tracks = alive_tracks  # 살아있는 트랙만 남김 -> 오래된 트랙이 자동으로 정리됨

        if len(outputs) == 0:  # 출력할 트랙이 없는지 확인함 -> 빈 프레임도 안전하게 처리할 수 있음
            return np.empty((0, 6), dtype=np.float32)  # 빈 결과를 반환함 -> 그릴 것이 없음을 알 수 있음

        return np.asarray(outputs, dtype=np.float32)  # 최종 추적 결과를 반환함 -> ID가 포함된 박스를 시각화할 수 있음

def make_color(track_id):  # 트랙 ID별 고정 색상을 만드는 함수를 정의함 -> 같은 ID는 항상 같은 색으로 볼 수 있음
    return ((37 * track_id) % 255, (17 * track_id) % 255, (29 * track_id) % 255)  # ID 기반 색상을 반환함 -> 객체별로 쉽게 구분할 수 있음

def draw_tracks(frame, tracks, class_names):  # 추적 결과를 프레임에 그리는 함수를 정의함 -> ID와 박스를 화면에 표시할 수 있음
    frame_height, frame_width = frame.shape[:2]  # 프레임 크기를 구함 -> 박스 좌표를 화면 범위 안으로 보정할 수 있음

    for track in tracks:  # 모든 추적 결과를 순회함 -> 각 객체를 하나씩 그릴 수 있음
        x1, y1, x2, y2, track_id, class_id = track  # 결과에서 박스와 ID와 클래스 번호를 꺼냄 -> 시각화에 필요한 정보를 얻을 수 있음
        x1 = int(max(0, min(frame_width - 1, x1)))  # 좌상단 x좌표를 화면 범위 안으로 보정함 -> 박스가 화면 밖으로 벗어나는 것을 막을 수 있음
        y1 = int(max(0, min(frame_height - 1, y1)))  # 좌상단 y좌표를 화면 범위 안으로 보정함 -> 박스가 화면 밖으로 벗어나는 것을 막을 수 있음
        x2 = int(max(0, min(frame_width - 1, x2)))  # 우하단 x좌표를 화면 범위 안으로 보정함 -> 박스가 화면 밖으로 벗어나는 것을 막을 수 있음
        y2 = int(max(0, min(frame_height - 1, y2)))  # 우하단 y좌표를 화면 범위 안으로 보정함 -> 박스가 화면 밖으로 벗어나는 것을 막을 수 있음

        color = make_color(int(track_id))  # 현재 ID의 색상을 만듦 -> 같은 객체를 같은 색으로 볼 수 있음
        label = f"ID {int(track_id)} | {class_names[int(class_id)]}"  # 표시할 텍스트를 만듦 -> 객체 종류와 고유 ID를 함께 확인할 수 있음

        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 객체 경계 상자를 그림 -> 추적 위치를 시각적으로 확인할 수 있음
        text_size, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.55, 2)  # 텍스트 크기를 계산함 -> 배경 박스를 적절한 크기로 만들 수 있음
        text_w, text_h = text_size  # 텍스트 너비와 높이를 꺼냄 -> 배경 사각형 좌표를 계산할 수 있음
        cv.rectangle(frame, (x1, max(0, y1 - text_h - 10)), (x1 + text_w + 6, y1), color, -1)  # 텍스트 배경 박스를 그림 -> 글자가 잘 보이게 만들 수 있음
        cv.putText(frame, label, (x1 + 3, y1 - 6), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)  # ID와 클래스명을 그림 -> 어떤 객체가 어떤 ID인지 확인할 수 있음

def main():  # 전체 실행 흐름을 담당하는 메인 함수를 정의함 -> 코드 시작점을 깔끔하게 관리할 수 있음
    base_dir = Path(__file__).resolve().parent/"L06"  # 현재 파이썬 파일이 있는 폴더를 기준 경로로 잡음 -> 관련 파일을 같은 폴더에서 쉽게 찾을 수 있음
    video_path = base_dir / "slow_traffic_small.mp4"  # 입력 비디오 경로를 지정함 -> 추적할 영상을 열 수 있음
    cfg_path = base_dir / "yolov3.cfg"  # YOLO 설정 파일 경로를 지정함 -> 네트워크 구조를 읽을 수 있음
    weights_path = base_dir / "yolov3.weights"  # YOLO 가중치 파일 경로를 지정함 -> 학습된 검출 모델을 사용할 수 있음
    output_path = base_dir / "sort_tracking_output.mp4"  # 저장할 결과 비디오 경로를 지정함 -> 추적 결과를 파일로 남길 수 있음

    detector = YOLOv3Detector(cfg_path, weights_path, conf_threshold=0.4, nms_threshold=0.4)  # YOLOv3 검출기를 생성함 -> 프레임마다 객체를 검출할 수 있음
    tracker = SortTracker(max_age=15, min_hits=3, iou_threshold=0.25)  # SORT 추적기를 생성함 -> 검출된 객체에 고유 ID를 붙여 추적할 수 있음

    cap = cv.VideoCapture(str(video_path))  # 비디오 파일을 엶 -> 프레임을 하나씩 읽을 수 있음

    if not cap.isOpened():  # 비디오가 제대로 열렸는지 확인함 -> 경로 문제를 바로 알 수 있음
        print("비디오 파일을 열 수 없습니다. 파일 경로를 확인하세요.")  # 오류 메시지를 출력함 -> 사용자가 원인을 바로 확인할 수 있음
        return  # 실행을 종료함 -> 잘못된 상태에서 계속 진행하지 않게 됨

    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))  # 비디오 프레임 너비를 가져옴 -> 출력 영상 크기를 맞출 수 있음
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))  # 비디오 프레임 높이를 가져옴 -> 출력 영상 크기를 맞출 수 있음
    fps = float(cap.get(cv.CAP_PROP_FPS))  # 원본 FPS를 가져옴 -> 출력 영상 속도를 맞출 수 있음

    if fps <= 0:  # FPS 정보를 정상적으로 못 가져왔는지 확인함 -> 저장 시 오류를 막을 수 있음
        fps = 20.0  # 기본 FPS를 20으로 설정함 -> 출력 영상을 무난하게 저장할 수 있음

    fourcc = cv.VideoWriter_fourcc(*"mp4v")  # mp4 저장용 코덱을 지정함 -> 결과 영상을 mp4로 저장할 수 있음
    writer = cv.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))  # 결과 비디오 저장기를 생성함 -> 추적 결과를 파일로 기록할 수 있음

    prev_time = time.time()  # 직전 프레임 시각을 저장함 -> FPS를 계산할 수 있음

    while True:  # 비디오 끝까지 반복하는 루프를 시작함 -> 모든 프레임을 처리할 수 있음
        ret, frame = cap.read()  # 다음 프레임을 읽음 -> 현재 프레임 영상 데이터를 얻을 수 있음

        if not ret:  # 프레임을 더 이상 읽지 못했는지 확인함 -> 비디오 끝에 도달했는지 알 수 있음
            break  # 루프를 종료함 -> 전체 처리를 마칠 수 있음

        detections = detector.detect(frame)  # 현재 프레임에서 YOLOv3 객체 검출을 수행함 -> 객체 박스와 클래스 정보를 얻을 수 있음
        tracks = tracker.update(detections)  # 검출 결과로 SORT 추적기를 갱신함 -> 각 객체에 고유 ID를 유지할 수 있음
        draw_tracks(frame, tracks, detector.class_names)  # 추적 결과를 프레임에 그림 -> ID와 박스를 눈으로 확인할 수 있음

        current_time = time.time()  # 현재 시각을 구함 -> 현재 처리 속도를 계산할 수 있음
        current_fps = 1.0 / max(current_time - prev_time, 1e-6)  # 프레임 간 시간으로 FPS를 계산함 -> 대략적인 실시간 처리 속도를 볼 수 있음
        prev_time = current_time  # 현재 시각을 다음 비교용으로 저장함 -> 다음 프레임 FPS 계산에 사용할 수 있음

        cv.putText(frame, f"FPS: {current_fps:.2f}", (20, 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # 화면 왼쪽 위에 FPS를 그림 -> 실행 속도를 실시간으로 확인할 수 있음
        writer.write(frame)  # 현재 프레임을 결과 비디오에 저장함 -> 추적 결과가 파일로 남게 됨
        cv.imshow("YOLOv3 + SORT Multi-Object Tracking", frame)  # 현재 프레임을 화면에 표시함 -> 실시간 추적 결과를 볼 수 있음

        key = cv.waitKey(1) & 0xFF  # 키 입력을 1ms 동안 확인함 -> 사용자가 중간에 종료할 수 있음
        if key == ord("q") or key == 27:  # q 또는 ESC 키를 눌렀는지 확인함 -> 수동 종료를 지원할 수 있음
            break  # 루프를 종료함 -> 사용자가 원하는 시점에 끝낼 수 있음

    cap.release()  # 비디오 입력 객체를 해제함 -> 파일 사용을 정상 종료할 수 있음
    writer.release()  # 비디오 저장 객체를 해제함 -> 결과 파일이 정상 저장되게 할 수 있음
    cv.destroyAllWindows()  # OpenCV 창을 모두 닫음 -> 실행 후 화면 자원을 정리할 수 있음
    print(f"추적 완료! 저장된 파일: {output_path}")  # 완료 메시지를 출력함 -> 결과 파일 위치를 바로 확인할 수 있음

if __name__ == "__main__":  # 이 파일을 직접 실행했는지 확인함 -> 메인 함수 자동 실행 여부를 제어할 수 있음
    main()  # 메인 함수를 실행함 -> YOLOv3 + SORT 다중 객체 추적이 시작됨
```

---

좋아, 코드가 길어서 핵심 코드가 더 있어 보이는 게 맞아.
지금 있는 5.1~5.4에 이어서 **중요한 부분을 더 추가한 버전**으로 바로 붙여넣을 수 있게 정리해줄게.

---

## 5. 핵심 코드

### 5.1 YOLOv3를 이용한 객체 검출

```python
blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
self.net.setInput(blob)
outputs = self.net.forward(self.output_layers)
```

이 부분은 현재 프레임을 YOLOv3 입력 형식으로 변환한 뒤, 신경망에 넣어 객체를 검출하는 코드이다.
입력 영상에서 사람, 자동차, 버스, 트럭 등의 객체를 찾아 경계 상자와 클래스 정보를 생성한다.

### 5.2 추적 대상 클래스만 선택

```python
class_name = self.class_names[class_id]

if class_name not in TRACK_CLASS_NAMES:
    continue
```

이 부분은 YOLOv3가 검출한 여러 객체 중에서 실제로 추적에 사용할 객체만 선택하는 코드이다.
본 과제에서는 교통 영상에 맞게 사람, 자전거, 자동차, 오토바이, 버스, 트럭만 추적 대상으로 설정하였다.
이를 통해 불필요한 객체를 제외하고 추적 성능과 처리 효율을 높일 수 있다.

### 5.3 NMS를 통한 중복 박스 제거

```python
indices = cv.dnn.NMSBoxes(boxes_xywh, confidences, self.conf_threshold, self.nms_threshold)

if len(indices) > 0:
    for idx in np.array(indices).flatten():
        x, y, w, h = boxes_xywh[idx]
        score = confidences[idx]
        class_id = class_ids[idx]
        detections.append([float(x), float(y), float(x + w), float(y + h), float(score), float(class_id)])
```

이 부분은 YOLOv3가 같은 객체에 대해 여러 개의 박스를 검출했을 때, 그중 가장 적절한 박스만 남기기 위한 코드이다.
NMS(Non-Maximum Suppression)를 적용하여 중복 검출을 줄이고, 이후 SORT가 더 안정적으로 객체를 추적할 수 있도록 한다.

### 5.4 IoU 기반 검출-추적 매칭

```python
iou_matrix[d, t] = compute_iou(det_box, trk_box)
cost_matrix = 1.0 - iou_matrix
row_indices, col_indices = linear_sum_assignment(cost_matrix)
```

이 부분은 새롭게 검출된 객체와 기존 추적 객체의 IoU를 계산하고, 헝가리안 알고리즘을 통해 최적의 매칭을 수행하는 코드이다.
이를 통해 같은 객체에는 같은 ID가 유지되도록 만든다.

### 5.5 IoU 임계값을 이용한 잘못된 매칭 방지

```python
for row, col in zip(row_indices, col_indices):
    if iou_matrix[row, col] < iou_threshold:
        continue
    matched_pairs.append([row, col])
```

이 부분은 헝가리안 알고리즘으로 연결된 결과 중에서도, 실제로 충분히 겹치는 경우만 최종 매칭으로 인정하는 코드이다.
즉, IoU 값이 너무 낮은 경우에는 같은 객체가 아니라고 판단하여 잘못된 ID 연결을 방지한다.

### 5.6 칼만 필터 상태 초기화

```python
self.kf = cv.KalmanFilter(7, 4)
self.kf.transitionMatrix = np.array([
    [1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1]
], dtype=np.float32)
self.kf.statePost[:4] = bbox_to_z(bbox)
```

이 부분은 새롭게 생성된 객체 트랙의 상태를 칼만 필터로 초기화하는 코드이다.
객체의 중심 좌표, 크기, 종횡비를 상태값으로 저장하고, 이후 프레임에서 객체의 위치를 예측할 수 있도록 준비한다.

### 5.7 칼만 필터 기반 위치 예측 및 보정

```python
predicted_state = self.kf.predict()
self.kf.correct(measurement)
```

이 부분은 객체의 다음 위치를 예측하고, 실제 검출 결과가 들어오면 예측값을 보정하는 코드이다.
즉, 객체가 잠깐 가려지거나 검출이 불안정하더라도 비교적 안정적으로 추적을 이어갈 수 있다.

### 5.8 새로운 객체에 고유 ID 부여

```python
for det_idx in unmatched_dets:
    self.tracks.append(KalmanBoxTracker(det_boxes[det_idx], det_classes[det_idx]))
```

이 부분은 현재 프레임에서 검출되었지만 기존 어떤 추적 객체와도 연결되지 않은 객체를 새로운 트랙으로 생성하는 코드이다.
새로운 객체가 등장하면 새로운 고유 ID가 부여되어 이후 프레임에서도 독립적으로 추적할 수 있다.

### 5.9 오래된 트랙 제거 및 안정된 트랙만 출력

```python
if track.time_since_update <= self.max_age:
    alive_tracks.append(track)

if track.time_since_update == 0 and (track.hits >= self.min_hits or self.frame_count <= self.min_hits):
    box = track.get_state()
    outputs.append([box[0], box[1], box[2], box[3], track.id, track.class_id])
```

이 부분은 일정 시간 동안 검출되지 않은 트랙을 제거하고, 충분히 안정적으로 검출된 트랙만 화면에 출력하는 코드이다.
이를 통해 순간적인 오검출이나 불안정한 추적 결과가 화면에 바로 나타나는 것을 줄일 수 있다.

### 5.10 추적 결과 시각화

```python
label = f"ID {int(track_id)} | {class_names[int(class_id)]}"
cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
cv.putText(frame, label, (x1 + 3, y1 - 6), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
```

이 부분은 각 객체에 대해 경계 상자와 ID, 클래스 이름을 화면에 표시하는 코드이다.
사용자는 어떤 객체가 어떤 ID로 추적되고 있는지 쉽게 확인할 수 있다.

---

## 6. 실행 방법

### 6.1 파일 구조

프로젝트 폴더 구조는 다음과 같이 구성하였다.

```text
chapter06_Dynamic Vision/
├─ sort_yolov3_tracker.py
└─ L06/
   ├─ slow_traffic_small.mp4
   ├─ yolov3.cfg
   └─ yolov3.weights
```

### 6.2 라이브러리 설치

아래 명령어를 실행하여 필요한 라이브러리를 설치한다.

```bash
pip install opencv-python numpy scipy
```

### 6.3 프로그램 실행

터미널에서 다음 명령어를 실행한다.

```bash
python sort_yolov3_tracker.py
```

### 6.4 실행 결과 저장

프로그램 실행 후 추적 결과가 표시된 영상은 다음 경로에 저장된다.

```text
L06/sort_tracking_output.mp4
```

---
출력 영상 파일: [sort_tracking_output.mp4](./chapter06_Dynamic Vision/L06/sort_tracking_output.mp4)

---

## 7. 실행 결과

프로그램을 실행하면 교통 영상의 각 프레임에서 **사람, 자전거, 자동차, 오토바이, 버스, 트럭** 등의 객체가 검출된다.
검출된 객체에는 서로 다른 색상의 경계 상자가 그려지고, 각 객체마다 **고유 ID** 가 함께 표시된다.
또한 화면 좌측 상단에는 현재 처리 속도를 나타내는 **FPS 값** 이 출력된다.

즉, 실행 결과 화면에서는 다음과 같은 내용을 확인할 수 있다.

* 이동하는 차량과 사람에 경계 상자가 표시됨
* 각 객체에 `ID 0`, `ID 1`, `ID 2` 와 같은 고유 번호가 부여됨
* 같은 객체는 프레임이 바뀌어도 동일한 ID를 유지함
* 화면 상단에 FPS가 표시됨
* 최종 결과 영상이 `sort_tracking_output.mp4` 파일로 저장됨

---

## 8. 실행 결과 분석

이번 실습에서는 YOLOv3와 SORT를 결합하여 다중 객체 추적기를 구현하였다.
YOLOv3는 각 프레임에서 객체를 검출하는 역할을 수행하고, SORT는 검출된 객체를 프레임 간 연결하여 동일 객체에 같은 ID를 부여하였다.
그 결과, 교통 영상 속 여러 차량과 사람을 동시에 추적할 수 있었고, 각 객체의 이동을 실시간으로 시각적으로 확인할 수 있었다.

특히 SORT는 칼만 필터를 사용하여 객체의 다음 위치를 예측하고, 헝가리안 알고리즘을 이용해 현재 검출 결과와 기존 트랙을 연결하므로 비교적 단순한 구조이면서도 빠르게 동작하였다.
또한 OpenCV DNN 모듈을 활용하여 YOLOv3를 직접 불러와 처리했기 때문에 추가적인 복잡한 딥러닝 프레임워크 없이도 구현이 가능했다.

---