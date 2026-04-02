import tensorflow as tf  # 텐서플로우를 사용하기 위해 불러옴 -> CIFAR-10 로드, CNN 구성, 학습, 평가, 예측이 가능해짐
from tensorflow.keras.datasets import cifar10  # CIFAR-10 데이터셋을 불러오기 위해 사용함 -> 10개 클래스의 이미지 데이터를 바로 받을 수 있음
from tensorflow.keras.models import Sequential  # 순차형 모델을 만들기 위해 사용함 -> CNN 층을 순서대로 쌓을 수 있음
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # CNN 구성에 필요한 층을 사용함 -> 특징 추출과 분류를 수행할 수 있음
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # 학습 제어를 위해 사용함 -> 과적합을 줄이고 불필요한 학습 시간을 줄일 수 있음
from pathlib import Path  # 파일 경로를 다루기 위해 사용함 -> images/dog.jpg 위치를 안전하게 찾을 수 있음
import numpy as np  # 예측 결과 처리에 사용함 -> 가장 높은 확률의 클래스를 고를 수 있음

# ===================== GPU 확인 및 설정 =====================
gpus = tf.config.list_physical_devices('GPU')  # 텐서플로우가 인식한 GPU 목록을 확인함 -> GPU 사용 가능 여부를 확인할 수 있음

if gpus:  # 사용 가능한 GPU가 있는지 확인함 -> GPU가 있으면 학습을 더 빠르게 진행할 수 있음
    try:  # GPU 메모리 증가 방식을 설정하기 위해 시도함 -> 처음부터 모든 GPU 메모리를 점유하지 않게 할 수 있음
        for gpu in gpus:  # 인식된 GPU 각각에 대해 반복함 -> 여러 GPU가 있어도 모두 설정할 수 있음
            tf.config.experimental.set_memory_growth(gpu, True)  # 필요한 만큼만 GPU 메모리를 사용하도록 설정함 -> 메모리 관련 오류를 줄이는 데 도움을 줌
        print("TensorFlow가 인식한 GPU:", gpus)  # GPU 목록을 출력함 -> 실제로 GPU가 잡혔는지 확인할 수 있음
    except RuntimeError as e:  # 장치 초기화 후 설정 시 오류가 날 수 있어 예외 처리함 -> 코드가 중단되지 않게 함
        print("GPU 설정 중 오류:", e)  # 오류 내용을 출력함 -> 설정 실패 원인을 확인할 수 있음
else:  # GPU가 없는 경우를 처리함 -> CPU로 실행됨
    print("GPU를 찾지 못했습니다. CPU로 실행됩니다.")  # GPU 미인식 상태를 출력함 -> 환경 점검이 가능해짐

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10 클래스 이름을 저장함 -> 예측 결과를 보기 쉽게 출력할 수 있음

# ===================== 데이터 로드 및 전처리 =====================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # CIFAR-10 데이터를 불러옴 -> 훈련용/테스트용 이미지와 레이블이 준비됨

x_train = x_train.astype('float32') / 255.0  # 훈련 이미지 픽셀을 정규화함 -> 0~255 값을 0~1 범위로 바꿔 학습을 안정화함
x_test = x_test.astype('float32') / 255.0  # 테스트 이미지 픽셀을 정규화함 -> 훈련 데이터와 같은 기준으로 평가할 수 있음

# ===================== CNN 모델 구성 =====================
model = Sequential()  # 순차형 CNN 모델 객체를 생성함 -> 층을 차례대로 추가할 수 있게 됨
model.add(Input(shape=(32, 32, 3)))  # 입력 이미지 크기를 지정함 -> 32x32 크기의 컬러 이미지를 받게 됨

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))  # 첫 번째 합성곱 층을 추가함 -> 기본적인 이미지 특징을 추출함
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))  # 합성곱 층을 한 번 더 추가함 -> 같은 단계의 특징을 더 풍부하게 학습함
model.add(MaxPooling2D((2, 2)))  # 풀링 층을 추가함 -> 특징 맵 크기를 줄여 계산량을 감소시킴
model.add(Dropout(0.25))  # 드롭아웃을 추가함 -> 과적합을 줄이는 데 도움을 줌

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 두 번째 블록의 합성곱 층을 추가함 -> 더 복잡한 특징을 추출할 수 있음
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # 같은 단계의 특징을 반복 추출함 -> 분류 성능 향상에 도움을 줌
model.add(MaxPooling2D((2, 2)))  # 두 번째 풀링 층을 추가함 -> 특징 맵 크기를 더 줄여 속도를 높임
model.add(Dropout(0.30))  # 드롭아웃을 추가함 -> 중간층 과적합을 줄일 수 있음

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))  # 세 번째 블록의 합성곱 층을 추가함 -> 더 높은 수준의 특징을 학습함
model.add(MaxPooling2D((2, 2)))  # 세 번째 풀링 층을 추가함 -> 마지막 특징 맵 크기를 줄여 Dense층 부담을 줄임
model.add(Dropout(0.30))  # 드롭아웃을 추가함 -> 마지막 특징 추출 단계에서도 과적합을 방지함

model.add(Flatten())  # 2차원 특징 맵을 1차원으로 펼침 -> Dense 층에 입력할 수 있는 형태가 됨
model.add(Dense(256, activation='relu'))  # 완전연결층을 추가함 -> 추출한 특징을 바탕으로 클래스를 구분함
model.add(Dropout(0.50))  # 드롭아웃을 추가함 -> Dense 구간의 과적합을 강하게 줄여줌
model.add(Dense(10, activation='softmax'))  # 출력층을 추가함 -> 10개 클래스의 확률을 출력함

# ===================== 모델 컴파일 =====================
model.compile(  # 모델의 학습 방식을 설정함 -> 옵티마이저, 손실 함수, 평가 지표가 정해짐
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Adam 옵티마이저를 사용함 -> 비교적 빠르고 안정적으로 학습할 수 있음
    loss='sparse_categorical_crossentropy',  # 정수형 레이블에 맞는 손실 함수를 사용함 -> 다중 클래스 분류 오차를 계산할 수 있음
    metrics=['accuracy']  # 정확도를 함께 확인함 -> 에폭마다 성능 변화를 볼 수 있음
)

# ===================== 콜백 설정 =====================
early_stopping = EarlyStopping(  # 조기 종료 콜백을 설정함 -> 성능 향상이 멈추면 학습을 자동으로 끝낼 수 있음
    monitor='val_accuracy',  # 검증 정확도를 감시함 -> 성능 기준으로 멈추게 됨
    patience=4,  # 4에폭 동안 개선이 없으면 중단함 -> 시간을 아낄 수 있음
    restore_best_weights=True  # 가장 성능이 좋았던 가중치로 복원함 -> 최종 모델 품질을 유지할 수 있음
)

reduce_lr = ReduceLROnPlateau(  # 학습률 감소 콜백을 설정함 -> 학습이 정체되면 더 세밀하게 학습할 수 있음
    monitor='val_loss',  # 검증 손실을 감시함 -> 손실 개선이 없을 때 학습률을 줄임
    factor=0.5,  # 학습률을 절반으로 줄임 -> 급하게 움직이던 학습을 더 안정적으로 바꿀 수 있음
    patience=2,  # 2에폭 동안 개선이 없으면 적용함 -> 너무 빨리 줄이지 않도록 함
    min_lr=1e-5  # 학습률 하한을 설정함 -> 지나치게 작아지는 것을 막음
)

# ===================== 모델 학습 =====================
history = model.fit(  # 모델 학습을 시작함 -> 훈련 데이터로 CNN이 패턴을 학습하게 됨
    x_train, y_train,  # 훈련 이미지와 정답 레이블을 전달함 -> 학습에 사용됨
    epochs=20,  # 최대 20에폭까지 학습함 -> 너무 오래 돌지 않도록 적당히 제한함
    batch_size=128,  # 배치 크기를 128로 설정함 -> GPU에서도 안정적으로 학습하기 좋음
    validation_split=0.1,  # 훈련 데이터의 10%를 검증용으로 사용함 -> 과적합 여부를 확인할 수 있음
    callbacks=[early_stopping, reduce_lr],  # 콜백을 적용함 -> 자동 조기 종료와 학습률 조정이 수행됨
    verbose=1  # 학습 과정을 출력함 -> 에폭별 손실과 정확도를 확인할 수 있음
)

# ===================== 모델 평가 =====================
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)  # 테스트 데이터로 모델을 평가함 -> 최종 손실값과 정확도가 계산됨

print("테스트 손실값:", test_loss)  # 테스트 손실값을 출력함 -> 테스트 데이터에서의 오차를 확인할 수 있음
print("테스트 정확도:", test_accuracy)  # 테스트 정확도를 출력함 -> 최종 분류 성능을 확인할 수 있음

# ===================== 이미지 예측 =====================
base_path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()  # 현재 파이썬 파일 기준 폴더를 구함 -> 실행 위치와 무관하게 파일을 찾을 수 있음
img_path = base_path / "images" / "dog.jpg"  # images/dog.jpg 경로를 지정함 -> VSCode 프로젝트 폴더 기준으로 이미지를 찾게 됨

if img_path.exists():  # dog.jpg 파일 존재 여부를 확인함 -> 파일이 있을 때만 예측을 진행함
    img_data = tf.io.read_file(str(img_path))  # 이미지 파일을 읽음 -> 원본 바이트 데이터를 불러옴
    img = tf.image.decode_jpeg(img_data, channels=3)  # JPEG 이미지를 텐서로 변환함 -> 컬러 이미지 배열이 만들어짐
    img = tf.image.resize(img, [32, 32])  # CIFAR-10 입력 크기에 맞게 조정함 -> 32x32 크기로 변환됨
    img = tf.cast(img, tf.float32) / 255.0  # 예측용 이미지도 정규화함 -> 훈련 데이터와 같은 범위로 맞춤
    img = tf.expand_dims(img, axis=0)  # 배치 차원을 추가함 -> 모델 입력 형식에 맞게 4차원으로 바뀜

    predictions = model.predict(img, verbose=0)  # dog.jpg의 클래스를 예측함 -> 10개 클래스 확률이 계산됨
    predicted_index = int(np.argmax(predictions[0]))  # 가장 확률이 높은 클래스 번호를 찾음 -> 최종 예측 인덱스가 결정됨
    predicted_class = class_names[predicted_index]  # 클래스 번호를 이름으로 바꿈 -> 사람이 읽기 쉬운 결과가 됨
    predicted_prob = float(predictions[0][predicted_index])  # 해당 클래스의 확률값을 저장함 -> 모델의 확신 정도를 확인할 수 있음

    print("dog.jpg 예측 클래스:", predicted_class)  # 예측된 클래스 이름을 출력함 -> 예를 들어 dog, cat 등이 표시됨
    print("dog.jpg 예측 확률:", predicted_prob)  # 예측 확률을 출력함 -> 해당 결과의 신뢰도를 볼 수 있음
else:  # dog.jpg가 없는 경우를 처리함 -> 예측 대신 안내 문구를 출력함
    print(f"파일을 찾을 수 없습니다: {img_path}")  # 실제 찾은 경로를 출력함 -> 경로 문제를 바로 확인할 수 있음