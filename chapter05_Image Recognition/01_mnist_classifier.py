import tensorflow as tf  # 텐서플로우를 사용하기 위해 불러옴 -> MNIST 데이터 로드, 모델 구성, 학습, 평가가 가능해짐
from tensorflow.keras.datasets import mnist  # MNIST 데이터셋을 사용하기 위해 불러옴 -> 손글씨 숫자 이미지 데이터를 바로 불러올 수 있음
from tensorflow.keras.models import Sequential  # 순차형 신경망 모델을 만들기 위해 불러옴 -> 층을 순서대로 쌓아 간단한 분류기를 만들 수 있음
from tensorflow.keras.layers import Flatten, Dense  # 입력 평탄화와 완전연결층을 사용하기 위해 불러옴 -> 이미지 데이터를 1차원으로 바꾸고 숫자를 분류할 수 있음

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # MNIST 데이터를 불러오기 위해 실행함 -> 훈련 세트와 테스트 세트가 각각 저장됨

x_train = x_train / 255.0  # 훈련 이미지의 픽셀 값을 정규화하기 위해 실행함 -> 0~255 값이 0~1 범위로 바뀌어 학습이 안정됨
x_test = x_test / 255.0  # 테스트 이미지의 픽셀 값을 정규화하기 위해 실행함 -> 평가할 때도 같은 기준의 입력값을 사용하게 됨

model = Sequential()  # 순차형 모델 객체를 만들기 위해 실행함 -> 신경망 층을 순서대로 추가할 준비가 됨
model.add(Flatten(input_shape=(28, 28)))  # 28x28 흑백 이미지를 1차원 벡터로 바꾸기 위해 실행함 -> 각 이미지가 784개의 입력값으로 변환됨
model.add(Dense(128, activation='relu'))  # 은닉층을 추가해 특징을 학습하기 위해 실행함 -> 이미지의 중요한 패턴을 학습할 수 있게 됨
model.add(Dense(10, activation='softmax'))  # 출력층을 추가해 0~9 숫자를 분류하기 위해 실행함 -> 각 숫자 클래스의 확률이 출력됨

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # 모델의 학습 방법과 평가 기준을 설정하기 위해 실행함 -> Adam으로 학습하고 정확도를 함께 확인할 수 있음

model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)  # 훈련 데이터를 사용해 모델을 학습시키기 위해 실행함 -> 에포크마다 손실과 정확도가 출력되며 분류 성능이 향상됨

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)  # 테스트 데이터로 모델 성능을 평가하기 위해 실행함 -> 최종 손실값과 테스트 정확도가 계산됨

print("테스트 손실값:", test_loss)  # 평가 결과의 손실값을 확인하기 위해 실행함 -> 테스트 데이터에서의 오차 크기가 출력됨
print("테스트 정확도:", test_accuracy)  # 평가 결과의 정확도를 확인하기 위해 실행함 -> 테스트 데이터에서 숫자를 맞춘 비율이 출력됨