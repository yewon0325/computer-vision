# 01. 간단한 이미지 분류기 구현

## 1) 제목

**MNIST 데이터셋을 이용한 간단한 이미지 분류기 구현**

---

## 2) 문제

손글씨 숫자 이미지 데이터셋인 **MNIST**를 이용하여 간단한 이미지 분류기를 구현한다.
주어진 숫자 이미지를 입력받아, 해당 이미지가 **0부터 9까지 어떤 숫자인지 분류**하는 신경망 모델을 작성하고 학습 및 평가한다.

---

## 3) 요구사항

* MNIST 데이터셋을 로드
* 데이터를 훈련 세트와 테스트 세트로 분할
* 간단한 신경망 모델 구축
* 모델을 훈련시키고 정확도를 평가

---

## 4) 개념

### 4-1. MNIST 데이터셋

MNIST는 손글씨 숫자 이미지로 이루어진 대표적인 이미지 분류 데이터셋이다.
각 이미지는 **28×28 크기의 흑백 이미지**이며, 정답 레이블은 **0~9 사이의 숫자**이다.

### 4-2. 정규화(Normalization)

이미지의 픽셀 값은 원래 0~255 범위이다.
이를 **255.0으로 나누어 0~1 범위**로 바꾸면 학습이 더 안정적으로 진행된다.

### 4-3. Sequential 모델

`Sequential`은 층을 순서대로 쌓는 가장 기본적인 신경망 모델이다.
이번 과제에서는 다음과 같은 구조를 사용한다.

* `Flatten`: 28×28 이미지를 1차원 벡터(784개 값)로 변환
* `Dense(128, activation='relu')`: 은닉층에서 특징 학습
* `Dense(10, activation='softmax')`: 10개의 숫자 클래스로 분류

### 4-4. 손실 함수와 정확도

* **loss = sparse_categorical_crossentropy**
  다중 클래스 분류 문제에서 사용되는 손실 함수이다.
* **metrics = ['accuracy']**
  전체 데이터 중 몇 개를 맞췄는지 정확도를 계산한다.

---

## 5) 전체 코드

```python
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
```

---

## 6) 핵심 코드

### 6-1. MNIST 데이터셋 불러오기

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

* MNIST 데이터셋을 불러온다.
* 훈련용 데이터와 테스트용 데이터가 각각 나누어져 저장된다.

### 6-2. 데이터 정규화

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

* 픽셀 값을 0~1 범위로 변환한다.
* 학습 속도와 안정성을 높일 수 있다.

### 6-3. 모델 구성

```python
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

* `Flatten`으로 2차원 이미지를 1차원으로 펼친다.
* `Dense(128)`에서 이미지 특징을 학습한다.
* 마지막 `Dense(10)`에서 10개 숫자 클래스를 분류한다.

### 6-4. 모델 학습

```python
model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)
```

* 훈련 데이터를 사용해 총 5번 반복 학습한다.
* 배치 크기는 32로 설정하여 학습한다.

### 6-5. 모델 평가

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
```

* 테스트 데이터를 이용해 최종 성능을 평가한다.
* 손실값과 정확도를 확인할 수 있다.

---

## 7) 실행 방법

### 7-1. 파일 저장

예시 파일명:

```bash
01_mnist_classifier.py
```

### 7-2. TensorFlow 설치

터미널에서 아래 명령어를 실행한다.

```bash
python -m pip install tensorflow
```

### 7-3. 프로그램 실행

터미널에서 아래 명령어를 실행한다.

```bash
python 01_mnist_classifier.py
```

---

## 8) 실행 결과

실행 결과, MNIST 데이터셋이 다운로드된 뒤 모델이 5 에포크 동안 학습되었고, 각 에포크의 정확도와 손실값이 출력되었다.

### 학습 과정

* Epoch 1/5: accuracy = **0.9270**, loss = **0.2547**
* Epoch 2/5: accuracy = **0.9675**, loss = **0.1107**
* Epoch 3/5: accuracy = **0.9772**, loss = **0.0763**
* Epoch 4/5: accuracy = **0.9836**, loss = **0.0555**
* Epoch 5/5: accuracy = **0.9868**, loss = **0.0433**

### 최종 테스트 결과

* 테스트 손실값: **0.0732131078839302**
* 테스트 정확도: **0.9764000177383423**

즉, 테스트 데이터에 대해 약 **97.64%의 정확도**를 얻었다.

> 실행 화면은 첨부한 결과 이미지 참고

<img src="./screenshots/01_result1.png" width="45%" alt="ROI, disparity, depth 시각화 결과">

---

## 9) 실행 결과 분석

이번 실습에서는 매우 간단한 신경망 구조를 사용했음에도 불구하고 **97% 이상의 높은 정확도**를 얻을 수 있었다.
이는 MNIST 데이터셋이 비교적 단순한 흑백 숫자 이미지로 이루어져 있고, 숫자 형태가 일정한 패턴을 가지기 때문이다.

에포크가 진행될수록

* **정확도는 0.9270 → 0.9868로 증가**
* **손실값은 0.2547 → 0.0433으로 감소**

하는 모습을 확인할 수 있었다.
즉, 모델이 반복 학습을 통해 숫자 이미지의 특징을 점점 더 잘 학습했다는 뜻이다.

또한 최종 테스트 정확도가 **0.9764**로 높게 나타났으므로, 훈련 데이터뿐 아니라 처음 보는 테스트 데이터에 대해서도 잘 분류한다고 볼 수 있다.

다만 실행 중 다음과 같은 메시지가 함께 출력되었다.

* `oneDNN custom operations are on`
* `Do not pass an input_shape / input_dim argument to a layer...`

이 메시지들은 **오류가 아니라 경고 또는 안내 메시지**이다.
프로그램 실행과 학습 결과에는 큰 문제가 없으며, 모델도 정상적으로 학습되었다.

이번 과제를 통해 다음 내용을 확인할 수 있었다.

* MNIST 데이터셋을 쉽게 불러와 사용할 수 있다.
* `Sequential`, `Flatten`, `Dense`만으로도 기본적인 이미지 분류기를 만들 수 있다.
* 데이터 정규화가 학습 안정성에 도움이 된다.
* 간단한 완전연결 신경망만으로도 높은 숫자 분류 정확도를 얻을 수 있다.

