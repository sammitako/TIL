# Pre-trained Network를 이용한 전이학습(Transfer Learning)

## **등장 배경**

단순한 형태의 흑백 이미지인 MNIST 예제에서 높은 정확도(99%)를 도출하기 위해서는 3개 이상의 Convolution layer와 1개 혹은 2개의 Pooling layer, 마지막으로 1개의 FC Layer를 구성했다.

만약 CPU를 이용하여 학습을 진행할 경우 1시간 이상이 소요가 되었다.

실무에서는 MNIST 처럼 단순한 이미지가 아닌 고해상도의 컬러 이미지가 대부분이기 때문에 최소 5개 이상의 Convolution layer와 Pooling layer가 필요하다. 또한 컬러 이미지 경우 FC layer 안의 Hidden layer도 1개 이상이 일반적으로 필요하다. 따라서 컬러 이미지를 처리하기 위해서는 상당히 많은 CNN 레이어 구조가 필요하게 된다.

결국 GPU 없이 CPU로 처리하게 될 경우, 학습 시간이 100시간은 쉽게 넘어간다. 즉, 학습에 상당히 오랜 시간이 소요된다. 더욱이 정확도가 좋지 않을 경우 hyper-parameter를 수정해서 다시 학습해야 되기 때문에 시간 소모가 굉장하다.

<br>

## **해결 방안 - Pre-trained Network**

이를 해결하기 위해, 기존에 이미 만들어진 네트워크를 활용해서 학습 시간을 대폭적으로 줄이게 된다. 결국 Pre-trained Network로 귀결이 되고 이를 **어떻게 활용하는 지에 따라** 결과가 다르게 도출된다.

이때 Pre-trained Network란, 특정 데이터를 이용해서 CNN 학습이 완료된 네트워크를 지칭한다.

1. VGG16, VGG19: 전통적인 모델
2. ResNet (Microsoft): 회사의 기술을 통해 만들어진 모델 Convolution layer 자체가 152층 이상으로 구성되어 있다.
3. Inception (Google): 회사의 기술을 통해 만들어진 모델로 Convolution layer 자체가 30층 이상으로 구성되어 있다.
4. MobileNet: 정확도 보다는 **수행 속도**에 강점이 있는 모델
5. **EfficientNet**: **B1~B6**까지 있으며 모델6으로 갈수록 레이어 수가 많아지기 때문에 정확도에 강점이 있는 모델이지만 **수행 속도가 느리다.** 학습 속도 대비 **높은 정확도**를 산출한다. (메모리 사용량이 많아서 무겁다.)

<br>

## **학습 데이터 -  ImageNet**

Training Data Set도 Pre-trained Network 처럼 여러 종류가 존재한다. 그 중 가장 대표적으로 ImageNet을 학습하여 모델을 완성시킨다. 이때 우리는 filter의 weight를 사용하는 것이다.

Pretrained Network에서 Classifier의 weight와 bias는 우리 이미지에 대한 분류와 다르기 때문에 사용하지 못한다. 따라서 이미지 특징 추출을 위해서는 Training Data Set (ImageNet)에서 이미 학습된 Convolution layer를 사용하고 이미지 분류 작업은 우리의 모델을 사용해야 한다.

따라서 전이학습이란, 이미 만들어진 CNN의 Filter Weight를 사용해서 이미지의 특징을 뽑아내고, 우리의 FC Layer를 통해 뽑아진 이미지 데이터를 학습하는 것이다.

결국 Convolution layer의 연산을 하지 않게 되므로 학습 시간이 대폭 단축되며 이미지 특징을 잘 뽑아내는 레이어를 이용하기 때문에 모델의 정확도가 높아진다.

마지막으로 우리 데이터가 아닌 다른 데이터로 만들어진 모델(기학습된 네트워크)을 가져와 사용해도 우리 이미지의 특징을 잘 뽑아내기 때문에 CNN에 유연성을 더해준다.

CNN 모델을 직접 만들어서 모델의 정확도를 높일 수 있지만 학습 시간이 많이 소요되기 때문에 학습 시간 대비 정확한 모델을 생성하는 전이학습으로 결국 귀결된다.

<br>

# Pretrained Network를 이용한 Feature Extraction 코드 구현

우리 데이터를 Pretrained Network의 특성을 추출하는 레이어(Convolution layer, Pooling layer) 안으로 입력하여 결과적으로 도출된 Activation Map을 우리의 Classifier인 FC Layer의 입력으로 사용하여 반복 학습을 통해 Weight와 bias를 갱신한다.



## **VGG16**

기학습된 모델을 통해 Classifier의 입력 데이터로 들어가는 Activation Map을 얻게된다.

- `weights`: 여러 개의 입력 데이터셋에 대해서 어떤 Training Data Set을 사용하는 지 명시
- `include_top`: FC Layer(Classifier)를 사용하지 않으므로 제외
- `input_shape`: Convolution layer의 입력으로 들어오는 input shape을 명시

```python
from tensorflow.keras.applications import VGG16

# 기학습된 모델
model_base = VGG16(weights='imagenet',
                   include_top=False,
                   input_shape=(150,150,3))

model_base.summary() # Activation Map(FC Layer의 입력 데이터): (None, 4, 4, 512)
```

개와 고양이 Training Data Set에 대한 Numpy Array 형태의 Activation Map 추출 (→ FC Layer 입력)

1. ImageDataGenerator을 사용하기 위한 폴더 경로 설정
2. ImageDataGenerator를 통한 이미지 정규화
3. 기학습된 모델을 적용하여 특징을 추출하는 함수 생성
   - 빈 Activation Map과 label 저장소 생성
   - 이미지 generator 생성 (이미지 폴더, 이미지 사이즈, 배치 사이즈, 분류 형태)
   - 특징을 추출하여 결과 이미지 데이터를 빈 저장소에 추가
   - generator가 무한 반복되지 않도록 break 후 함수 리턴
4. 함수를 실행하여 이미지 특징 추출 작업 시행

```python
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 경로 설정
base_dir = '/content/drive/MyDrive/ML Colab/data/CAT_DOG/cat_dog_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# ImageDataGenerator 생성 (데이터 증식 제외)
datagen = ImageDataGenerator(rescale=1/255)

# 배치 사이즈 설정 (ImageDataGenerator - 무한히 반복하여 데이터를 추출)
batch_size = 20

# 입력 이미지에 대해 VGG16을 적용하여 특징을 추출하는 함수 생성 (->Activation Map)
# 개와 고양이 폴더를 합친 총 데이터의 개수인 sample_count중에서 batch_size개씩을 이미지를 가져옴
def extract_feature(directory, sample_count): # 입력 데이터 폴더, 데이터 개수
    
    # 최종 결과물 features: Activation Map(Numpy Array)
		# 기학습된 모델에 배치 사이즈만큼 입력 데이터를 넣어서 나온 결과를 features에 추가해줄 것이다.
    features = np.zeros(shape=(sample_count, 4,4,512)) 

    # 분류값(1차원: feature 이미지 하나 당 값 1개로 표현)
    labels = np.zeros(shape=(sample_count,)) 

    # 특정 폴더의 이미지를 가져와서 generator 생성
    generator = datagen.flow_from_directory(
          directory, # 이미지 폴더
          target_size=(150,150)
          batch_size=batch_size # 1epoch 당 가져올 데이터의 개수
          class_mode='binary'
    )

    # 이미지로부터 픽셀 데이터와 레이블을 generator의 batch_size만큼 가져옴
    i = 0
    for x_data_batch, t_data_batch in generator:
      # Convolution layer, Pooling layer에 이미지 입력
      feature_batch = model_base.predict(x_data_batch) # 최종 예측값이 나와야하지만 Classifier를 제외했으므로 Activation Map이 나옴
      features[i*batch_size:(i+1)*batch_size] = feature_batch # 행단위로 추가됨
      labels[i*batch_size:(i+1)*batch_size] = t_data_batch

      i += 1
      
      if i*batch_size >= sample_count:
        break

    return features, labels

# Feature Extraction 실행
train_features, train_labels = extract_feature(train_dir, 2000)
validation_features, validation_labels = extract_feature(validation_dir, 1000)
```

Classifier를 직접 만들어서 최종 결과인 Activation Map을 FC Layer의 입력 데이터로 사용한다.

```python
# 2차원 데이터로 변경
train_features = np.reshape(train_features, (2000, 4*4*512))
validation_features = np.reshape(validation_features, (1000, 4*4*512))

# DNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 모델 생성
model = Sequential()

# FC Layer
model.add(Dense(256, activation='relu', input_shape=(4*4*512,)))

# Dropout
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

# Optimizer
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# learning
history = model.fit(train_features, train_labels,
                    epochs=30, batch_size=64, 
                    validation_data=(validation_features, validation_labels))
```

<br>

-----

Reference: [DL_0329_Feature_Extraction_(Pretrained_Network)](https://github.com/sammitako/TIL/blob/master/Deep%20Learning/source-code/DL_0329_Feature_Extraction_(Pretrained_Network).ipynb)