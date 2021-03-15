## BMI 예제를 통한 Multinomial Classification 구현


- Test Data Set: 모델의 최종 Accuracy를 측정하기 위해 사용

- Train Data Set: K-fold Cross Validation을 이용해서 내부적인 평가를 진행

  <br>

- 상관분석

  종속변수에 영향을 미치지 않는 독립변수 제외

  - 1: 상관관계 있음

  - 0: 상관관계 없음

    <br>

## 데이터 전처리

  - 결측치

  - 이상치

  - **데이터 분리**

    - `train_test_split(피쳐들의 2차원 행렬, 레이블의 1차원 형태, test_size=나누는 비율, random_state=0)`
    - `random_state=0`: 전체 데이터를 위에서부터 차례대로 분리하지 않고 랜덤하게 데이터를 분리한다. seed의 개념과 동일하게 똑같은 형태로 분리되도록 설정 (난수의 재현성)

  - 정규화

  ```python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
import tensorflow as tf
  
from sklearn.preprocessing import MinMaxScaler
  from sklearn.model_selection import train_test_split
  
  df = pd.read_csv('bmi.csv', skiprows=3)
  display(df) # label: 0(저체중), 1(정상), 2(과체중)
  
  # 결측치 확인
  print(df.info())
  
  # 이상치 확인
  fig = plt.figure()
  fig_label = fig.add_subplot(1,3,1)
  fig_height = fig.add_subplot(1,3,2)
  fig_weight = fig.add_subplot(1,3,3)
  
  fig_label.set_title('label')
  fig_label.boxplot(df['label'])
  
  fig_height.set_title('height')
  fig_height.boxplot(df['height'])
  
  fig_weight.set_title('weight')
  fig_weight.boxplot(df['weight'])
  
  fig.tight_layout()
  plt.show()
  
  # 데이터 분리
  x_train, x_test, t_train, t_test = \\
  train_test_split(df[['height', 'weight']], df['label'], 
                   test_size=0.3, random_state=0)
  
  # 정규화
  scaler_x = MinMaxScaler()
  
  scaler_x.fit(x_train)
  x_train_norm = scaler_x.transform(x_train)
  
  scaler_x.fit(x_test)
  x_test_norm = scaler_x.transform(x_test)
  
  print(x_data_train_norm)
  
  # 안쓰는 변수 삭제
  del x_train
  del x_test
  
  # one-hot encoding
  sess = tf.Session()
  t_train_onehot = sess.run(tf.one_hot(t_train, depth=3))
  t_test_onehot = sess.run(tf.one_hot(t_test, depth=3))
  
  # 안쓰는 변수 삭제
  del t_train
  del t_test
  
  print(t_train_onehot)
  ```

<br>

## 모델 구현

```python
# X, T
X = tf.placeholder(shape=[None,2], dtype=tf.float32)
T = tf.placeholder(shape=[None,3], dtype=tf.float32)

# W, b
W = tf.Variable(tf.random.normal([2,3]), name='weight')
b = tf.Variable(tf.random.normal([3]), name='bias') # logistic regression 3개

# Hypothesis
logit = np.matmul(X, W) + b
H = tf.nn.softmax(logit)

# log loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=T))

# gradient descent algorithm
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
```

- 예전 학습 코드의 문제점

  - 시스템 마다 가용할 수 있는 메모리 공간이 있는데, `feed_dict`에 입력되는 피쳐 데이터는 모두 메모리에 저장된다.

  - 즉, 입력으로 들어오는 `x_train` 데이터의 양이 클 경우, tensorflow 그래프의 노드인 placeholder에 입력 데이터가 모두 들어오지 못하게 된다.

  - 따라서 예전 학습 코드에서 학습용 데이터(입력)가 클 경우, 메모리의 한계 때문에 런타임 오류가 나면서 프로그램이 종료된다.

    따라서 메모리의 한계를 방지하기 위해, 배치 프로세싱을 해야 한다.

해결 방안으로 학습과정에 **배치 처리** 도입

<br>

## 배치 처리를 통한 학습 함수 생성

```python
# parameter
num_of_epoch = 1000
batch_size = 100

def run_train(sess, train_x, train_t):
    print('=====START LEARNING=====')
    
    # 초기화
    sess.run(tf.global_variables_initializer())
    
    # 반복학습
    for step in range(num_of_epoch):
        # 1 epoch: 입력 데이터(train_x)를 100개씩 잘라서 총 140번 학습
        total_batch = int(train_x.shape[0] / batch_size) # shape: (14000, 2)
        
        for i in range(total_batch):
            # 100개씩: 0~99, 100~199, ...
            batch_x = train_x[i*batch_size:(i+1)*batch_size] 
            batch_t = train_t[i*batch_size:(i+1)*batch_size]
            _, loss_val = sess.run([train, loss], feed_dict={X: batch_x, T: batch_t})
            
        if step % 100 == 0:
            print('Loss: {}'.format(loss_val))
    
    print('=====FINISH LEARNING=====')
```

<br>

## Accuracy 측정

: Ture를 True로 맞추고, False를 False로 맞춘 비율

- 모델: `H = [0.1 0.3 0.6]` → 확률값

  (참고: `sigmoid=[0.8 0.3 0.4]`)

- 정답: `T = [0 0 1]` → One-hot Encoding값

- `tf.argmax(axis=1)`: 현재 가지고 있는 ndarray 중 가장 큰 값의 index를 알려줌

  - 예) 

    - H: `[0.1 0.3 0.6]` → 2
    - T: `[0 0 1]` → 2
  
- `cast(해당 데이터, dtype=tf.float32)`: 해당 데이터를 실수값으로 바꿔줌

```python
predict = tf.argmax(H, 1) # axis=1: 열방향으로 비교
correct = tf.equal(predict, tf.argmax(T, 1)) # True or False?
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

# learning
run_train(sess, x_train_norm, t_train_onehot)

# Training Data Set을 이용하여 성능평가 (X)
# Training Data Set으로 학습하고 Training Data Set으로 평가하면 당연히 성능이 좋게 나온다.
result = sess.run(accuracy, feed_dict={X: x_train_norm, T: t_train_onehot})
print('Accuracy: {}'.format(result))
```

<br>

## K-fold Cross-Validation

전체 데이터 셋의 양이 많을 경우, **Validation Data Set을 이용해서 내부적으로 모델을 수정한 후 성능평가 진행할 수 있다.**

그러나, **데이터 셋의 양이 적을 경우** Train Data Set을 두개로 나누지 않고 전체 데이터를 Train Data Set과 Test Data Set 2개로 분리해서 내부적으로 모델의 성능평가를 진행할 수 있다.

결국 k번의 학습과 k번의 Validation(내부 성능 평가)이 이뤄지기 때문에 시간이 k배가 걸리지만, 더 정확한 성능평가가 이뤄진다. 따라서 우리가 만든 모델이 잘 만들어졌는 지 확인하는 용도로 사용되는 것이다.

(참고: K-fold Cross-Validation에 사용되는 전체 데이터는 이미 앞에서 Training Set과 Testing Set으로 잘라놓은 Training Data Set을 의미한다.)

<br>

[**코드 이해]**

- `cv`: Fold의 수

- `results`: 각 Fold당 `cv`번의 학습과 성능평가가 진행되는데, 이 때 `cv`개의 fold 각각에 대한 Accuracy 결과를 저장하기 위한 용도

- `KFold(n_splits=cv, shuffle=True)`: (Fold의 개수, 전체 데이터를 섞어서 Validation Data를 랜덤하게 뽑음)


```python
# 총 5번의 학습과 5번의 평가가 이뤄짐
cv = 5      
results = [] 

# Split Train Data and Validation Data
kf = KFold(n_splits=cv, shuffle=True) 

# cv만큼 Train Data가 잘림 -> 한 폴드에 대해 train용 데이터의 인덱스, validation용 데이터의 인덱스 뽑힘
for training_idx, validation_idx in kf.split(x_train_norm): # x_train_norm: Kfold로 나누어진 row 인덱스값이 나옴
    
    training_x = x_train_norm[training_idx] # Fancy Indexing
    training_t = t_train_onehot[training_idx]
    
    validation_x = x_train_norm[validation_idx]
    validation_t = t_train_onehot[validation_idx]
    
    # 1. training data로 learning
    run_train(sess, training_x, training_t)
    
    # 2. testing data로 validation: #1에서 만든 학습모델에 대한 정확도(성능평가) 측정
    results.append(sess.run(accuracy, feed_dict={X:validation_x, T: validation_t}))
    
print('측정한 각 Fold의 정확도: {}'.format(results))

# 3. results 안 각각의 학습 결과에 대한 평가값(정확도)의 평균
final_acc = np.mean(results)
print('K-Fold Validation을 통한 우리 모델의 최종 Accuracy: {}'.format(final_acc))
```

<br>

## Testing

```python
run_train(sess, training_x, training_t)
final_accuracy = sess.run(accuracy, feed_dict={X: x_test_norm, T: t_test_onehot})
print('우리 모델의 최종 정확도: {}'.format(final_accuracy))
```

-----

Reference: [ML_0308](https://github.com/sammitako/TIL/blob/master/Machine%20Learning/source-code/ML_0308.ipynb)