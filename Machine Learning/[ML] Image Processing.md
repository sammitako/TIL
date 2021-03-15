# 이미지 처리

모델에 입력할 이미지를 모델이 예측할 수 있는 형태로 바꾸어주는 데이터(이미지) 전처리가 필요하다.

## **이미지 데이터의 특징**

- 이미지 데이터는 기본적으로 3차원이다. (3: 각 픽셀에 대해 RGB 값을 따로 저장)

- 흑백 이미지의 경우, 2차원으로도 표현이 가능하다. (가로, 세로, 1)

  1. RGB 각각의 값을 RGB의 평균으로 모두 동일하게 설정하여 흑백 이미지로 변경
2. `cv2.cvtColor`라는 OpenCV 함수를 사용하여 흑백 이미지로 변경

- 참고로, png 파일은 RGB 외 투명값을 나타내는 알파값이 존재하므로 4로 표현된다.

## **[이미지 처리를 위한 라이브러리 설치]**

- `conda install opencv`

  : OpenCV를 이용해서 컬러 이미지: 3차원 이미지를 흑백(Grayscale) 이미지: 2차원 형태로 바꿈

- `conda install Pillow`

  : 이미지 파일을 객체로 반환해줌 (Pillow가 제공해주는 함수로 객체 이미지에 대한 처리를 쉽게 할 수 있다.)

```python
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# 이미지 파일을 컬러로 읽어오기
my_img = cv2.imread('data/mnist/number.jpg', cv2.IMREAD_COLOR) 
print(my_img.shape) # (572, 406, 3)

# 컬러 이미지를 흑백 이미지(Grayscale)로 변경
# 즉, 3차원 데이터를 2차원 형태로 변경
im_grey = cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY) 
print(im_grey.shape) # (572, 406)

# 최종 이미지 파일을 생성 (2차원 흑백 이미지)
# MNIST의 입력 데이터
cv2.imwrite('data/mnist/number_grey.jpg', im_grey)

# 흑백 이미지 파일을 Matplotlib으로 출력해보자.
img = Image.open('data/mnist/number_grey.jpg') # JPG 객체 반환
plt.imshow(img, cmap='Greys') # 2차원 데이터에 색깔 정보가 없으므로 color map을 넣어야됨
plt.show()

# 사이즈 처리 (28x28)
pixel = np.array(img) # 이미지 객체의 픽셀 정보(2차원 ndarray)
print('이미지의 크기: {}'.format(pixel.size))
print('이미지의 형태: {}'.format(pixel.shape))

resize_img = img.resize((28,28)) # resize(tuple): Pillow 기능 중 하나로 이미지 처리가 쉬워짐
plt.imshow(resize_img, cmap='Greys') # 2차원 데이터에 색깔 정보가 없으므로 color map 넣어야됨
plt.show()

# 이미지 반전 처리 (바탕색: 흰색, 숫자: 검정색)
# 흰색: (0,0,0)을 검정색: (255,255,255)로 반전하기 위해서는 255에서 현재 RGB 값을 빼면 된다.
# 또는 cv2.bitwise_not(이미지 객체) 함수 사용
img_grey = 255 - np.array(resize_img)

# 픽셀 정규화 처리
x_img = img_grey.reshape(-1, 784)
x_img_norm = scaler.transform(x_img)

# Prediction
result = sess.run(predict, feed_dict={X: x_img_norm})
print('예측값: {}'.format(result))
```

------

Reference: [ML_0310](https://github.com/sammitako/TIL/blob/master/Machine%20Learning/source-code/ML_0310.ipynb)