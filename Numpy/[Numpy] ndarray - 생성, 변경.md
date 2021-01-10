# Numpy 라이브러리

> 수치 연산을 위한 파이썬 **모듈(Numerical python)**로, 행렬(matrix) 및 벡터(vector) 연산에 있어서 편리성을 제공하는 라이브러리

**[용어 정리]**

- <u>라이브러리(library)</u>: 여러 **모듈과 패키지**의 묶음
- <u>패키지(package)</u>
  - 특정 기능과 관련된 여러 **모듈들**을 하나의 상위 폴더에 넣어 놓은 것
  - 패키지 안에 여러가지 폴더가 더 존재할 수 있음
  - 패키지를 표현해주기 위해 `__init__.py` 가 존재해야 함
  - 파이썬 3.3부터는 없어도 되지만, 하위 버전 호환을 위해 `[init.py](<http://init.py>)` 파일을 생성하는 것이 안전함
- <u>모듈(module)</u>: 특정 기능들(함수, 변수, 클래스 등)이 구현되어있는 **파일(.py)**

# ndarray(n-dimensional array)

> Numpy 모듈의 기본적인 자료구조

## 환경 설정

```bash
conda activate data_env
conda install numpy
```

## [리뷰] 파이썬 리스트

- **콤마**로 원소 구분
- 리스트는 **클래스**임
- **중첩** 리스트 (파이썬의 리스트는 차원 개념이 없음)
- 리스트는 모든 원소가 같은 데이터 타입을 가지지 않아도 상관 없음

```bash
# 파이썬 리스트

a = [1, 2, 3, 4] # literal - 프로그램적 기호를 이용해서 표현
a = list() # list라는 클래스를 '명시적으로' 이용해서 객체 생성
print(type(a)) # <class 'list'>
print(a) # [1, 2, 3, 4]

# 중첩 리스트
my_list = [[1, 2, 3], [4, 5, 6]] # 리스트 중첩, 차원 개념이 없음
```

## numpy의 ndarray 생성

- **공백**으로 원소 구분

- ndarray도 **클래스**임

- ⭐ ndarray는 모든 원소가 **같은 데이터 타입**을 가져야 함 (✔️실수 사용)

- ndarray는 **차원** 개념이 존재함 (인덱스: 0부터 시작)

  **[사용 방법]**

  - `numpy.array(파이썬 리스트, 파이썬 리스트, ..., dtype=np.float64)`
  - `ndarray로 만든 변수 이름[행, 열]`

```python
# numpy의 ndarray

import numpy as np

b = np.array([1, 2, 3, 4])
print(b) # [1 2 3 4]
print(type(b)) # <class 'numpy.ndarray'>

# ndarray는 모든 원소가 같은 데이터 타입을 가져야 함
print(b.dtype) # data type, int64

b[0] # 1
print(type(b[0])) # <class 'numpy.int64'>

# 다차원 ndarray 

my_array = np.array([[1, 2, 3], [4, 5, 6]])
my_array[1, 1] # 5

# data type 지정
my_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
print(my_array)
```

## ndarray의 대표적인 속성

> ndarray: 클래스를 기반으로한 인스턴스

### ndim, shape

- ndim: 차원의 수를 나타냄
- shape: 차원과 요소의 개수를 **tuple** 형태로 표현함

```python
import numpy as np

#1차원
my_list = [1, 2, 3, 4]
arr = np.array(my_list)

print(arr.ndim)  # ndim 속성: 차원의 수를 나타냄, 1
print(arr.shape) # shape 속성: 차원과 요소의 개수를 tuple 형태로 표현함, (4, ) 
                 # 1차원 -> (요소의 개수, ) -> 괄호 안의 원소의 개수 = 차원의 개수

# 2차원
my_list = [[1, 2, 3], [4, 5 ,6]]
arr = np.array(my_list)
print(arr.ndim)  # 2
print(arr.shape) # (2, 3), (행의 개수, 열의 개수)

# 3차원
my_list = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
arr = np.array(my_list)
print(arr.ndim)  # 3
print(arr.shape) # (2, 2, 3)
```

## len, <u>size</u>

- len: 파이썬 함수이므로 차원에 상관 없이(기준이 1차원이므로) 1차원의 요소의 개수 리턴함
- **size**: Numpy의 ndarray의 전체 크기를 알기 위해서 이 속성을 사용해야 함 (모든 요소의 개수)

```python
import numpy as np

# 1차원
my_list = [1, 2, 3, 4]
arr = np.array(my_list)
print('shape: {}'.format(arr.shape)) # (4, )
print('len: {}'.format(len(arr))) # 4
print('size: {}'.format(arr.size)) # 4

# 2차원
my_list = [[1, 2, 3], [4, 5, 6]]
arr = np.array(my_list)
print('shape: {}'.format(arr.shape)) # (2, 3)
print('len: {}'.format(len(arr))) # 2
print('size: {}'.format(arr.size)) # 6
```

## 차원 이동

```python
import numpy as np

my_list = [1, 2, 3 ,4]
arr = np.array(my_list)
print(arr)

print(arr.shape) # (4, )

# shape 바꾸기 (차원 이동)
arr.shape = (2, 2) # shape을 변경 시, 직접적으로 shape의 속성을 변경하지 않음 (비추천)
                   # arr.reshape(2, 2)
```

## 데이터 타입 변경 (기본: np.float64)

> astype(): 데이터 타입 변경하는 메소드

```python
# astype() - 데이터 타입 변경
import numpy as np

arr = np.array([1.2, 2.3, 3.5, 4.1, 5.7])
arr = arr.astype(np.int32) # 소수점 이하 버림
print(arr)
```

## ndarry의 다양한 생성 함수

### 1. ndarray를 만드는 또 다른 방법

- `np.array([리스트])`: 파이썬 리스트가 numpy의 ndarry로 생성됨
- `np.zeros`: 특정 형태의 ndarray를 만들어서 내용을 0으로 채움 (인자: shape)
- `np.ones`: 원하는 shape을 만들어서 내용을 1로 채움
- `np.empty`: 공간을 초기화를 하지 않기 때문에 빠르게 원하는 공간만 설정 시 사용 (쓰레기 값이 들어감)
- `np.full`: 초기값을 원하는 값으로 채움

```python
# ndarray의 다양한 생성 함수 (zeros, ones, empty, full)

import numpy as np

arr = np.zeros((3, 4)) # shape -> tuple

arr = np.zeros((3, 4), dtype=np.int32)  
print(arr)

arr = np.ones((3, 4), dtype=np.int32)
print(arr)

arr = np.full((3, 4), 7)
print(arr)
```

### 2. ndarray를 만드는 또 다른 방법 - arange()

```python
# ndarray를 만드는 또 다른 방법 - arange()

# python range()
range(0, 10, 2) # 메모리 상에 의미를 가짐

# numpy arange()
arr = np.arange(0, 10, 2) # 메모리 상에 실제 값을 가짐
print(arr)
print(arr.reshape(2, 2)) # 2차원 행렬로 바꿈
```

### 3. ndarray를 만드는 또 다른 방법 - random 기반의 생성 방법(5)

그래프 모듈인 `matplotlib` 를 사용해서 데이터 분포 파악 가능

- `np.random.normal(평균, 표준편차, shape)`: **정규분포**에서 **실수**형태의 난수를 추출
- ✔️`np.random.rand(d0, d1, d2, d3, ...shape의 요소)`: 0이상 1미만의 **실수**를 **균등분포**로 난수를 추출
- `np.random.randn()`: **표준 정규분포**(평균: 0, 표준편차: 1)에서 **실수** 형태로 난수를 추출
- `np.random.randint(low, high, shape)`: **균등분포**로 **정수** 표본을 추출
- ✔️`np.random.random((shape))`: 0이상 1미만의 실수를 균등분포로 난수로 추출 (= #2번째)

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. np.random.normal()
my_mean = 50 # 평균
my_std = 2 # 표준편차
arr = np.random.normal(my_mean, my_std, (10000, )) # shape - 1차원
print(arr)

plt.hist(arr, bins=100) # 전체 영역을 100개의 영역으로 나눔
plt.show() # 추출된 난수값 분포를 보여줌

# 2. np.random.rand()
arr = np.random.rand(10000) # 10000개의 데이터를 1차원으로 추출
print(arr)
plt.hist(arr, bins=100) # 전체 영역을 100개의 영역으로 나눔
plt.show()

# 3. np.random.randn()
arr = np.random.randn(10000)
print(arr)
plt.hist(arr, bins=100) # 전체 영역을 100개의 영역으로 나눔
plt.show()

# 4. np.random.randint(low, high, shape)
arr = np.random.randint(10, 100, (10000))
print(arr)
plt.hist(arr, bins=100) # 전체 영역을 100개의 영역으로 나눔
plt.show() 

# 5. np.random.random((shape))
arr = np.random.random((10000))
print(arr)
plt.hist(arr, bins=100) # 전체 영역을 100개의 영역으로 나눔
plt.show()
```

### 4. random에 관련된 부가적인 함수

- `seed(수)`: 난수의 재현 - 실행할 때마다 같은 난수가 추출되도록 설정

  ```python
  # 같은 시드 값에 대해서 같은 난수가 추출됨
  np.random.seed(3)
  arr = np.random.randint(10, 100, (10, ))
  print(arr)
  ```

- `shuffle()`: ndarray 안에 있는 데이터의 순서를 임의적으로 변경

  ```python
  # 이미 만들어진 ndarray 데이터의 순서를 랜덤하게 셔플(섞음)
  arr = np.arange(10) # ndarray를 간단하게 만드는 방법
  print(arr) 
  np.random.shuffle(arr)
  print(arr)
  ```

## reshape()에 대한 심화 학습

### reshape()은 새로운 ndarray를 만드는 것이 아니라 view를 생성!

> reshape(): 생성된 ndarray의 형태(shape)를 제어

- 새로운 ndarray를 만들 지 않음

- **view를 생성해서 같은 데이터를 shape만 변경해서 보여줌**

  - **[정의]** view: 데이터를 보여주는 창문 (데이터 원본을 공유)

    → 결국 같은 메모리 공간을 공유하여 메모리를 절약함

- **따라서 reshape을 통해 데이터 변경 시, 원본도 변경되므로 유의해야 함**

```python
import numpy as np

# reshape()
arr = np.arange(12) # 12개의 요소를 가지는 1차원의 ndarray
print(arr)

# view 생성 - 데이터를 보여지게 하는 창문과 같은 역할 (같은 메모리 공간 공유)
arr1 = arr.reshape(3, 4) # 행부터 채움
print(arr1)

# 조심! 원본에 영향을 미침!
print(arr1[0, 2])
arr1[0, 2] = 200
print(arr1) # 200
print(arr) # 200
```

### 그러면 새로운 ndarray 만드려면?

> copy(): 원본의 내용을 복사해서 새로운 ndarray를 생성

```python
# reshape()은 새로운 ndarray를 만드는게 아니라, view를 만드는 작업
# 그러면 새로운 ndaray 만드려면?
arr = np.arange(12) # 12개의 요소를 가지는 1차원의 ndarray
print(arr)

arr1 = arr.reshape(3, 4).copy() # 원본의 내용을 복사해서 새로운 ndarray를 생성
print(arr1)
arr1[0, 0] = 100
print(arr1) # 100
print(arr) # 0
```

### reshape()에서 -1의 의미는?

> reshape( 행 , -1 ): 행 먼저 채우고 열(-1)을 나머지로 할당

```python
arr = np.arange(12) # 12개의 요소를 가지는 1차원의 ndarray
print(arr)

arr1 = arr.reshape(3, 4)
print(arr1)

# 2차원
arr1 = arr.reshape(-1, 2)
arr1 = arr.reshape(3, -1)

# 3차원
arr2 = arr.reshape(2, 2, 3) # 2면 2행 3열
arr2 = arr.reshape(2, 3, -1)
arr2 = arr.reshape(2, -1, -1) # 오류 발생
```

## reshape()과 유사한 함수

### ravel(): ndarray가 가지고 있는 모든 요소를 포함하는 1차원의 ndarray로 변경

> ravel()함수는 view를 리턴함

```python
# ravel()

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr)

arr1 = arr.ravel() # view
print(arr1)
```

### resize()

> 사이즈가 원본과 다를 경우, 0으로 채워지거나, 행 단위로 먼저 진행되므로 남는 데이터는 자동적으로 삭제됨

1. **원본.resize(): 결과를 리턴하지 않고 원본 자체를 바꿈**

   ```python
   # resize()
   arr = np.array([[1, 2, 3], [4, 5, 6]])
   print(arr)
   
   # Case 1 -> 결과를 리턴하지 않고 원본을 바꿈
   arr1 = arr.resize(1, 6) # 1행 6열
   print(arr1) # None
   print(arr) # [[1 2 3 4 5 6]]
   
   # 올바른 표현
   arr.resize(1, 6)
   
   # Case 3 -> 원본 변경
   arr.resize(3, 4) # 원본이 2행 3열인데???
                    # reshape은 안됨
       
   print(arr) # 부족한 부분은 0으로 채워짐
   
   arr.resize(1, 2)
   print(arr) # 행 단위로 먼저 진행되므로 남는 데이터는 버림
   ```

2. **numpy.resize(원본, (사이즈)): 원본은 불변, 복사본이 생성**

   ```python
   # Case 2 -> Numpy 자체의 기능 사용
   arr1 = np.resize(arr, (1, 6)) # 원본 불변, 복사본이 만들어짐
   print(arr1)
   ```

# 🔥 Numpy ndarray의 기본적인 함수 퀵 리뷰 

## 생성

- np.array(), np.ones(), np.zeros(), np.full(), np.empty()
- np.arange()
- np.random.normal(), np.random.rand(), np.random.rands(), np.random.random(), np.random.randint()

## 변경

- np.shape(), np.reshape(), np.reshape().copy()
- np.astype()
- np.ravel()
- np.resize()