# 인덱싱, 슬라이싱

**[부록]** `enumerate()` 함수: 인덱스와 값을 tuple 형태로 리턴함 (반복문 사용 시, index를 추출하기 위해 사용)

## 기본

> ndarray의 슬라이싱도 파이썬의 리스트와 동일한 형태로 사용 가능함

- **indexing**: 위치값으로 차원이 줄어듬
- **slicing**: 원본과 차원이 같으므로, 차원은 불변
- 슬라이싱과 인덱싱을 같이 쓸 수 있음

```python
arr = np.arange(0, 5, 1)
print(arr)
print(arr[0:2])

print(arr[0:-1]) # 맨 마지막 요소만 제외하고 슬라이싱
print(arr[1:4:2]) # 2칸 씩 이동하면서 슬라이싱

# 2차원
arr = np.array([[1, 2, 3, 4], 
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]])
print(arr)
print(arr[1, 1]) # 6
print(arr[1, 1:3]) # 슬라이싱 + 인덱싱 # [6 7]
print(arr[1:3, [1,2]]) # [[ 6  7]
                          [10 11]]

print(arr[0]) # [1 2 3 4]
```

## Boolean Indexing

> 조건을 가지고 원하는 데이터를 추출

**[사용방법]** `ndarray[조건]`

- boolean mask: 원본 ndarray와 shape이 같고, 그 요소값이 모두 boolean으로 되어 있음
- 이 boolean mask를 이용해서 indexing 하는 방식

```python
import numpy as np

np.random.seed(1)
arr = np.random.randint(1, 20, (10,)) 
print(arr) # [ 6 12 13  9 10 12  6 16  1 17]

print(arr%2 == 0) # [ True  True False False  True  True  True  True False False]
arr[arr%2 == 0] # array([ 6, 12, 10, 12,  6, 16])
```

## Fancy Indexing

> ndarray에 인덱스 배열을 리스트 형식으로 전달하여 배열 요소를 참조하는 방식

**[사용방법]** `ndarray[[인덱스 넘버]]`

```python
# 내가 원하는 위치만 지정해서 원하는 원소만 추출
# 인덱싱 안에 인덱스 넘버가 담겨진 리스트를 넘김

arr = np.array([1, 2, 3, 4, 5])
print(arr[[1, 3, 4]]) # [2 4 5]
```

## 🏀 연습

> ⚠️ 넘파이는 행과 열에 Fancy Indexing을 동시에 적용할 수 없으므로 함수: `np.ix_()` 사용

```python
arr = np.arange(0, 12, 1).reshape(3, 4).copy()
print(arr)

print(arr[2, 2]) # 10, indexing: 위치값 (차원 줄어듬)
print(arr[1:2, 2]) # [6], slicing: 원본 = 슬라이싱 결과 (차원 불변)
print(arr[1:2, 1:2]) # [[5]]

print(arr[[0,2], 2]) # [ 2 10]
print(arr[[0,2], 2:3]) # [[ 2]
#                        [10]]

# [[0 2]
#  [8 10]]
# print(arr[[0,2], [0,2]]) # 오류 발생

# 이런 경우를 위해 numpy가 함수를 제공 -> np.ix_()
print(arr[np.ix_([0,2], [0,2])])
```

## ndarray에 대한 연산

- **사칙 연산** 시, 같은 위치에 있는 원소끼리 연산을 수행함

  **[부록]** 파이썬 리스트에서 `+` 연산은 concatenation, 즉 리스트 연장을 의미

  ￮  shape이 다른 ndarray의 사칙연산 시, broadcasting을 지원

  ￮  **[주의] 열(뒤)**부터 비교해서 연산하므로, 열(뒤) 숫자가 안 맞을 경우, 오류 발생

- **행렬곱 연산** 시, (n×m), (m×k) 규칙 유념하기

  ￮  **[주의]** 행렬곱에서는 broadcasting이 발생하지 않음

```python
# ndarray에 대한 연산
import numpy as np

arr1 = np.array([[1,2,3],
                [4,5,6]]) # 2행 3열
arr2 = np.arange(10,16,1).reshape(2,3).copy() # 2행 3열을 새로 만듬
arr3 = np.arange(10,16,1).reshape(3,2).copy() # 3행 2열을 새로 만듬

# 같은 위치에 있는 원소끼리 계산 (사칙연산)
# 파이썬 리스트의 '+'는 cancatenation
print(arr1 + arr2) # [[11 13 15]
#                    [17 19 21]]

# broadcasting
arr1 = np.array([[1,2,3],
                 [4,5,6]]) # (2,3) ndarray
arr2 = np.array([7,8,9]) # (3,) ndarray
print(arr1 + arr2) # [[ 8 10 12]
 #                    [11 13 15]]

print(arr1 + 10) # 스칼라 [[11 12 13]
#                        [14 15 16]]

# 행렬곱연산
print(np.matmul(arr1, arr3))
```

## 전치행렬(transpose)

- **전치행렬**: 원본행렬의 행은 열로, 열은 행으로 바꾼 행렬을 의미

  ￮  view: 데이터를 원본과 공유함

  ￮  [**사용방법]** `ndarray.T`

- 1차원 벡터에 대해서는 전치행렬을 구할 수 없음

```python
# 전치행렬(transpose) -> view
import numpy as np

arr = np.array([[1,2,3],
                [4,5,6]])
print(arr)
print(arr.T)

# 1차원 벡터에 대해 전치행렬을 구하려면? -> 2차원으로 shape 변경 후 가능
```

## 반복자(iterator) = 지시자(포인터)

- **반복문**은 일반적으로 for문과 while문을 사용

  ￮  for - 반복하는 횟수를 알고 있거나, numpy ndarray를 반복 처리할 때 사용

  ￮  while - 조건에 따라서 반복 처리할 때 사용

- **반복자**: ndarray 각각의 방을 가리키고 있는 **화살표**를 의미함

  ￮  **[사용방법]** `numpy.nditer(ndarray, flags=['스타일'])`

  ￮  `flags`: 어떤 형태(스타일)로 움직이는 지 설정

  ⬩ 1차원 → `c_index`

  ⬩ 2차원 → `multi_index`

```python
## 1차원
# for문을 이용한 반복처리
arr = np.array([1,2,3,4,5])
for tmp in arr:
    print(tmp, end=' ')
    
# while문과 iterator를 이용해서 반복처리
arr = np.array([1,2,3,4,5])

# 반복자라는 객체 구하기
it = np.nditer(arr, flags=['c_index']) # c언어의 index 방식을 따름 
while not it.finished:
    idx = it.index # 0 (C에서 인덱스의 시작은 0)
    print(arr[idx], end=' ') # indexing
    it.iternext() # 화살표 옮기는 작업

## 2차원
## 행렬의 각 요소를 출력

# for문 사용
arr = np.array([[1,2,3],
                [4,5,6]])
print(arr)

for tmp1 in range(arr.shape[0]): # arr.shape => (2, 3)
    for tmp2 in range(arr.shape[1]):
        print(arr[tmp1, tmp2], end=' ')

# iterator 사용
it = np.nditer(arr, flags=['multi_index'])

while not it.finished:
    idx = it.multi_index # (행, 열)
    print(arr[idx], end=' ')
    it.iternext() # 화살표 옮기는 작업
```

## ndarray의 비교연산

- 비교연산도 사칙연산과 유사하게 동작 (같은 위치의 원소끼리 비교)
- boolean mask 대신 **비교함수** 사용

```python
import numpy as np

np.random.seed(4)
arr1 = np.random.randint(0,10,(2,3))
arr2 = np.random.randint(0,10,(2,3))

print(arr1)
print(arr2)
print(arr1 == arr2) # boolean mask

arr1 = np.arange(10)
arr2 = np.arange(10)
print(np.array_equal(arr1, arr2)) # 비교함수
```

## 집계함수, 수학함수 (계산 속도가 로직 처리보다 훨~씬! 빠름)

- 집계함수: 합, 평균, 표준편차, 분산, 중앙값
- 수학함수: 최대, 최소, 제곱근, 제곱값, 로그값
- ⭐ 특히 인덱스 리턴 함수가 중요함 → `numpy.argmax(ndarray)`, `numpy.argmin(ndarray)`

```python
import numpy as np

arr = np.arange(1,7,1).reshape(2,3).copy()

# 합
print(np.sum(arr)) # numpy 기능 사용
print(arr.sum()) # numpy ndarray의 함수 이용

# 평균
print(np.mean(arr)) 
print(arr.mean()) 

# 최대값
print(np.max(arr)) 
print(arr.max())

# 최소값
print(np.min(arr)) 
print(arr.min())

# 인덱스 리턴
print(np.argmax(arr)) # 최대값의 인덱스 리턴
print(np.argmin(arr)) # 최소값의 인덱스 리턴

# 그 외
print(np.std(arr)) # 표준편차(standard deviation)
print(np.sqrt(arr)) # 제곱근
```

## axis(축)

> Numpy의 모든 집계함수는 axis를 기준으로 계산함

- axis를 지정하지 않으면 `axis=None` 으로 설정되고, 대상 범위가 배열의 전체로 지정됨

- axis의 숫자의 의미는 변함

  ￮  2차원 → 0: 행방향, 1: 열방향

  ￮  3차원 → 0: 면, 1: 행방향, 2: 열방향

```python
# 1차원, axis 의미 없음
arr = np.array([1,2,3,4,5]) 
print(np.sum(arr, axis=0))

# 2차원
arr = np.array([[1,2,3],
                [4,5,6],
                [7,8,9],
                [10,11,12]])

print(arr.shape) # (4,3)
print(arr.sum()) # 78, 대상 -> 전체 ndarray
print(arr.sum(axis=0)) # 행방향, [22 26 30]
print(arr.sum(axis=1)) # 열방향, [ 6 15 24 33]

print(arr.argmax(axis=0)) # [3 3 3]
print(arr.argmax(axis=1)) # [2 2 2 2]
```

## 🍭 POP QUIZ

```python
# ndarray arr안에 10보다 큰 수가 몇 개 있는 지 알아보려면?

arr = np.array([[1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [13,14,15,16]])

print(len(arr[arr > 10])) # 6
print((arr > 10).sum()) # 6 -> True = 1
```

## 정렬

- `np.sort(ndarray)`: 인자로 들어가는 원본 ndarray는 변화가 없고, 정렬된 **복사본**이 만들어져서 리턴됨
- `ndarray.sort()`: 원본 배열을 정렬하고, 리턴이 없음
- `[::-1]`: 역순으로 정렬

```python
import numpy as np

arr = np.arange(10)
np.random.shuffle(arr)

print(np.sort(arr)) # [0 1 2 3 4 5 6 7 8 9]
print(arr) # 원본 불변, [3 8 6 4 5 7 0 1 2 9]

print(arr.sort()) # None, 원본 배열을 정렬함
print(arr) # [0 1 2 3 4 5 6 7 8 9]

# 특수한 슬라이싱을 이용하면 역순으로 정렬할 수 있음
print(np.sort(arr)[::-1]) # [9 8 7 6 5 4 3 2 1 0]
```

## ndarray 연결, numpy.concatenate( ) 함수 사용

```python
import numpy as np

arr = np.array([[1,2,3],
                [4,5,6]]) # (2,3)
new_row = np.array([7,8,9]) # (3,)

# 행방향으로 붙임
result = np.concatenate((arr, new_row.reshape(1, 3)), axis=0) 
print(result)

# 결과
[[1 2 3]
 [4 5 6]
 [7 8 9]]

# 열방향으로 붙임
new_col = np.array([7,8,9,10])
result = np.concatenate((arr, new_col.reshape(2,2)), axis=1)
print(result)

# 결과
[[ 1  2  3  7  8]
 [ 4  5  6  9 10]]
```

## numpy.delete( )

- axis를 기준으로 지움
- 만약 axis를 명시하지 않는다면, 자동으로 1차 배열로 변환이 된 후 삭제가 됨
- 2차원에서 `axis=-1` 일 경우, '열'을 지칭함

```python
import numpy as np

arr = np.array([[1,2,3],
                [4,5,6]]) # (2,3)

result = np.delete(arr, 1) # 1차 배열로 변환한 후 삭제
                           # [1 2 3 4 5 6] -> 1번 인덱스: 2 삭제됨
print(result) # [1 3 4 5 6]

result = np.delete(arr, 1, axis=0)
print(result) # [[1 2 3]]

result = np.delete(arr, 2, axis=1)
print(result) # [[1 2]
#               [4 5]]
```