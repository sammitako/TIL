# 🐼 Pandas

> Numpy를 기본으로 그 위에 1)Series, 2)DataFrame 이라는 자료구조를 정의해서 사용

### Series

------

: 동일한 데이터 타입의 복수개의 성분으로 구성 (ndarray + alpha)

- 1차원 자료구조
- 같은 데이터 타입

### DataFrame

------

: 여러개의 Series를 컬럼 방향으로 모아 놓은, 즉 테이블 형식으로 데이터를 저장하는 자료구조

### Pandas 설치

- Terminal(설치): `conda install pandas`
- Jupyter Notebook(체크): `import pandas as pd`

# Series (ndarray 1차원 벡터에 대한 확장판)

## 1. 리스트를 이용한 Series 생성

> pandas.Series([리스트])

- **<인덱스, 데이터 값>**  + 데이터 타입을 리턴함
- `dtype`: numpy의 데이터 타입을 사용
- `values`: numpy의 ndarray 타입으로 데이터 값을 리턴함
- `index`: 특수한 데이터 타입 형태로 리턴함 (예, RangeIndex)

```python
import numpy as np
import pandas as pd

# ndarray(dtype: np.float64)
arr = np.array([1,2,3,4,5], dtype=np.float64)
print(arr) # [1. 2. 3. 4. 5.]

arr1 = np.array([1, 3.14, True, 'Hello'])
print(arr1) # ['1' '3.14' 'True' 'Hello']
print(arr1.dtype) # <U32

arr2 = np.array([1, 3.14, True, 'Hello'], dtype=np.object)
print(arr2) # [1 3.14 True 'Hello']
print(arr2.dtype) # object

# Series
s = pd.Series([1,2,3,4,5], dtype=np.float64)

# 인덱스 데이터 값
print(s) # 0    1.0
#          1    2.0
#          2    3.0
#          3    4.0
#          4    5.0
#          dtype: float64

print('Series의 값: {}'.format(s.values)) # ndarray 리턴: [1. 2. 3. 4. 5.]
print('Series의 인덱스: {}'.format(s.index)) # RangeIndex(start=0, stop=5, step=1)
```

## Series의 인덱스 지정

- 인덱스가 변경되도 기본적으로 숫자 인덱스는 사용할 수 있음
- 인덱스가 중복되어도 중복된 인덱스의 값을 모두 Series 형태로 리턴함

```python
# Series의 index
import numpy as np
import pandas as pd

s = pd.Series([1,5,8,10], dtype=np.int32, index=['a','b','c','d'])
print(s)

# 5 값 출력?
print(s['b'])
print(s[1])

# 인덱스 중복
s = pd.Series([1,5,8,10], dtype=np.int32, index=['a','b','a','d'])
print(s['a']) # 중복된 값 모두 리턴
print(type(s['a'])) # <class 'pandas.core.series.Series'>

print(type(s['b'])) # <class 'numpy.int32'>
```

## Slicing

- ndarray의 슬라이싱을 그대로 적용
- 문자열 인덱스를 사용해서 슬라이싱하면 `[앞:뒤]` 앞, 뒤 모두 포함됨

```python
import numpy as np
import pandas as pd

s = pd.Series([1,5,8,10], dtype=np.int32, index=['a','b','c','d'])
print(s[0:3])
print(s['b':'d']) # 문자열 인덱싱 사용
```

## Fancy indexing, Boolean indexing

```python
import numpy as np
import pandas as pd
s = pd.Series([1,5,8,10], dtype=np.int32, index=['a','b','c','d'])

# Fancy indexing
print(s[[0,2]]) # Series 형태로 (a 1) (c 8) 리턴
print(s[['a','c']])

# Boolean indexing
print(s[s%2==0]) # 값 조건
```

## 집계함수

```python
import numpy as np
import pandas as pd
s = pd.Series([1,5,8,10], dtype=np.int32, index=['a','b','c','d'])

print(s.sum())
```

## 연습문제

- `pip3 install datetime`
- `from datetime import datetime`

> A 공장의 2020-01-01부터 10일간 생산량을 Series로 저장 생산량은 평균: 50, 표준편차: 5인 정규분포에서 랜덤하게 정수로 생성 형식) 2020-01-01 52 2020-01-12 49 2020-01-13 55 B 공장의 2020-01-01부터 10일간 생산량을 Series로 저장 생산량은 평균: 70, 표준편차: 8인 정규분포에서 랜덤하게 정수로 생성 형식) 2020-01-01 52 2020-01-12 49 2020-01-13 55 (문제) 날짜별로 모든(A공장, B공장)의 생산량의 합계?

- 힌트

  - list comprehension

  - 날짜연산 - 일반적으로 함수를 이용해서 일/월/년 단위로 증감 또는 주단위로 증감

    ￮   일단위 - `timedelta(days=n간격)`

    ￮   월/년 단위 - `relativedelta`

  - **Series의 사칙연산은 같은 인덱스를 기반으로 수행됨**

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(1)
start_day = datetime(2020,1,1)
print(start_day) # 2020-01-01 00:00:00

# A공장
factory_A = pd.Series([int(x) for x in np.random.normal(50, 5, (10,))],
                     dtype=np.int32, index=[start_day + timedelta(days=x)
                                            for x in range(10)])
                                           
print(factory_A)

# B공장
factory_B = pd.Series([int(x) for x in np.random.normal(70, 8, (10,))],
                     dtype=np.int32, index=[start_day + timedelta(days=x)
                                            for x in range(10)])
                                        
print(factory_B)

# 답
print(factory_A + factory_B) # Series의 덧셈은 같은 인덱스에 한해서 수행됨
```

## 인덱스가 다를 경우, Series의 연산?

- 인덱스가 맞지 않는 경우, 연산이 안되고 `NaN`을 리턴
- 그 외는 제대로 연산이 진행됨

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# A공장
np.random.seed(1)
start_day = datetime(2020,1,1)
print(start_day) # 2020-01-01 00:00:00

factory_A = pd.Series([int(x) for x in np.random.normal(50, 5, (10,))],
                     dtype=np.int32, index=[start_day + timedelta(days=x)
                                            for x in range(10)])
                                           
print(factory_A)

# B공장
np.random.seed(1)
start_day = datetime(2020,1,5)
print(start_day) # 2020-01-05 00:00:00

factory_B = pd.Series([int(x) for x in np.random.normal(70, 8, (10,))],
                     dtype=np.int32, index=[start_day + timedelta(days=x)
                                            for x in range(10)])
                                        
print(factory_B)

# 답
print(factory_A + factory_B) # NaN(Not a Number)

# 결과
2020-01-01      NaN
2020-01-02      NaN
2020-01-03      NaN
2020-01-04      NaN
2020-01-05    136.0
2020-01-06    103.0
2020-01-07    123.0
2020-01-08    107.0
2020-01-09    127.0
2020-01-10     99.0
2020-01-11      NaN
2020-01-12      NaN
2020-01-13      NaN
2020-01-14      NaN
dtype: float64
```

## 2. dict를 이용한 Series 생성

> pandas.Series({ key : value }), 이때 key 값이 인덱스가 됨

```python
import pandas as pd

my_dict = { '서울':1000, '인천':2000, '수원':3000 }
s = pd.Series(my_dict)
print(s)

s.name = '지역별 가격 데이터' # Series 자체에 이름 붙임
print(s) # Name: 지역별 가격 데이터, dtype: int64

# 인덱스를 리스트 타입 처럼 활용
print(s.index) # Index(['서울', '인천', '수원'], dtype='object')
s.index = ['Seoul', 'Incheon', 'Suwon']
s.index.name = 'Region' # 인덱스에 이름 붙임
print(s)
```