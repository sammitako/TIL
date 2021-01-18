# DataFrame이 지원하는 함수들

- **기댓값**: 어떤 확률을 가진 사건을 무한히 반복했을 때, 얻을 수 있는 값의 평균으로 기대할 수 있는 값

- **편차**

  : 확률변수 X와 평균값의 차이 (편차의 합계: 0) → 데이터의 흩어진 정도를 나타낼 수 있는 값

  - 편차의 합계는 결국 0이기 때문에 데이터의 흩어진 정도를 수치화하기가 힘듦
  - 따라서 분산을 사용함

- **분산**: 편차의 제곱의 평균 (기대값으로부터 흩어져 있는 정도)

- **표준편차**: 분산의 제곱근

------

## 공분산(covariance)

: 두 개의 확률변수의 **상관 관계**를 보여주는 값 (두 확률변수 편차의 곱에 대한 평균으로 계산)

- **성질**
  - 양의 상관관계와 음의 상관관계만 알 수 있음 (방향성만 알 수 있음)
  - 서로 어느 정도의 영향을 있는 지 알수는 없음 (밀접함의 강도는 모름)

```python
# 공분산(covariance)

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr # 금융 데이터
from datetime import datetime

start = datetime(2019,1,1) # 2019-01-01 날짜 객체 생성
end = datetime(2019,12,31) # 2019-12-31 날짜 객체 생성

# YAHOO에서 제공하는 KOSPI 지수
df_KOSPI = pdr.DataReader('^KS11', 'yahoo', start, end)
display(df_KOSPI['Close']) # Series
display(df_KOSPI['Close'].values) # 데이터를 ndarray로 반환

# YAHOO에서 제공하는 삼성전자 지수
df_SE = pdr.DataReader('005930.KS', 'yahoo', start, end)
display(df_SE['Close']) # Series
display(df_SE['Close'].values) # 데이터를 ndarray로 반환

# numpy가 제공하는 함수를 이용해서 공분산을 계산
print(np.cov(df_KOSPI['Close'].values, df_SE['Close'].values))

# 결과 (양의 상관관계)
~~# 0행 0열: KOSPI에 대한 공분산 (KOSPI & KOSPI)~~
# 0행 1열: KOSTPI와 삼성전자의 공분산
# 1행 0열: 삼성전자와 KOSPI의 공분산
~~# 1행 1열: 삼성전자의 공분산 (삼성전자 & 삼성전자)~~

[[6.28958682e+03 9.46863621e+04]
 [9.46863621e+04 1.41592089e+07]]
```

## 상관계수(correlation coefficient)

: -1과 1사이의 실수값 (피어슨 상관계수) 으로 하나의 변수가 변할 때 다른 변수가 변화하는 정도

공분산과 다르게 두 대상의 밀접한 관계성을 알 수 있음

**[부록]** 상관관계(correlation): 두 대상이 서로 연관성이 있다고 추측되는 관계

그러나, 상관관계는 인과관계를 설명할 수는 없음 (인과관계 → 회귀분석 사용)

- 성질
  - 양수값 - 정적 상관관계 (방향)
  - 음수값 - 부적 상관관계 (방향)
  - 0 - 관련성이 없음
  - 절대값 1 - 관련성이 높음

```python
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr # 금융 데이터
from datetime import datetime

start = datetime(2018,1,1) # 2018-01-01 날짜 객체 생성
end = datetime(2018,12,31) # 2018-12-31 날짜 객체 생성

# YAHOO에서 제공하는 종목 지수
df_KOSPI = pdr.DataReader('^KS11', 'yahoo', start, end) # KOSPI
df_SE = pdr.DataReader('005930.KS', 'yahoo', start, end) # 삼성전자
df_PUSAN = pdr.DataReader('011390.KS', 'yahoo', start, end) # 부산산업(남북경헙)
df_LIG = pdr.DataReader('079550.KS', 'yahoo', start, end) #LIG넥스원(방위)

my_dict = {
    'KOSPI' : df_KOSPI['Close'],
    'SAMSUNG' : df_SE['Close'],
    'PUSAN' : df_PUSAN['Close'],
    'LIG_NEXONE' : df_LIG['Close']
}
df = pd.DataFrame(my_dict)
display(df)

display(df.corr()) # DataFrame이 가지고 있는 상관계수를 구하는 함수를 이용
```

- 테이블 확인

## 분석용 함수

- 연산 시, `NaN` 은 완전히 배제됨
- (예외) `sum()` 에서는 `NaN` 을 0으로 간주함

```python
import numpy as np
import pandas as pd

data = [[2, np.nan],
        [7, -3],
        [np.nan, np.nan],
        [1, -2]]

df = pd.DataFrame(data, columns=['one', 'two'],
                  index=['a', 'b', 'c', 'd'])
display(df)

# Series로 리턴
display(df.sum()) # axis 생략 -> axis=0, NaN 무시 -> skipna=True
display(df.sum(axis=1)) # NaN + NaN = 0 (예외)

# Column indexing
print(df['two'].sum()) # -5.0
print(df.loc['b'].sum()) # 4.0

# skipna=False
print(df.mean(axis=0,skipna=False)) # NaN은 연산이 안되므로 NaN을 연산하게 되면 NaN이됨
print(df.mean(axis=0,skipna=True))  # NaN을 배제해서 계산, 3.33, -2.5
```

## 결측치 처리 방법

- 행 자체 지움
- **평균값으로 대체**
- 머신러닝 기법을 이용해서 NaN의 값을 예측해서 대체

```python
# 평균으로 대체
df['one'] = df['one'].fillna(value=df['one'].mean()) # 원본 데이터 바꿈
display(df)
```

## 정렬

- `shuffle()`: 원본 자체를 섞음
- `permutation()`: 인덱스 값을 바꿀 때, 순서가 바뀐 복사본을 리턴함
- `reindex()`: 레코드 인덱스와 컬럼 인덱스 모두 변경하여 복사본을 리턴함

```python
import numpy as np
import pandas as pd

np.random.seed(1)
df = pd.DataFrame(np.random.randint(0, 10,(6,4)))
display(df)

df.columns = ['A', 'B', 'C', 'D']
df.index = pd.date_range('20200101', periods=6) # 6일 동안
display(df)

# shuffle(): 원본 데이터를 바꿈
arr = np.array([1,2,3,4])
np.random.shuffle(arr) # [1 3 4 2]
print(arr)
np.random.shuffle(df.index) # (오류) 인덱스 값을 바꿀 수는 없음 -> 다른 함수 사용

# permutation(): 원본은 변경하지 않고, 순서가 변환된 복사본을 리턴함
new_index = np.random.permutation(df.index)
display(new_index)

# index 다시 설정 - 레코드 인덱스와 컬럼 인덱스 모두 변경 가능
# 원본은 안바뀌고 복사본을 리턴함
df2 = df.reindex(index=new_index, columns=['B','A','D','C'])
display(df2)
```

### 정렬은 기본적으로 axis를 기준으로 정렬

```python
display(df2.sort_index(axis=0, ascending=True)) # 행단위, 오름차순
display(df2.sort_index(axis=1, ascending=True)) # 열단위, 오름차순

# 특정 column의 값으로 행을 정렬
df2.sort_values(by='B')
df2.sort_values(by=['B','A']) # B에 동률이 있을 경우, A로 2차 정렬
```

## 기타 함수들

> `unique()`, `value_counts()`, `isin()`

```python
import numpy as np
import pandas as pd

np.random.seed(1)
df = pd.DataFrame(np.random.randint(0,10,(6,4)))
df.columns = ['A','B','C','D']
df.index = pd.date_range('20200101', periods=6)

df['E'] = ['AA','BB','CC','CC','AA','CC']
display(df)

print(df['E'].unique()) # Series -> ndarray, ['AA' 'BB' 'CC']
df['E'].value_counts() # 값 빈도수 (각 value값들의 개수를 Series로 리턴)
df['E'].isin('AA','BB') # Boolean, 조건을 검색할 때 많이 이용함
```

------

### 👏🏼 잠깐만! Pandas 정리

- Series
- DataFrame
  - 생성 - CSV파일, 리스트, dict, Open API, Database
  - indexing (열, 행)
    - indexing, slicing, fancy indexing, boolean indexing
    - loc[ ]
  - 함수 - 수학적 집계함수, 정렬함수, 유틸리티 함수
  - 결합

------

# DataFrame 병합

## merge 함수

### ☺️ Database의 inner join과 비슷, 일치하는 값이 있는 것만 결합

```python
import numpy as np
import pandas as pd

data1 = {
    'Code' : [1,2,3,4],
    'Name' : ['Sam', 'Chris', 'John', 'Anna'],
    'Year' : [2, 4, 1, 3]
}

data2 = {
    'Code' : [1,2,4,5],
    'Dept' : ['CS', 'Math', 'Lit', 'Stats'],
    'GPA' : [3.5, 2.7, 4.0, 4.3]
}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

display(df1)
display(df2)

pd.merge(df1, df2, on='Code', how='inner') # 있는 것만 결합
```

### outer: 양쪽의 없는 값들도 모두 결합해서 NaN으로 처리 (=full outer)

```python
# outer 
pd.merge(df1, df2, on='Code', how='outer')

# left outer: 왼쪽에 있는 df1만 붙임
pd.merge(df1, df2, on='Code', how='left')

# right outer: 오른쪽에 있는 df2만 붙임
pd.merge(df1, df2, on='Code', how='right')
```

### 결합하려는 컬럼명이 다를 경우

> 결합 결과, 다른 컬럼명 모두 출력됨

```python
pd.merge(df1, df2, left_on='Code', right_on='std_Code',how='inner')
```

### 결합하려는 컬럼이 index로 사용된 경우

> 레코드의 인덱스 값이 변경됨 (순서대로 안될 수 있음)

```python
import numpy as np
import pandas as pd

data1 = {
    'Code' : [1,2,3,4],
    'Name' : ['Sam', 'Chris', 'John', 'Anna'],
    'Year' : [2, 4, 1, 3]
}

data2 = {
    'Dept' : ['CS', 'Math', 'Lit', 'Stats'],
    'GPA' : [3.5, 2.7, 4.0, 4.3]
}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2, index=[1,2,4,5]) # 학번이 index로 사용된 경우

display(df1)
display(df2)

result = pd.merge(df1, df2,
                  left_on='Code',
                  right_index=True,
                  how='inner') 
display(result)
print(result.loc[3]) # Anna
print(result.iloc[2]) # Anna

# 두 DataFrame 모두 학번이 index로 사용된 경우,
df1 = pd.DataFrame(data1, index=[1,2,3,5]) # 학번이 index로 사용된 경우
df2 = pd.DataFrame(data2, index=[1,2,4,5]) # 학번이 index로 사용된 경우

display(df1)
display(df2)

result = pd.merge(df1, df2,
                  left_index=True,
                  right_index=True,
                  how='inner') 
display(result)
```

## concatenate 함수

```python
import numpy as np
import pandas as pd

df1 = pd.DataFrame(np.arange(6).reshape(3,2),
                   index=['a','b','c'], columns=['one', 'two'])

df2 = pd.DataFrame(np.arange(4).reshape(2,2),
                   index=['a','b'], columns=['three', 'four'])
display(df1)
display(df2)

result = pd.concat([df1, df2],
                   axis=1)
display(result)

result = pd.concat([df1, df2],
                   axis=0)
display(result)

result = pd.concat([df1, df2],
                   axis=0,
                   sort=True) # 인덱스 정렬
display(result)

result = pd.concat([df1, df2],
                   axis=0,
                   ignore_index=True) # 동률 인덱스를 무시하고 새로 만듬
display(result)
```