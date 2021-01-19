# DataFrame 조작

## NaN 값 삭제

- `how='any'`: NaN가 하나라도 해당 행에 존재하면 해당 행 자체를 삭제
- `how='all'`: 모든 컬럼의 값이 NaN인 경우 행을 삭제
- `inplace=False`: default

```python
# DataFrame 조작
import numpy as np
import pandas as pd

# random값을 도출해서 DataFrame을 생성 => np.random.randint()

np.random.seed(1)
df = pd.DataFrame(np.random.randint(0,10,(6,4)))

df.index = pd.date_range('20200101', '20200106') # periods=6
df.columns = ['A','B','C','D']
df['E'] = [7,np.nan, 4, np.nan, 2, np.nan]
display(df)

# NaN은 데이터분석이나 머신러닝, 딥러닝 전에 반드시 처리해야하는 값
new_df = df.dropna(how='any') 
display(new_df)

# fillna()
new_df = df.fillna(value=0)
display(new_df)
```



## NaN 값 확인

```python
new_df = df.isna() # boolean mask 생성됨
display(new_df) 

# 'E' column의 값이 NaN인 행들을 찾아 해당행의 모든 column값을 출력
new_df = df[df['E'].isna() == True]
# df.loc[df['E'].isnull(), :]
display(new_df)
```



## 중복행 처리

- `duplicated()`: 각 행에 대한 중복여부를 찾아서 boolean mask로 반환
- `drop_duplicates()`: 중복된 행 삭제
- 컬럼을 새로 만들어서 중복행 처리 가능함
- 컬럼을 기준으로 중복행 처리 가능함

```python
# 중복행 처리
import numpy as np
import pandas as pd

my_dict = {
    'k1': ['one'] * 3  + ['two'] * 4, # list 연결
    'k2': [1,1,2,3,3,4,4]
}

df = pd.DataFrame(my_dict)
display(df)

print(df.duplicated()) 
df.loc[df.duplicated(), :] # True인 행만 찾음

new_df = df.drop_duplicates()
display(new_df)

# 새로운 컬럼 생성
df['k3'] = np.arange(7)
display(df)

# 특정 컬럼을 기준으로 중복 제거
display(df.drop_duplicates(['k1']))
```



## 값 대체

- `replace(변경할 값, 대체할 값)`

```python
import numpy as np
import pandas as pd
np.random.seed(1)
df = pd.DataFrame(np.random.randint(0,10,(6,4)))

df.index = pd.date_range('20200101', '20200106') 
df.columns = ['A','B','C','D']
df['E'] = [7,np.nan, 4, np.nan, 2, np.nan]

df.replace(np.nan, -100)
```



------

# Series, DataFrame의 그룹핑

## 기준 데이터

```python
import numpy as np
import pandas as pd

my_dict = {
    'Dept': ['CS', 'Business', 'CS', 'Business', 'CS'],
    'Year': [1,2,3,2,3],
    'Name': ['Sam', 'Chris', 'Bob', 'Lilly', 'Sarah'],
    'GPA': [1.5, 4.4, 3.7, 4.5, 4.2]
}
df = pd.DataFrame(my_dict)
display(df)
```

## Series Groupby

------

### 특정 column을 기준으로 그룹핑

- `get_group()`: 그룹핑 된 그룹 안의 데이터를 확인
- `size()`: 각 그룹 안에 몇 개의 데이터가 있는 지 Series 형태로 리턴
- 집계함수 사용

```python
import numpy as np
import pandas as pd

my_dict = {
    'Dept': ['CS', 'Business', 'CS', 'Business', 'CS'],
    'Year': [1,2,3,2,3],
    'Name': ['Sam', 'Chris', 'Bob', 'Lilly', 'Sarah'],
    'GPA': [1.5, 4.4, 3.7, 4.5, 4.2]
}
df = pd.DataFrame(my_dict)
display(df)

# Dept를 기준으로 Grouping
score = df['GPA'].groupby(df['Dept']) # 객체 생성
score.get_group('CS')
score.size() # Series

# 집계함수 이용
score.mean()

# Fancy Indexing -> DataFrame
score = df[['GPA', 'Name']].groupby(df['Dept'])
print(score.get_group('Business'))
score.mean() # 학점 평균만 계산됨(숫자)
```



### 2단계 그룹핑

```python
score = df['GPA'].groupby([df['Dept'], df['Year']]) # 객체 생성
score.mean() # Series(멀티인덱스)
score.mean().unstack()
```



## DataFrame Groupby

------

```python
score = df.groupby(df['Dept'])
score.get_group('Business')

score.mean() # 숫자는 무조건 평균 계산 -> 학년도 평균으로 계산됨
```

### 연습문제

- `size()`: NaN을 포함해서 요소를 카운트
- `count()`: NaN을 포함하지 않고 요소를 카운트
- `len()`: 리스트의 길이

```python
# 1. 학과별 평균학점?
df['GPA'].groupby(df['Dept']).mean()
df.groupby(df['Dept'])['GPA'].mean()

# 2. 학과별 몇명 존재?
df.groupby(df['Dept'])['Name'].count()
```

## 그룹핑 반복, for문

```python
import numpy as np
import pandas as pd

my_dict = {
    'Dept': ['CS', 'Business', 'CS', 'Business', 'CS'],
    'Year': [1,2,3,2,3],
    'Name': ['Sam', 'Chris', 'Bob', 'Lilly', 'Sarah'],
    'GPA': [1.5, 4.4, 3.7, 4.5, 4.2]
}
df = pd.DataFrame(my_dict)
display(df)

for dept, group in df.groupby(df['Dept']): # tuple(학과, 학과로 묶인 DataFrame)
    print(dept)
    print(group)
```

