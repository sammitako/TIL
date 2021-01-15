# Column Indexing

## 특정 column 추출하면 Series로 받음

```python
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore') # 오류 끔
warnings.filterwarnings(action='default') # 오류 켬

data = {'Name': ['Sam', 'Sarah', 'Bob', 'Chris', 'John'],
        'Dept': ['CS', 'Math', 'Law', 'Lit', 'Stats'],
        'Year': [1, 2, 2, 4, 3],
        'GPA': [1.3, 3.5, 2.7, 4.3, 4.5]}

df = pd.DataFrame(data, 
                  columns=['Dept', 'Name', 'Year', 'GPA'],
                  index=['one', 'two', 'three', 'four','five'])

df['Name'] # Series
std_name = df['Name'] # view 생성
std_name['three'] = 'David' # 원본에 적용됨
print(std_name)
print(df)

std_name = df['Name'].copy() # 별도의 객체 생성
std_name['three'] = 'David' # 원본에 적용 안됨
print(std_name)
print(df)
```

## Fancy Indexing - 복수의 columns 추출

```python
# fancy indexing
df[['Dept', 'Name']] # DataFrame, view
```

## 특정값 수정

> 결측치값 항상 주의

```python
# 특정 값 수정
data = {'Name': ['Sam', 'Sarah', 'Bob', 'Chris', 'John'],
        'Dept': ['CS', 'Math', 'Law', 'Lit', 'Stats'],
        'Year': [1, 2, 2, 4, 3],
        'GPA': [1.3, 3.5, 2.7, 4.3, 4.5]}

df = pd.DataFrame(data, 
                  columns=['Dept', 'Name', 'Year', 'GPA', 'Class'],
                  index=['one', 'two', 'three', 'four','five'])

display(df)
df['Class'] = 'A' # 브로드캐스팅
df['Class'] = ['A', 'C', 'B', 'A', 'C'] # list => ndarray (자동)
df['Class'] = nd.array(['A', 'C', 'B', 'A', 'C']) # ndarray

# 결측치값 설정 (사이즈가 다를 경우, 사이즈를 맞춰서 설정해야 하므로 꼭 써야함)
df['Class'] = ['A', 'C', 'B', 'A', np.nan]
display(df.to_numpy()) # ndarray -> nan
display(df) # DataFram -> NaN
```

## Column 추가

**[주의]**

- ndarray, list는 갯수만 맞추면 DataFrame에 column 추가 시 오류 안남
- Series는 **인덱스 기반**으로 Column이 추가 되므로, 기존 DataFrame와 인덱스만 맞추면 column 추가 시 오류 안남
- Series는 원소들 모두 같은 데이터 타입을 가져야 하므로 결측치가 있을 경우 결측치를 실수로 간주함

```python
data = {'Name': ['Sam', 'Sarah', 'Bob', 'Chris', 'John'],
        'Dept': ['CS', 'Math', 'Law', 'Lit', 'Stats'],
        'Year': [1, 2, 2, 4, 3],
        'GPA': [1.3, 3.5, 2.7, 4.3, 4.5]}

df = pd.DataFrame(data, 
                  columns=['Dept', 'Name', 'Year', 'GPA'],
                  index=['one', 'two', 'three', 'four','five'])
display(df)

## 갯수 맞춰야됨
# ndarray로 추가
df['Class'] = np.arange(1,10,2) # 5개의 값을 이용
display(df)

# list로 추가
df['Age'] = [14,20,24,30,38]
df['Age'] = [14,20,24,30] # Value Error: 행의 개수가 맞지 않아서 오류남

# Series로 추가
## 인덱스만 맞추면 됨
df['Age'] = pd.Series([15,20,25,30,35]) # 오류 -> 인덱스가 다름
df['Age'] = pd.Series([15,20,25,30,35],
                      index=['one', 'two', 'three', 'four','five'])
display(df)

# 따라서 인덱스만 맞춰서 넣으면 오류 안남
# 특히 결측치는 NaN으로 표시되고, Series의 특성상
# 모든 원소의 타입은 같아야 하므로 아래의 경우 정수가 실수로 표현됨
df['Age'] = pd.Series([15,20,25],
                      index=['one', 'three','five']) # NaN으로 처리됨
display(df)
```

## 기존 column에 특정 연산을 통해서 새로운 column을 추가

```python
data = {'Name': ['Sam', 'Sarah', 'Bob', 'Chris', 'John'],
        'Dept': ['CS', 'Math', 'Law', 'Lit', 'Stats'],
        'Year': [1, 2, 2, 4, 3],
        'GPA': [1.3, 3.5, 2.7, 4.3, 4.5]}

df = pd.DataFrame(data, 
                  columns=['Dept', 'Name', 'Year', 'GPA'],
                  index=['one', 'two', 'three', 'four','five'])

# 연산을 통한 cloumn 추가
df['Scholarship'] = df['GPA'] > 4.0 # broadcasting
display(df)
```

## DataFrame의 record와 column 삭제

> 원본이 변하는 지, 처리된 복사본이 리턴되는 지 확인 필요

```python
data = {'Name': ['Sam', 'Sarah', 'Bob', 'Chris', 'John'],
        'Dept': ['CS', 'Math', 'Law', 'Lit', 'Stats'],
        'Year': [1, 2, 2, 4, 3],
        'GPA': [1.3, 3.5, 2.7, 4.3, 4.5]}

df = pd.DataFrame(data, 
                  columns=['Dept', 'Name', 'Year', 'GPA'],
                  index=['one', 'two', 'three', 'four','five'])

## 데이터 처리에서 default값은 '원본 보존'
# 데이터를 삭제하는 경우, 원본에서 삭제하는 경우
df.drop('Dept', axis=1, inplace=True) # column 삭제
display(df) # 원본 변경

# 데이터를 삭제하는 경우, 원본은 보존하고 삭제처리된 복사본이 생성 (변수로 받아줘야 함)
new_df = df.drop('Dept', axis=1, inplace=False) # record 삭제
display(new_df) # 복사본 변경
display(df) # 원본 변경 안됨
```

# Record Indexing

**[column indexing]**

- 컬럼명으로 인덱싱 가능
- Fancy Indexing 가능
- 슬라이싱 불가능

😵 **[record indexing]**

- 행에 대한 *숫자 인덱스로* 단일 인덱싱 *불가능*
- 행에 대한 *숫자 및 인덱스로* 행을 Fancy indexing은 *불가능*
- 행에 대한 숫자 및 인덱스로 슬라이싱(view) 가능, 그러나 인덱스로 슬라이싱 하면 뒷 부분 포함됨
- **Boolean Indexing** 가능

```python
# record
data = {'Name': ['Sam', 'Sarah', 'Bob', 'Chris', 'John'],
        'Dept': ['CS', 'Math', 'Law', 'Lit', 'Stats'],
        'Year': [1, 2, 2, 4, 3],
        'GPA': [1.3, 3.5, 2.7, 4.3, 4.5]}

df = pd.DataFrame(data, 
                  columns=['Dept', 'Name', 'Year', 'GPA'],
                  index=['one', 'two', 'three', 'four','five'])

# column indexing
print(df['Name']) # Series
print(df['Name':'Year']) # 오류: column은 슬라이싱이 안됨
display(df[['Name', 'Year']]) # Fancy Indexing은 허용

# Boolean Indexing은 무조건 행 인덱싱할때만 사용됨

# 헷갈리는 row indexing
# 숫자
print(df[1]) # 에러 -> 행에 대한 숫자 인덱스로 단일 인덱싱이 안됨
print(df[1:3]) # 행에 대한 숫자 인덱스로 슬라이싱은 가능 -> view
display(df[[1,3]]) # record에 대한 Fancy indexing -> Error

# 인덱스
print(df['two']) # 오류 -> 컬럼 인덱싱(추출) 표현
display(df['two':'four'])
display(df["two":-1]) # 에러 -> 숫자 인덱스와 일반 인덱스를 혼용해서 사용할 수 없음
display(df[['one', 'three']]) # 에러 -> column에 대한 Fancy Indexing
```

-----

## 올바른 record indexing 방법

- `loc[]`: 사용자가 지정한 또는 DataFrame의 인덱스에 대한 record indexing (숫자 인덱스 금지)
- `iloc[]`: 내부 인덱스(default index)에 대한 record indexing (숫자 인덱스만 허용)

```python
# 올바른 row indexing
data = {'Name': ['Sam', 'Sarah', 'Bob', 'Chris', 'John'],
        'Dept': ['CS', 'Math', 'Law', 'Lit', 'Stats'],
        'Year': [1, 2, 2, 4, 3],
        'GPA': [1.3, 3.5, 2.7, 4.3, 4.5]}

df = pd.DataFrame(data, 
                  columns=['Dept', 'Name', 'Year', 'GPA'],
                  index=['one', 'two', 'three', 'four','five'])

# loc[]
display(df.loc['one']) # 단일 레코드 추출 -> Series (컬럼명이 인덱스가 됨)
display(df.loc['one':'three']) # 복수 레코드 추출 -> DataFrame (뒷부분 포함)
display(df.loc[['one','four']]) # Fancy Indexing

# iloc[]
display(df.iloc[1])
display(df.iloc[1:3]) # 뒷부분 포함 안됨
display(df.iloc[[0,3]])
```

# 🤝 컬럼과 행 쪼인해서 추출 (default: 행)

> 나만의 행, 컬럼 제어 규칙 만들기

```python
display(df.loc['one':'three']) # 행
display(df.loc['one':'three'], df[['Year','Name']]) # 행, 컬럼
display(df.loc['one':'three'], df['Dept':'Year']) # 오류 -> 컬럼에 대한 슬라이싱 지원 안함
display(df.loc['one':'three', 'Dept':'Year']) # loc 이용하면 됨
```

## Boolean Indexing (record)

```python
# 학점이 4.0을 초과하는 학생의 이름과 학점을 DataFrame으로 출력
df['GPA'] > 4.0 # boolean mask
display(df.loc[df['GPA'] > 4.0,['Name', 'GPA']]) # 행, 열
```



## loc을 이용한 row 추가 및 변경

```python
data = {'Name': ['Sam', 'Sarah', 'Bob', 'Chris', 'John'],
        'Dept': ['CS', 'Math', 'Law', 'Lit', 'Stats'],
        'Year': [1, 2, 2, 4, 3],
        'GPA': [1.3, 3.5, 2.7, 4.3, 4.5]}

df = pd.DataFrame(data, 
                  columns=['Dept', 'Name', 'Year', 'GPA'],
                  index=['one', 'two', 'three', 'four','five'])

df.loc['six', :] = ['Business', 'Maria', 3, 3.7] # 추가
display(df)

df.loc['five', :] = ['Business', 'Maria', 3, 3.7] # 변경
display(df)

df.loc['seven', 'Name':'GPA'] = ['Gio', 3, 4.5] # 인덱스만 잘 맞추면 잘 들어가고 그 외는 NaN
```

## row 삭제

**[default 값]**

- `axis=0`
- `inplace=False`

```python
# column 삭제
df.drop('Year', axis=1, inplace=True)
display(df)

# row 삭제
df.drop('four', axis=0, inplace=True)
display(df)

# Fancy indexing 가능
df.drop(['one', 'three'], axis=0, inplace=True) 
display(df)

# 슬라이싱은 불가
df.drop('one':'three', axis=0, inplace=True) # 오류남
```

# 연습문제

> 1. 이름이 박동훈인 사람을 찾아 이름과 학점을 DataFrame으로 출력 
> 2. 학점이 (1.5, 2.5)인 사람을 찾아 학과, 이름, 학점을 DataFrame으로 출력
> 3. 학점이 3.0을 초과하는 사람을 찾아 등급을 'A'로 설정

```python
import numpy as np
import pandas as pd

data = {'Name': ['이지은', '박동훈', '홍길동', '강감찬', '오혜영'],
        'Dept': ['CS', 'Math', 'Stats', 'Lit', 'Stats'],
        'Year': [1, 2, 2, 4, 3],
        'GPA': [1.5, 2.0, 3.1, 1.1, 2.3]}

df = pd.DataFrame(data, 
                  columns=['Dept', 'Name', 'Year', 'GPA', 'Class'],
                  index=['one', 'two', 'three', 'four','five'])
display(df)

# 1. 이름이 박동훈인 사람을 찾아 이름과 학점을 DataFrame으로 출력

display(df.loc[df['Name'] == '박동훈', 'Name':'GPA'])

# 2. 학점이 (1.5, 2.5)인 사람을 찾아 학과, 이름, 학점을 DataFrame으로 출력
display(df.loc[(df['GPA'] > 1.5) & (df['GPA'] < 2.5) ,'Dept':'GPA'])

# 3. 학점이 3.0을 초과하는 사람을 찾아 등급을 'A'로 설정
df.loc[df['GPA'] > 3.0, 'Class']  = 'A'
display(df)
```