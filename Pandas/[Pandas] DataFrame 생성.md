# DataFrame

## DataFrame 생성

### 1. dict를 이용해서 DataFrame을 생성

- DataFrame을 출력할 시, `display()` 를 사용하면 보기 좋타 🥰
- key → column명
- 여러개의 데이터가 하나의 레코드로 표현됨
- 사이즈가 다를 경우, `None`을 주어야 에러가 없음

```python
import numpy as np
import pandas as pd

my_dict = {'name': ['Sam', 'Bob', 'Lorie'],
           'year': [2015, 2014, 2017],
           'point': [3.5, None, 2.0]}

# dict로 Series 생성
s = pd.Series(my_dict)
print(s)

# dict로 DataFrame 생성
df = pd.DataFrame(my_dict)
display(df)

# 속성
print(df.shape) # (3,3)
print(df.size) # 데이터의 개수
print(df.index) # RangeIndex(start=0, stop=3, step=1)
print(df.columns) # Index(['name', 'year', 'point'], dtype='object')
print(df.values) # ndarray로 리턴

df.index.name= 'student number'
df.columns.name ='student info'
display(df)
```

-----

### 대용량의 외부 데이터를 사용(file, DB, Open API)하여 DataFrame을 생성

**[일반적으로 많이 사용되어지는 데이터 표현 방식 3가지]**

1. **CSV(Comma Separated Values)**

   - 장점: 데이터 사이즈를 작게 만들 수 있어서 많은 데이터를 표현하기에 적합함

   - 단점

     ￮   구조적 데이터 표현이 힘들어서 사용이 힘듦

     ￮   데이터 처리를 위해 따로 프로그램을 만들어야함

     ￮   데이터가 변경됬을 때 프로그램도 같이 변경되야 하므로 유지 보수 문제가 발생함

   따라서, 데이터 크기가 엄청 크고 데이터의 형태가 잘 변하지 않는 경우 가장 알맞은 형태

2. **XML(eXtended Markup Language)**

   - 장점: 데이터의 구성을 알기 쉽고 프로그램적 유지보수가 쉬움(parser 프로그램 사용)
   - 단점: 용량이 큼 (부가적인 데이터가 많음)

3. **JSON(Javascript Object Notation)**

   : 자바스크립트 객체표현방식을 이용해서 데이터를 표현하는 방식 (현재 일반적인 데이터 표현방식)

   - 장점: XML 장점 + 용량이 작음
   - 단점: CSV에 비해서는 용량이 큼 (부가적인 데이터가 많음)

------

### 2. CSV 파일을 이용해서 DataFrame 생성

 **[실습파일]** [movies.csv](../../강사파일/Pandas/movies.csv) 

```python
import numpy as np
import pandas as pd

df = pd.read_csv('./movies.csv')
display(df.head()) # 상위 5개의 행을 출력
print(df.shape) # (9742, 3)
```

## 3. 기존 Database로부터 데이터를 읽어서 DataFrame 생성

**[실습파일]** [_BookTableDump.sql](../../강사파일/Pandas/_BookTableDump.sql) 

**[환경설정] MySQL 5.6.50 버전 (단독실행파일)**

### Mysql - DBMS(Database Management System)

```bash
# Mac OS 실행
/usr/local/mysql/bin
mysql -u root -p

# 설정
create user data@localhost identified by "data";
create database library;
grant all privileges on library.* to data; 
grant all privileges on library.* to data@localhost;

flush privileges;
exit;

# 다운로드 받은 파일을 bin 폴더에 넣기
/usr/local/mysql/bin/mysql -udata -p library < _BookTableDump.sql
```

### 3-1. 기존 Database(MySQL)로부터 데이터를 읽어서 DataFrame 생성

**[환경설정]**

- [Toad Download (Toad Edge)](https://www.toadworld.com/downloads#mysql)
  - Server Host: localhost

  - Port: 3306

  - Database: library

  - Authentication: data/data

- pymysql 모듈 설치: `conda install pymysql`

```python
import pymysql
import numpy as np
import pandas as pd

# 데이터베이스 접속
# 만약 연결에 성공하면 연결 객체가 생성됨
conn = pymysql.connect(host='localhost',
                       user='data',
                       password='data',
                       db='library',
                       charset='utf8')

# 접속 성공 시 데이터를 가져오자.
# 책 제목에 특정 키워드가 들어가 있는 책들을 검색해서 해당 책의 isbn,제목,저자,가격 정보를 가져옴
keyword = '여행'

# 데이터베이스에서 데이터를 가져옴
sql = "SELECT bisbn, btitle, bauthor, bprice FROM book WHERE btitle LIKE '%{}%'".format(keyword)

# 외부 데이터를 이용하므로 python의 예외 처리 사용
try:
    df = pd.read_sql(sql, con=conn)
    display(df)
except Exception as err:
    print(err)
finally:
    conn.close()
```

### 3-2. [3-1]에서 가져온 데이터를 JSON 형태로 파일에 저장

------

DataFrame 형식을 JSON를 포함한 표준 데이터 표현 방식 CSV, XML 등과 같이 바꾸는 이유?

: 내가 가진 DataFrame(메모리에 있는 휘발성 데이터)의 내용을 표준 형태의 데이터 표현방식으로 변환시켜 파일로 저장하면 다른 컴퓨터(사람)과 그 데이터를 공유할 수 있기 때문!

> JSON Formatter 사용하여 JSON 구조 보기

- `orient=columns`: 컬럼명을 key값으로 이용해서 그 안에 또 다른 JSON을 생성하여 리턴
- `orient=records`: 한 행의 정보를 하나의 JSON으로 생성하여 JSON 배열 형태로 리턴

```python
# python으로 파일 처리 순서
# 1. 파일 오픈: f = open('test.txt')
# 2. 파일 처리: f.readline()
# 3. 파일 클로즈: f.close()

# orient=columns
with open('./data/books_orient_column.json', 'w', encoding='utf-8') as f1:
    df.to_json(f1, force_ascii=False, orient='columns') # JSON Formatter & Validator 사용해서 구조 파악

# orient=records
with open('./data/books_orient_records.json', 'w', encoding='utf-8') as f2:
    df.to_json(f2, force_ascii=False, orient='records')
```

### 3-3. [3-2]에서 만든 JSON 파일을 pandas의 DataFrame으로 생성

```python
import numpy as np
import pandas as pd
import json # 내장모듈

# 첫번째 파일
with open('./data/books_orient_column.json', 'r', encoding='utf-8') as f1:
    dict_book = json.load(f1) # JSON -> python의 dict로 변환

# python의 dict -> DataFrame
df = pd.DataFrame(dict_book)
display(df)

# 두번째 파일
with open('./data/books_orient_records.json', 'r', encoding='utf-8') as f2:
    dict_book = json.load(f2) # JSON -> list(내용: python의 dict)로 변환

# python의 list -> DataFrame
df = pd.DataFrame(dict_book)
display(df)
```

**여기까지가 Database, JSON, DataFrame 변환 🙈**



## 4. Open API를 이용해서 DataFrame 생성

### Open API 사용 방법

- [영화진흥위원회 오픈 api](https://www.kobis.or.kr/kobisopenapi/homepg/main/main.do)

- 키 발급받기

- REST 방식 요청 URL: http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json

- key(문자열): 발급받은키 값을 입력합니다.

- targetDt: 조회하고자 하는 날짜를 yyyymmdd 형식으로 입력합니다.

  **[GET 방식으로 실행해보자]**

  - 크롬 웹 스토어의 json formatter 다운로드

  - 결과URL을 웹 브라우저에 띄우기


### GET 방식?

------

Query String을 이용해서 호출

- Query String: 요청인자를 전달하기 위한 특별한 형식
- **[사용법]** `?변수1=값1&변수2=값2&...`
- REST 방식 요청 URL 뒤에 GET 방식으로 바로 붙여넣기

### JSON 데이터 핸들링

🔨 **[웹 살짝쿵]**

- request: url을 가지고 해당 웹 서버에 데이터를 요청하는 행위
- response: 웹 서버에서 결과를 리턴하는 행위

------

#### 1. (API →) JSON → dict

- `.read()`: 결과 객체 안에 들어있는 json을 얻어옴
- `json.loads()`: json → 파이썬 dict

```python
# Open API를 이용해서 DataFrame 생성

import numpy as np
import pandas as pd
import json
import urllib # 네트워크 관련 모듈

# Open API URL
open_api = '<http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json>'
query_string = '?key=2d8e579e31759d1dc24ccba95acb6553&targetDt=20210113'
open_api_url = open_api + query_string
print(open_api_url) # Success

# url을 프로그램 내에서 호출 -> request
page_obj = urllib.request.urlopen(open_api_url) # url을 열어주세요 (요청)

# request의 결과를 웹 서버에서 우리에게 전달하는 행위 -> response
print(type(page_obj)) # <class 'http.client.HTTPResponse'> # 결과로 객체를 돌려줌 (응답)

# 객체 안에 json이 포함됨 
json_page = json.loads(page_obj.read())
print(json_page)
print(type(json_page)) # <class 'dict'>
```

#### 2. JSON → DataFrame

> 우리가 가진 json 형태는 바로 DataFrame(2차원 구조)로 바꿀 수 없으므로, 일단 **json을 분석**해서 내가 원하는 2차원 구조로 변경. 그 다음 DataFrame으로 변경해야 함.

- 파이썬 dict 복습

```python
# 순위(rank), 영화제목(movieNm), 해당일 매출액(salesAmt)

mv_dict = dict() # {}, 빈 dict 생성
rank_list = list() # [], 빈 list 생성
title_list = list()
sales_list = list()

# list 생성
for tmp_dict in json_page['boxOfficeResult']['dailyBoxOfficeList']: # dict키값 -> dict value -> 키값: 리스트
    rank_list.append(tmp_dict['rank'])
    title_list.append(tmp_dict['movieNm'])
    sales_list.append(tmp_dict['salesAmt'])

# dict 생성
mv_dict['Rank'] = rank_list
mv_dict['Title'] = title_list
mv_dict['Sales'] = sales_list

# DataFrame 생성
df = pd.DataFrame(mv_dict)
display(df)
```

**[리뷰] DataFrame 생성 방법**

------

1. dict를 이용해서 직접 데이터를 입력하여 생성
2. CSV 파일을 이용해서 생성
3. Database에 있는 데이터를 이용해서 생성
4. Open API를 이용해서 생성

## DataFrame 생성 시, 플러스알파

```python
# dict를 이용해서 직접 데이터를 입력하여 DataFrame 생성

import numpy as np
import pandas as pd

data = {'Name': ['Sam', 'Sarah', 'Bob', 'Chris', 'John'],
        'Dept': ['CS', 'Math', 'Law', 'Lit', 'Stats'],
        'Year': [1, 2, 2, 4, 3],
        'GPA': [1.3, 3.5, 2.7, 4.3, 4.5]}

df = pd.DataFrame(data) # record, column => Series, Series 묶음 => DataFrame

# columns, index
df = pd.DataFrame(data, 
                  columns=['Dept', 'Name', 'Year', 'GPA'],
                  index=['one', 'two', 'three', 'four','five']) 
display(df)
```