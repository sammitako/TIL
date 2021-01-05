# Data Type

## 1. Numeric

> int(정수), float(실수), complex(복소수)

- `type()`
- 몫: `//`, 지수표현: `**`

> ⚠️ 수치연산 시 무조건 같은 데이터 타입끼리 일어남

- 타입이 다를 경우, 내부적으로 캐스팅 되지만 안될 경우는 오류 발생
- python3 부터는 `int/int => float` 가능해짐



## 2. sequence (순서가 있는 데이터 타입)

### (1) **list: 임의의 데이터를 순서대로 저장하는 집합 자료형**

   - 빈 리스트 생성

     ```python
     a = list()
     a = []
     ```

   - concatenation → `+`

     ```python
     a = [1,2,3]
     b = [4,5,6]
     print(a + b)  # [1,2,3,4,5,6]
     ```

   #### [혼동되는 부분 정리]

   - list 수정 시,

     ```python
     a = [1,2,3]
     
     a[0] = 5   # [5,2,3]
     a[0:1] = [7,8,9]   # [7, 8, 9, 2, 3]
     
     a[0] = [7,8,9]       # [[7, 8, 9], 2, 3]
     ```

   - `sort()`: 리턴값이 없는 함수로, 원본을 제어함

     

### (2) **tuple: list와 유사하지만, 변경 불가한 READ ONLY**

   - 빈 튜플 생성

     ```python
     a = tuple()
     a = ()
     ```

   - 소괄호 생략 가능



### (3) **range(시작, 끝, 증가치)**

- [**range와 list의 차이점]**

  - `range`: 데이터의 의미를 가지고 있음

  - `list`: 메모리 상에 실제 데이터를 가짐

    ```python
    a = range(10) # 0부터 9까지 1씩 증가하는 sequence
    b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(a) # range(0, 10)
    ```

- **논리 연산자 in**

  ```python
  print(7 in range1) # True
  print(10 in range1) # False
  ```



### (4) Text Sequence type (문자열)

- 일반적으로 `' '` 을 사용함

- 자료형이 다른 변수를 수치 연산 시, 자동 캐스팅이 안됨

- 대소문자 구별함

  

## 3. Mapping (dict)

```python
# { key : value }

a = { 'name' : '홍길동', 'age' : 30, 'address' : '인천' }

# 지금 가지고 있는 dict에 새로운 내용을 추가
a['address'] = '서울'   #  key값이 존재하지 않으면 데이터를 추가
                       #  프로그래밍의 유연성측면에서는 좋음
                       #  논리오류에는 취약함
        
print(a.keys())  # key들만 뽑아서 dict_keys라는 데이터를 리턴
print(a.values())

for key in a.keys(): 
    print('key : {}, value : {}'.format(key, a[key]))
```



## 4. Set

> 중복을 배제, 순서가 없는 자료구조

```python
# set에 데이터 추가, 삭제
a = set()
a.add(7)   # 1개를 추가할때는 add
print(a)
a.update([9,2,3,4]) # 여러개를 추가할때는 update 
print(a)
a.difference_update({3,4})
```



## 5. Bool

- bool 논리형: True, False
- 사용할 수 있는 연산자: and, or, not

### [부록] Python에서 다음의 경우는 False 로 간주!

1. 빈문자열( '' )
2. 빈리스트( [] )
3. 빈Tuple( () )
4. 빈 dict( {} )
5. 숫자 0 (나머지 숫자는 True로 간주)
6. None