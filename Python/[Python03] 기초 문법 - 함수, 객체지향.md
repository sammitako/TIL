# 파이썬 이름 표기법 (관용적☺️)

- **파스칼 표기법(PascalCase)**: 클래스 이름
- **뱀 표기법(snake_case)**: 변수 및 함수의 이름

# 함수

> 특정 기능을 수행하는 코드 묶음, 객체

## 1. 사용자 정의 함수

- 파이썬은 내장함수의 재정의를 허용함 (유연하지만 책임은 모두 나의 것...😅)

### [Case 1] 함수를 정의 시, 인자의 개수가 가변적인 경우

```python
def my_sum(*args): # tuple로 인자를 받음
    print(args)
    result = 0
    for i in args:
        result += i
    return result
print('결과값: {}'.format(my_sum(1, 2, 3)))
```

### [Case 2] 여러 개의 값을 리턴하는 함수 (원래는 tuple 리턴 1개 안에 여러 값이 들어가 있음)

> tuple 은 ()를 생략할 수 있다!

```python
def multi_return(a, b):
    result1 = a + b
    result2 = a * b
    return result1, result2 # (result1,  result2)

data1 = multi_return(10, 20)
print(type(data1)) # <class 'tuple'>
print(data1) # (30, 200)

data1, data2 = multi_return(10, 20) # (data1, data2)
print(data1) # 30
print(data2) # 200
```

### [Case 3] default parameter

> 매개변수 값을 기본으로 정해놓음 → 인자가 넘어오지 않아도 정해놓은 값으로 지정됨

```python
# 맨 마지막에 사용

def default_param(a, b, c=False):
    if c:
        result = a + b
    else:
        result = a * b
    return result

print(default_param(10, 20, True)) # 30
```

### [Case 4] immutable vs. mutable

- **immutable(call-by-value)**

  : 넘겨준 인자값이 변경되지 않는 경우

  - 복사본이 넘어가므로, **원본 변경 안됨**
  - **[예]** 숫자, 문자열, tuple

- **mutable(call-by-reference)**

  : 넘겨준 인자값이 변경되는 경우

  - 객체 자체, 즉 객체의 주소값(원본)이 넘어가므로 **원본이 변경됨**
  - **[예]** list, dict

```python
def my_func(tmp_value, tmp_list):
    tmp_value = tmp_value + 100
    tmp_list.append(100)

data_x = 10 # immutable
data_list = [10, 20] # mutable

my_func(data_x, data_list)

print('data_x: {}'.format(data_x)) # 10(원본), 값 복사됨(복사본)
print('data_list: {}'.format(data_list)) # [10, 20, 100], 객체의 주소값 복사됨 (객체 자체가 넘어감)
```

### [Case 5] 변수의 scope

- **local variable**
- **global variable**

```python
tmp = 100

def my_func(x):
    global tmp # 비권장
    tmp = tmp + x
    return tmp

print(my_func(20))

# 그 대신, 매개변수 사용
tmp = 100

def my_func(my_temp, x):
		my_tmp = my_tmp + x
		return my_temp

print(my_func(tmp, 20))
```

## 2. 내장함수

### (1) all(x)

- 반복가능한 자료형 x에 대해 모든 값이 True → Ture
- 만약 하나라도 False → False

```python
a = [3.14, 100, 'Hello', True]
print(all(a)) # True
```

### (2) any(x)

- 반복가능한 자료형 x에 대해서 하나라도 True → True
- 모든 값이 False → False

```python
a = [0, '', None]
print(any(a)) # False
```

### (3) len(x)

- x의 길이를 알려주는 함수

### (4) int(), float(), list(), tuple(), dict(), str(), set()

- 해당 데이터 타입으로 전환용
- 초기화용

## lambda '식'

> 한 줄로 함수를 정의하는 방법

- 함수의 이름이 없는 **익명 함수(anonymous function)**
- 함수가 아니므로 **리턴값(결과값)이 없음**
- `lambda`는 **대체 표현식**으로 local scope를 가지지 않기 때문에 스택에 별도의 메모리 공간을 차지하지 않음
- **[표현방법]** `lambda 매개변수1, 매개변수2, ...: 매개변수를 이용한 표현식`

```python
f = lambda a, b, c: a + b + c # f = 표현식
print(f(10, 20, 30))
```

# 객체지향

> 😎 유지보수, 가독성, 편리한 프로그래밍

## 들어가기전,

### 절차적 프로그래밍 (C)

- 프로그램을 **기능** 단위로 모듈로 세분화
- 결국 더 이상 세분화 할 수 없는 단위 기능들이 도출됨
- 따라서 **함수**로 단위 기능과 모듈을 구현함

**[장점]**

- 프로그램을 쉽고 빠르게 구현
- 프로그램의 설계를 빠르게 할 수 있고 누가 설계를 하던 지 거의 동일한 설계 구조가 나옴

**[단점]**

- 프로그램의 규모가 커지게 되면 유지 보수가 힘들어짐

- 개발 비용 < 유지 보수 비용

- 기존 코드의 재사용성에 한계

  → 함수 단위로 가져다 쓰던지, 코드를 복사・붙여넣기해서 재사용함

------

### 객체 지향 프로그래밍 (C++, Java)

- 해결해야 되는 문제를 그대로 프로그램으로 묘사하는 방식

- 프로그램을 구성하는 주체를 파악함 → 주체들의 상호작용을 프로그램으로 표현

- 따라서, 현실 세계의 개체를 프로그램적으로 모델링하는게 중요함

  → 개체의 속성(변수)와 행위(함수)를 파악하면 됨

**[장점]**

- 프로그램의 유지 보수성과 재사용성에 유용함

**[단점]**

- 프로그램의 설계와 구현이 상대적으로 어려움

**[용어] 객체와 클래스**

- 클래스: 객체를 모델링하기 위해 사용하는 프로그램 **단위**

  1. 속성, 필드, property, attribute (←변수)
  2. 메서드, method (←함수)

  ⇒ (정리) 클래스는 속성과 메서드의 집합!

- 객체 (클래스의 인스턴스): 클래스를 기반으로 프로그램에서 사용할 수 있는 메모리 영역을 할당받음

### 💡함수 vs. 메서드

- **함수**: 독립적으로 존재하며, 로직 처리 이후 사용자가 원하는 결과를 반환함

- **메서드**: 클래스에 종속되어 존재하며, 해당 클래스에 대한 객체가 생성되어야 사용할 수 있음

  → `static` 의 경우는 제외함

## 클래스 (=설계도)

- **[관점 1]** **객체 모델링의 수단**

  : 현실 세계의 객체를 프로그램적으로 모델링하는 프로그래밍 수단

- **[관점 2] 추상 데이터 타입**

  : (아직 만들어지지 않은) 새로운 데이터 타입을 만들어내는 수단

### 실제 구현 코드

- Python에서 모든 클래스는 상속 관계가 있기 때문에 클래스 생성 시 항상 상속 명시해야 함

  - 적어도 object class 를 상속받고 있지만,  **object 클래스는 생략 가능**
  - 클래스는 계층관계로 생성됨 (상속 개념)

- `__init__`: 생성자 (Initializer)

- **인스턴스**: 특정 메모리 공간이 할당됨 (메모리 주소가 리턴됨)

- **변수**: 인ㄴ스턴스를 가리키는 메모리 시작 주소를 가지고 있는 참조값(reference)

- `self`: 인스턴스의 메모리 공간의 시작 주소를 가지고 있음

  → 즉, 현재 자신이 사용하고 있는 메모리 공간을 지칭하므로 인스턴스(객체)를 의미함

- `dot operator`: 객체가 가지는 속성이나 메서드를 사용(access)할 시 사용

> 👋🏼 **잠깐만, 여기서 용어 정리 (클래스 > 변수, 함수)**

- java: 클래스 > 필드(변수), 함수(메서드)
- C++: 클래스 > 멤버변수(member variable), 멤버함수(member function)
- python, javascript: 클래스 > property(속성), method(메소드)

```python
# 학생
# - 속성: 이름, 학과, 학번, 학점
# - 기능: 자신의 정보를 문자열로 만들어서 리턴

class Student(object): # oject 클래스는 항상 모든 클래스에 상속되므로 생략 가능
    # Initializer(생성자 - constructor)
    # self: 인스턴스의 메모리 주소가 들어가 있음
    def __init__(self, name, dept, num, grade):
        self.name = name # self.name: 클래스의 속성, name: 인자로 들어오는 매개변수
        self.dept = dept
        self.num = num
        self.grade =grade
        
# 객체(클래스의 인스턴스)를 만들어보자!
# std
std1 = Student('Sam', 'Computer Science', '20200101', 3.5)  
std2 = Student('John', 'Mathematics', '20200102', 4.5)
std3 = Student('Chris', 'Literature', '20200103', 1.5)

# 리스트로 학생 객체들 관리
students =[]
students.append(std1)
students.append(std2)
students.append(std3)

# 객체가 가지는 속성이나 메서드를 사용할 시, dot operator 연산자를 사용함
print(students[1].name) # John

# 결국,                                                        
my_list = list() # list class의 instance를 생성하는 코드

# type(): 인스턴스가 어떤 클래스로부터 파생이 되었는 지 알려줌
print(type(my_list)) # <class 'list'>

# 메서드 구현
class Student():
    def __init__(self, name, dept, num, grade):
        self.name = name 
        self.dept = dept
        self.num = num
        self.grade =grade
    
    # 현재 객체가 가지고 있는 학생의 정보를 문자열로 만들어서 리턴
    def get_std_info(self):
        return '이름: {}, 학과: {}'.format(self.name, self.dept)

std = Student('Mathias', 'Business', '20200105', 4.0)
print(std.get_std_info())
```