# 객체지향: 속성이 동작에 의해 제어되는 개념



# 클래스

## self

> 객체(인스턴스)의 메모리 공간의 시작주소를 지칭하는 **reference variable**

- 각각의 인스턴스는 메모리 공간을 따로 가짐
- 그 메모리 공간의 시작주소를 self로 표현함

## Python은 동적으로 새로운 속성과 메서드 추가를 허용함

> 가능은 하지만, 객체지향적으로는 비추천 (유연성)

```bash
class Student():
    def __init__(self, name, dept):
        self.name = name 
        self.dept = dept 
        
    def get_std_info(self):
        return '이름: {}, 학과: {}'.format(self.name, self.dept)

std1 = Student('Sam', 'CS')
std2 = Student('Chris', 'Business')

print(std2.name)  # Chris

# 가능하지만 객체지향적으로는 비추천 (유연성)
# 속성 추가
std1.grade = 4.5 # 가능
print(std1.grade) # 4.5
print(std2.grade) # 에러남

# 메소드 추가
def my_func():
    pass
std1.my_func = my_func
```

## 함수명 = 메모리 공간의 시작주소

```python
class Student():
    def __init__(self, name, dept):
        self.name = name 
        self.dept = dept 
        
    def get_std_info(self): # 함수명도 변수처럼 취급됨 -> 실제 메모리 공간의 시작 주소 가짐
        return '이름: {}, 학과: {}'.format(self.name, self.dept)

std1 = Student('Sam', 'CS')
std2 = Student('Chris', 'Business')print(std1.get_std_info)

print(std1.get_std_info()) # 메소드 실행
print(std1.get_std_info) # 함수명 = 메모리 주소값
```



## 클래스 변수 vs. 인스턴스 변수

### **클래스 변수(class variable)**: 클래스 내부의 속성

- 모든 인스턴스가 공유함 (비추천)
- 클래스 내부에 데이터가 저장됨

### **인스턴스 변수(instance variable):** 인스턴스가 개별적으로 가지고 있는 속성

- 인스턴스 내에 각각 해당하는 변수
- 각각의 인스턴스가 개별적으로 가지고 있음
- 인스턴스 메소드도 마찬가지 (instance method)

```bash
class Student():
   
    scholarship_rate = 3.0  # class variable, 모든 인스턴스가 공유함
    
    def __init__(self, name, dept):
        self.name = name # instance variable, 인스턴스 내에 각각 해당되는 변수
        self.dept = dept # instance variable
        
    # instance method, 각 인스턴스가 개별적으로 가지고 있음
    def get_std_info(self): # 함수명도 변수처럼 취급됨 -> 실제 메모리 블록의 시작 주소 가짐
        return '이름: {}, 학과: {}'.format(self.name, self.dept)

std1 = Student('Sam', 'CS')
std2 = Student('Chris', 'Business')

# 똑같은 변수를 공유
print(std1.scholarship_rate) # 3.0
print(std2.scholarship_rate) # 3.0 

# 마치 class variable이 변경된 것 같지만, 
# namespace를 알면 간단히 이해 가능
std1.scholarship_rate = 3.5
print(std1.scholarship_rate) # 3.5
print(std1.scholarship_rate) # 3.0

# class variable 변경방법 (클래스 자체를 변경함)
Student.scholarship_rate = 3.5
print(std1.scholarship_rate) # 3.5
print(std1.scholarship_rate) # 3.5
```



# namespace

> 객체들의 요소들을 나누어서 관리하는 메모리 공간을 지칭

## namespace의 종류

1. **instance namespace**
2. **class namespace**
3. **super class namespace**

## namespace의 특징

- **다른** **namespace**에서 **같은 이름의 변수**를 쓸 경우, 메모리 주소가 관리되는 공간이 다르기 때문에  **다른 변수로 인지함**

- 클래스의 속성이나 메소드를 사용할 때 계층구조를 이용해서(namespace를 따라가면서) 속성과 메소드를 찾음

  - 이 때 **따라가는 순서는 계층구조(상속)의 역순**

    : instance namespace → class namespace → super class namespace

  - 즉 이 방향(→)으로 사용하려는 속성이나 메소드를 찾음

```bash
class Student():
   
    scholarship_rate = 3.0  # class variable
    
    def __init__(self, name, dept, grade):
        self.name = name # instance variable
        self.dept = dept # instance variable
        self.grade = grade # instance variable
        
    
    def get_std_info(self): # instance method
        return '이름: {}, 학과: {}, 학점: {}'.format(self.name, self.dept, self.grade)

    
    def is_scholarship(self): # instance method
        if self.grade >= Student.scholarship_rate: # self.scholarship_rate 도 동작함
            return 'YES'
        else:
            return 'NO'
        
std1 = Student('Sam', 'CS', 2.0)
std2 = Student('Chris', 'Business', 4.5)

print(std1.is_scholarship())
print(std2.is_scholarship())

std1.scholarship_rate = 4.5 # instance namespace에 scholarship_rate 속성이 새롭게 추가됨
print(std1.scholarship_rate) # 4.5 # instance namespace
print(std2.scholarship_rate) # 3.0 # class namespace

Student.scholarship_rate = 4.0 # class variable 변경
```



# [객체 지향] 클래스 정의시, 메서드를 통해서 속성을 '제어'

> 정보은닉: 인스턴스가 가지고 있는 속성을 외부에서 직접적인 변경(접근)이 불가능하도록 보호

## 클래스 내부에서 정보은닉이 일어나는 상황,

### 1. **Instance method: instance variable을 생성, 변경, 참조하기 위해 사용되는 메소드**

- **[사용 방법]** 클래스 내에서 메소드 정의, `self` 를 매개변수로 지정
- **[주의]** Instance method 안에서도 Instance variable 생성 가능함

### 2. Class method: instance variable가 공유하는 class variable를 생성, 변경, 참조하기 위해 사용되는 메소드

- **[사용 방법]** 데코레이터 `@classmethod` 사용, 클래스를 지칭하는 `cls` 를 매개변수로 지정

### 3. Static method: self, cls 와 같은 레퍼런스 변수를 받지 않음

- **[사용 방법]** 데코레이터 `staticmethod` 사용
- self, cls로 매개변수를 받지 않는 메소드
- 일반 함수가 클래스 내부에 존재함

```bash
# 속성과 메소드를 이용하는 방식
# 속성값 변경시?

class Student(): # 속성과 메소드들의 집합
   
    scholarship_rate = 3.0  # class variable
    
    def __init__(self, name, dept, grade):
        self.name = name # instance variable
        self.dept = dept # instance variable
        self.grade = grade # instance variable
        
    
    def get_std_info(self): # instance method
        return '이름: {}, 학과: {}, 학점: {}'.format(self.name, self.dept, self.grade)

    
    def is_scholarship(self): # instance method
        if self.grade >= Student.scholarship_rate: 
            return 'YES'
        else:
            return 'NO'
        
	# 1. Instance method 사용
    def change_info(self, name, dept):
        self.name = name
        self.dept = dept

  # 2. Class method 사용 
    **@classmethod**
    def change_scholarship_rate(**cls**, rate):
        cls.scholarship_rate = rate
     
	# 3. Static method 사용   
    **@staticmethod**
    def print_hello():
        print('Hello')
      
        
std1 = Student('Sam', 'CS', 2.0)
std2 = Student('Chris', 'Business', 4.5)

# 객체지향적으로 객체의 속성을 임의로 바꾸는 것은 옳지 않음 (가능은 하지만...)
std1.name = 'John' # 비추천
std1.dept = 'Mathematics' # 비추천

# 1. Instance method 올바른 사용법
std1.change_info('John', 'Mathematics') # 맞는 방법

# 그럼 class variable을 변경하려면...?
Student.scholarship_rate = 4.0 # 비추천

# 2. Class method 올바른 사용법 
Student.change_scholarship_rate(4.0)

# 3. Static method 사용
std1.print_hello()
```

## 🧐 [퀵 정리] 클래스 내부를 들여다 보면,

1. **Class variable**
2. **Class method**
3. **Instance variable**
4. **Instance method**
5. **Static method (객체지향과 맞지 않는 개념)**



# public vs. private

## public

- (속성과 함수를) 어디에서나 사용할 수 있는 경우로, 바람직하지 않지만 틀리진 않음
- 기본적으로 Python은 Instance variable, Instance method를 public으로 지정함

## private

- 메소드를 통해서만 접근이 가능함
- `__메소드명`: 클래스 내부에서만 사용이 가능함 (같은 클래스 내 다른 메소드 내부에서 사용 가능)
- **[주의] 메소드만 private이 적용됨**

```python
class Student(): # 클래스: 변수와 메서드들의 집합
   
    scholarship_rate = 3.0  # class variable
    
    def __init__(self, name, dept, grade):
        self.name = name # instance variable => public
        self.__dept = dept # instance variable => public
        self.grade = grade # instance variable => public
        
    
    def get_std_info(self): # instance method => public
        return '이름: {}, 학과: {}, 학점: {}'.format(self.name, self.dept, self.grade)

    
    def is_scholarship(self):
        if self.grade >= Student.scholarship_rate: 
            return 'YES'
        else:
            return 'NO'
        
    def __change_info(self, name, dept): # instance method => private
        self.name = name
        self.dept = dept

std1.__dept = "English" # 오류 발생 안하네...
std1.__change_info('Sam', 3.2) # 오류 발생
```



## 상속 (클래스간의 계층 관계가 성립)

> 상속은 상위 클래스의 특징을 이어받아서 확장된 하위 클래스를 만들어내는 방법으로 코드의 재사용성을 확보함

- 상위 클래스(super class): 상속을 내려주는 클래스
- 하위 클래스(sub class): 상속을 받아서 확장하는 클래스

**[사용 방법]**

- ```
  class 하위 클래스명(상위 클래스명):
  ```

  - `super(하위 클래스명, self).__init__(상위 클래스의 매개변수)`

**[장점]**

- 코드의 반복을 줄이고 재활용성을 높임

**[단점]**

- 클래스를 재활용하기 위해서는 독립적인 클래스인 경우가 더 좋은데,
- 즉 상위 클래스와 하위 클래스가 서로 긴밀하게 연결되어 있음 (tightly coupled)

```python
# 상위 클래스 (super class, parent class, base class)
class Unit(object):
    def __init__(self, damage, life):
        self.utype = self.__class__.__name__ # 현재 객체의 클래스 대한 정보.이름
        self.damage = damage
        self.life = life
    
my_unit = Unit(100, 200)
print(my_unit.damage) # 바람직하지 않음
print(my_unit.utype) # Unit # 바람직하지 않음
    

# 하위 클래스 (sub class, child class)
class Marine(Unit):
    pass

marine_1 = Marine(300, 400)
print(marine_1.damage)
print(marine_1.utype) # Marine

# 하위 클래스 (sub class, child class)
class Marine(Unit):
    def __init__(self, damage, life, offense_upgrade):
        super(Marine, self).__init__(damage, life) # 자기(Marine) 상위 클래스의 __init__ 호출 
        self.offense_upgrade = offense_upgrade

marine_2 = Marine(400, 500, 2)
print(marine_2.damage)
```



# magic function - 클래스 내부의 특수한 함수들

> 코드 상에서 직접 호출안하고 특정 경우에 자동으로 호출되는 클래스 내부의 함수, 오버라이딩과 비슷

## 가장 많이 쓰이는 __함수__ 3가지

### 1. **__init__**

- 객체(인스턴스) 생성 시 자동으로 초기화해주는 함수

### 2. **del**

- 객체가 삭제될 때 이 메소드가 자동적으로 호출됨
- 실제로는 객체(인스턴스)가 메모리에서 삭제되기 이전에 호출되어 이 객체(인스턴스)가 사용한 resource를 해제함
- 그리고 나서, 비로소 객체(인스턴스)를 메모리 상에서 삭제

**[객체가 삭제되는 경우 2가지]**

1. 해당 객체의 reference가 끊기는 경우 객체는 자동적으로 소멸됨

2. `del 객체(인스턴스)명`

### 3. **str**

- 현재 클래스로부터 파생된 객체(인스턴스)를 문자열로 변경 시 호출됨

```python
class Student(object):
    def __init__(self, name, dept, grade):
        print('객체 생성')
        self.name = name
        self.dept = dept
        self.grade = grade
        
    def __del__(self): 
        print('객체 소멸')

    def __str__(self): # -- (*)
        return '이름은: {}, 학과는: {}'.format(self.name, self.dept)
    
    def __gt__(self, other): # (std1, std2)
        return '> 연산자에 의해 호출됨'

# 인스턴스: Student('Sam', 'English'), std1: 메모리 주소값(reference variable)
std1 = Student('Sam', 'English', 3.0) 
print(std1) # 메모리 주소값이 출력됨
            # 특정한 문자열을 출력하고 싶은 경우는? -- (*)
    
# 객체가 생성되면 특정 메모리 주소에 메모리 공간이 할당됨 (0x100)
# 객체가 만들어질 때마다 메모리 주소가 달라짐
# 따라서 두번 째 실행에서 객체가 생성되면 특정 메모리 주소에 공간이 할당됨 (0x200)
    # 객체 생성
    # 객체 소멸

del std1
     
std2 = Student('Chris', 'Business', 4.0)

# 객체에 대한 연산자의 의미를 의도대로 바꿈
print(std1 > std2) # 원래는 오류 코드이지만, 내 의도대로 작성하기 위해 magic function 사용
```



# 모듈(객체)

> 함수, 변수, 클래스를 모아놓은 파일로, 확장자가 `.py` 로 끝나야함

## 모듈의 기능

- 외부의 파이썬 파일들을 현재 파일에 이용할 수 있도록 해줌
- 코드의 재사용성을 높이고 코드 관리를 용이하게 함

## 모듈 사용 시,

- `import` 로 모듈을 불러옴
- 이 때 파일 안에 있는 내용을 모두 **객체화** 시켜서 우리 파일이 사용하는 **메모리에 로드**시킴

## 실습(윈도우)

### 환경 세팅

1. 시스템 속성 → 고급 → **환경변수**
2. 사용자 변수 새로 만들기
3. 새 사용자 변수
   - 변수 이름: PYTHONPATH
   - 변수 값: *특정 디렉토리 경로 지정* + `;.`
     - 예를 들어, 경로: *자신이 원하는 경로에서*/python_lib
     - 현재 폴더를 지칭하는 `.` 입력
4. 새로 시작

------

### 1. 모듈 생성

> 지정한 디렉토리 경로 폴더(python_lib)에서 파일 생성: [module1.py](http://module1.py/)

```python
def my_sum(a, b):
    return a + b

my_pi = 3.141592

class Student(object):
    def __init__(self, name, dept):
        self.name = name
        self.dept = dept
```

------

### 2. 모듈 사용

> import 시, 파일의 이름만 기입

```python
# module을 import 하면 module이 객체화(인스턴스화)되어 들어옴

import module1 as m1 # 별칭 사용
from module1 import * 

print(module1.my_pi) # 3.141592
print(m1.my_sum(10, 20)) # 30
```

### 3. 패키지 사용

> from 모듈 import 특정 속성 또는 메소드

```python
from module1 import my_pi

print(my_pi) # 3.141592
```

### 4. 중첩된 폴더 내 모듈 사용

- python_lib (폴더)
  - my_folder1
    - my_folder2
      - my_module.py → `variable1 = 'success!'`

```python
# 비권장
import my_folder1.my_folder2.my_module as m1
print(my_folder1.my_folder2.my_module.variable1)

# 폴더가 계층 구조로 되어 있을 때의 모듈 사용 시 권장 방법
from my_folder1.my_folder2 import my_module
print(my_module.variable1)
```