# 예외 처리

## [정의] 런타임 에러를 잡아서, 프로그램이 강제 종료됨을 막는다.

> try, except, else, finally

```python
# 예외 처리: 런타임 에러를 잡음

def my_func(list_data):
    
    my_sum = 0
    try:
        my_sum = list_data[0] + list_data[1] + list_data[2]
    except Exception as e: # Exception - 모든 에러
        print('실행 시 문제가 발생했어요ㅠㅠ')
        my_sum = 0
    else:
        # 만약 오류가 발생하지 않으면, 아래의 코드가 실행됨
        print('오류가 없네요!')
        
    finally:
        # 에러가 있건 없건 무조건 실행
        print('나는 항상 실행되는 구문:)')
    
    return my_sum
    
my_list1 = [1, 2]
print(my_func(my_list1)) # 오류 발생 코드를 예외 처리해서 잡음

my_list2 = [1, 2, 3]
print(my_func(my_list2)) # 6
```

## 언제 써야 되지?

- 외부 리소스를 연결할 경우 (예를 들어, 네트워크, 데이터베이스 사용 시)
- 다른 요인으로 나의 프로그램이 오류가 날 경우
- 코드 내에서 예상할 수 없는 오류가 날 경우

# 파일 처리

## 먼저, 외부파일 만들기

1. jupyter notebook 홈 디렉토리로 이동

2. temp_txt.txt 파일 생성

   ```python
   이것은
   소리없는
   아우성
   (4번째 줄은 공백)
   ```

## 파일 읽기

- 모든 라인 읽기 위해서, `while` 문 사용
- 파일 읽고 반드시 리소스 해제 처리
- 실제로 작업 시에는 `pandas` 에 있는 파일 처리 기능을 이용함

```python
# 파일 처리 -> 나중에 pandas를 이용!

my_file = open('temp_txt.txt', 'r') # 읽기용으로 오픈

# 파일 안에 있는 모든 line 읽기(출력)
# '', 공백 문자열은 False로 간주
while True:
    line = my_file.readline()
    print(line)
    if not line: # not + False = True
        break

my_file.close() # 반드시, 사용한 리소스 해제 처리!
```