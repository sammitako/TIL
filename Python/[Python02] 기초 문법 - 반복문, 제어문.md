# 잠깐! 리뷰타임 ☕️

## 파이썬의 데이터 타입 (built-in type)

>1. numeric - int, float, complex
>2. sequence - list, tuple, range
>3. text sequence - str
>4. mapping - dict
>5. set - set
>6. bool -bool



## 반복문

### for 문 - 반복 횟수를 알고 있을 때 사용

```python
for tmp in [1,2,3,4,5]:
    print(tmp)           # print()는 default형태로 사용하면 출력후 줄바꿈함
                         # 만약 내용출력후 줄바꿈대신 다른 처리를 하려면 end 속성을 이용
                       
for tmp in [1,2,3,4,5]:
    print(tmp, end='-')
```

### while 문 - 조건에 따라서 반복할 때 사용

```python
idx = 0
while idx < 10:
    print(idx)
    idx += 1
```

### list comprehension - 리스트를 생성할 때 사용

```python
a = [1,2,3,4,5,6,7]

# [2,4,6,8,10,12,14]
list1 = [tmp * 2 for tmp in a if tmp % 2 == 0]
print(list1)
```

## 제어문

### if 문, if-elif-else 문

```python
a = 20

if a % 3 == 0:
    print('3의 배수')
elif a % 5 == 0:
    print('5의 배수')
elif a % 7 == 0:
    print('7의 배수')
elif a % 11 == 0:
    pass
else:
    print('조건에 해당되는게 없음')
```