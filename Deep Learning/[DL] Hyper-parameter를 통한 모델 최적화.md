# Hyper-parameter를 이용한 머신러닝 모델 최적화

최적의 hyper-parameter를 찾는 과정을 자동화시킬 수 없을까?

## 1. `GridSearchCV`

hyper-parameter의 값을 몇개로 정해서 내부적으로 Cross Validation을 k번 반복해서 처리한다.

### 코드 구현

Hyper-parameter에 대한 리스트 안에 딕셔너리 형태로 파라미터를 설정한다. key값으로 hyper-parameter 이름을 넣어주고, value값으로 적용할 값들을 리스트 형식으로 넣어준다.

  - `kernel`
  - `C`
  - `gamma`

  ```python
  from sklearn.model_selection import GridSearchCV
  
  # data split
  x_data_train, x_data_test, t_data_train, t_data_test = \\
  train_test_split(x_data, t_data, test_size=0.2, random_state=0)
  
  # hyper-parameter
  param_grid = [
      # 8번
      {'kernel': ['linear'],
       'C': [10,30,100,300,1000,3000,10000,30000]},
      # C(8번)⨉gamma(6번) = 48번
      {'kernel': ['rbf'],
       'C': [1,3,10,30,100,300,1000,3000],
       'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}
  ]
  
  # model (옵션 X)
  model = SVC() 
  
  # 최적화
  # 5⨉(8번 + 48번): Fitting 5 folds for each of 56 candidates, totalling 280 fits
  grid_search = GridSearchCV(model, param_grid, 
                             cv=5, scoring='accuracy', verbose=2) 
  
  # learning
  grid_search.fit(x_data_train, t_data_train)
  
  # 최적의 파라미터 출력
  result = grid_search.best_params_
  score = grid_search.best_score_
  print(result)
  print(score)
  ```

## 2. `RandomizeSearchCV`

hyper-parameter의 값을 선정할 수 있는 범위를 정해서 값의 개수를 정해주면 수학적 분포 내에서 값을 랜덤하게 추출해서 내부적으로 Cross Validation을 k번 반복해서 처리한다.

### 코드 구현

- Log 분포(C) - `from scipy.stats import reciprocal`

- Exponential 분포(gamma) - `from scipy.stats import expon`

- `n_iter`: 로그 함수와 지수 함수 내에서 값을 랜덤하게 몇 번 추출할지 설정

```python
from scipy.stats import expon, reciprocal
from sklearn.model_selection import RandomizedSearchCV

# data split
x_data_train, x_data_test, t_data_train, t_data_test = \
train_test_split(x_data, t_data, test_size=0.2, random_state=0)

# hyper-parameter
param_dist = {
    'kernel': ['linear', 'rbf'],
    'C': reciprocal(20,200000),
    'gamma': expon(scale=1.0)
}

# model (옵션 X)
model = SVC() 

# 최적화
random_search = RandomizedSearchCV(model, param_dist, 
                           cv=5, n_iter=50,
                           scoring='accuracy', verbose=2) 

# learning
random_search.fit(x_data_train, t_data_train)

# 최적의 파라미터 출력
result = random_search.best_params_
score = random_search.best_score_
print(result)
print(score)
```

<br>

-----

Reference: [ML_0401_SVM]()