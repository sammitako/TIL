# Machine Learning

## 용어정리

👉🏼 **AI(Artificial Intelligence)**

: 인간이 가지는 특유의 2가지 능력(학습능력, 추론능력)을 컴퓨터로 구현하려는 가장 포괄적인 개념

- **ML(Machine Learning)** - AI를 구현하는 방법 중 하나

  : 데이터의 특성을 파악하고 패턴을 **학습**해서 그 결과로 미지의 데이터에 대한 추정치를 계산(**추론, 예측**)하는 프로그래밍 기법  

  

  **[프로그래밍 기법]**

  1. **Regression**

  2. SVM(Support Vector Machine)

  3. Decision Tree / Random Forest (앙상블)

  4. Naive Bayes

  5. KNN

  6. **Neural Network ⇒ Deep Learning (CNN, RNN, LSTM, GAN)**

     : 신경망을 이용해서 학습하는 구조와 알고리즘이 최근에 개선, 개발되고 있음

  7. Clustering

     - K-means
     - DBSCAN

  8. Reinforcement Learning (강화학습)  

     

- **Deep Learning**

  : Machine Learning 기법 중 Neural Network의 확장 버전

  - 추론의 정확성 높음
  
  - 학습 시간이 느림  
  
    

👉🏼 **Data Mining**

: 데이터 **분석**에서 사용되며 데이터 속성(feature) 간의 상관관계를 파악하거나, 기존 데이터에서 새로운 속성(feature)을 알아낼 때 사용  



## WHY Machine Learning?

- Explicit program으로 해결할 수 없는 문제를 해결하기 위해서 1960년대에 등장

  - **Explicit Program이란?**

    : Rule based program을 지칭, 즉 규칙, 조건이 정해져있는 프로그램

- 경우의 수(규칙, 조건)가 너무 많아서 프로그램이 불가할 경우에 사용

- 예) 바둑, 이메일 필터, 자율 주행  

  

## Machine Learning 학습 방법


4번을 제외하고 데이터를 기반하는 학습 방법  



1. **지도학습(Supervised Learning) = Regression**

   : Training Data Set(학습 데이터 셋)을 이용하여 학습해서 미지의 데이터(입력값)에 대해 (정답을) 예측

   Training Data Set (정제된 데이터)로 학습을 진행하여 예측 모델(Prediction Model)을 만들고, 즉 수학 공식을 만들게 되는데 이 모델로 예측 작업을 시행  

   

   **[Training Data Set]**

   - 입력값(feature) - x로 표현

   - ⭐ **정답(lable) - t로 표현 (실수)**  

   - 예) Lable의 형태에 따라 머신러닝의 기법이 정해짐  

       

     - 공부 시간에 따른 시험점수 예측 - 공부시간(x), 시험점수(t)

       **[데이터 형태]** 공부시간, 시험점수

       **[질문]** 7시간 공부하면  몇점 받을까?

       **[기법] Regression:** 연속적인 숫자값을 예측, 즉 "얼마나"를 예측  

       

     - 공부 시간에 따른 시험 합격여부 예측 - 공부시간(x), 합격여부(t)

       **[데이터 형태]** 공부시간, 0(faile)또는 1(pass)로 표현

       **[질문]** 7시간 공부하면 합격할까?

       **[기법] Binary Classification**: Lable이 둘 중 하나로 정해져 있는 경우

       Binary Classification 예) 기존 카드 결제 패턴 (정상거래? 비정상 거래? → 카드 도용?)

       - **Classification(분류)**

         **:** 이미 정해져 있는 종류(분류) 중  어떤 종류(분류)의 값이 도출될 지를 예측, 즉 "어떤 것"을 예측  
       
         

     - 공부 시간에 따른 성적 Grade(학점) 예측
     
       **[데이터 형태]** 공부시간, One-Hot-Encoding(A, B, C, D, F)
     
       **[질문]** 7시간 공부하면 학점이 어떻게 될까?
     
       **[기법] Multinomial Classification**: Label이 둘 이상으로 정해져 있는 경우  
       
       

2. 비지도학습(Unsupervised Learning)

   : Label이 존재하지 않는 Training Data Set으로 학습하여 비슷한(연관성이 있는) 데이터(입력값)끼리 묶음

   **예)** 각 뉴스 기사의 의미 파악하여 같은 부류끼리 Clustering(분류)해서 정치, 연예, 스포츠 기사로 나눔  

   

3. 준지도학습(Semi-supervised Learning)

   : 지도학습, 비지도학습이 섞여 있는 형태로 Training Data Set이 어떤 입력값에 대해서는 Lable이 존재하지만 어떤 입력값에 대해서는 Lable이 존재하지 않는 경우

   → 일단, 입력 데이터로 비지도 학습(Clustering)을 한 후 클러스터링 안에서 Lable이 있는 데이터를 기준으로 그 Lable을 모두 같이 맞춰줘서 학습하게 됨  

   

4. 강화학습(Reinforcement Learning) - 리워드(보상)를 기반으로 한 학습 방법