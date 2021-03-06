# TFRecord: 텐서플로우의 표준 데이터 포맷

일반적으로 프로그램이 특정 부분에서 병목 현상이 발생하여 프로그램 전반의 성능을 저하시키는데, 그 중 가장 큰 요인으로 IO(입출력) Latency에 따른 속도 지연이다.

IO에 대한 오버헤드를 방지하기 위해 텐서플로우의 데이터 포맷 형식인 TFRecord를 사용하여 속도를 향상시킬 수 있다.

## **TFRecord 정의**

입력 데이터를 Binary 압축 파일 하나로 만들어서 Deep Learning을 수행할 때 필요한 데이터를 보관하기 위한 자료구조(데이터 포맷)이라 할 수 있다.

- 프로토콜 버퍼 형식으로 데이터를 저장
- 여러 개의 데이터를 일렬로(serializable) 저장

<br>

## **TFRecord 특징**

- 입력 데이터(X)와 레이블 데이터(T)를 하나의 파일 안에서 같이 관리할 수 있다.
- 이진 데이터이기 때문에 인코딩, 디코딩 작업이 필요없으므로 처리 속도가 빠르다.
- 압축 파일로 만들어지기 때문에 속도와 용량면에서 상당한 이득이 있다.

<br>

## TFRecord 코드 

1. Tensorflow에서 제공하는 타입별 Feature 객체들 생성
2. DataFrame 생성
3. TFRecord 생성 함수
4. Data Split 후 TFRecord 생성
5. TFRecord 파일 사용

<details>
  <summary>Cats and Dogs Example Code</summary>
  <a href="https://github.com/sammitako/TIL/blob/master/Deep%20Learning/source-code/DL_0331_TFRecord_CATS%26DOGS.ipynb">DL_0331_TFRecord_CATS&DOGS</a>
</details>

