# Google Colaboratory(Colab)

구글이 제공해주는 클라우드 기반의 Jupyter Notebook으로 학습 속도가 굉장히 빠르다.

### **Colab 구성**

1. 디스크 스페이스
2. 메모리 사이즈

------

### **Colab 특징**

- 협업용으로 좋음
- 환경설정이 쉬움
- (무료) 최대 세션 시간이 12시간
- (유료) 최대 세션 기간이 24시간

<br>

## 구글 드라이브

특정 폴더에 `.ipynb` 파일을 작성할 수 있다.

### 설정

- 내 드라이브에서 새폴더 (ML Colab) 생성
- 설정의 앱 관리에서 **기본값으로 사용** 체크 후 완료
- ML Colab 폴더에서 Google Colaboratory 새로 만들기
- 런타임의 런타임 유형 변경에서 하드웨어 가속기를 TPU(Tensor 개선용 CPU) 또는 GPU로 변경해서 저장
- 폴더에서 구글 드라이브로 마운트(연결)

### 텐서플로우 버전

Colab은 텐서플로우 최신버전이 기본으로 설치되어 있다. (거의 모든 머신러닝 라이브러리가 이미 내장되어 있음)

`tf.__version__` 확인 결과, `2.4.1` 버전이 설치되어 있다. 따라서 2.x 버전을 삭제하고 1.15 버전을 설치해서 사용해야 한다.

(참고: Colab에 새로 접속할 때마다 런타임이 초기화가 된다. 즉, 구글에서 리소스를 재할당 받아서 사용하기 때문에 텐서플로우가 최신 버전으로 설정되어 있으므로 아래의 코드를 항상 실행해줘야 한다.)

아래의 두가지 방법 중 하나를 선택해서 사용하면 된다.

1. 현재 설치되어 있는 2.x 버전(최신 버전) 삭제 후 1.x 버전 설치
   - `!pip uninstall tensorflow`
   - `!pip install tensorflow==1.15`
   - RESTART RUNTIME 클릭
2. 현재 설치되어 있는 2.x 버전(최신 버전)을 비활성화 후 1.x 버전 설치
   - `import tensorflow.compat.v1 as tf`
   - `tf.disable_v2_behavior()`

### 단축키

- `cmd+M+L`: 라인 넘버 표시하기
- `cmd+M+M`: 마크다운으로 전환
- `cmd+M+Y`: 코드로 전환
- `cmd+M+A`: 바로 윗줄에 셀 생성
- `cmd+M+B`: 바로 아랫줄에 셀 생성
- `cmd+M+D`: 셀 삭제