# [Python] 들어가기 전, 개발환경 세팅 소개

## Anaconda 패키지 설치

1. 아나콘다 홈페이지에서 다운로드 설치

2. (pip를 최신버전으로 업데이트)

   ```bash
   python -m pip install --upgrade pip
   ```

3. **가상환경**을 하나 설치함 (기본 가상공간: base)

   ```bash
   conda create -n data_env python=3.7 **oepnssl**
   conda info --envs # 가상환경 리스트 확인, 현재 가상환경은 * 로 표시해줌
   
   # 가상환경 삭제 (사용된 폴더는 자동으로 지워지지 않음)
   conda remove --name __가상환경 이름__ --all
   ```

4. 기본 가상공간에서 생성된 가상환경으로 전환

   ```bash
   conda activate data_env
   ```

5. `nb_conda` 라는 패키지 설치

   - Jupyter Notebook 사용을 위함

   ```bash
   conda install nb_conda
   ```

6. 환경설정 파일 생성

   - Jupyter Notebook 파일 저장 경로 지정을 위함
   - 환경설정 파일이 홈 디렉토리를 설정함

   ```bash
   jupyter notebook --generate-config
   ```

   ```bash
   # Code-editor에서 수정, 385번째 줄
   
   c.NotebookApp.notebook_dir = '홈 디렉토리 PATH'
   ```

7. 지정한 홈 디렉토리에, 소스코드가 저장될 폴더 생성

8. Jupyter Notebook 실행

   ```bash
   jupyter notebook
   ```



## Jupyter Notebook

### Python [conda env.data_env]

![jupyter-notebook](../md-images/jupyter-notebook.png)

### 기억해야 할 용어 및 단축키

- cell
- cell 선택 시 파랑색으로 바뀜
- `b`: 아래쪽에 새로운 cell 생성
- `a`: 위쪽에 새로운 cell 생성
- `dd`: 해당 cell 삭제
- `ctrl + enter`: cell 안의 코드 실행