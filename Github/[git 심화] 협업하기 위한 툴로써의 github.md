# [git 심화] 협업툴로써의 github



## .gitignore 파일 (프로젝트 초기 설정 단계에서 실행)

> git 저장소 내에서 git으로 관리하고 싶지 않은 파일이 있다면, .gitignore 파일을 만들어서 관리

```bash
# 특정파일
data.csv

# 특정폴더
images/

# 특정확장자
*.png
!profile.png # 특정파일 제외
```

- git으로 관리하지 파일을 `.gitignore` 문서 안에 파일 명을 입력 (Code Editor로 입력 권장)
- 일반적으로, 개발환경, 운영체제, 특정 언어 등에서 임시 파일과 같이 개발 소스코드와 관련 없는 파일은 git으로 관리하지 않음
- git으로 관리하지 않는 파일 체크용 링크



## 본격적으로 원격 저장소(github)을 활용해보자 🥰

### github 사용 시 꼭 기억해야 할 명령어!

#### • push

```bash
git add .
git commit -m 'message'
git push origin master
```

##### push 충돌 상황

**[충돌 상황]** **github 상에서** README **작업하자** **자동 커밋됨,** 그 후 로컬 상 해당 폴더를 `push` 하자 충돌 발생

**[충돌 이유]** 로컬의 커밋 파일과 원격 저장소의 커밋 파일과 다름, 즉 커밋 버전이 달라서 충돌남

**[해결방안]**

1. pull

   ```bash
   git pull origin master # Esc + :wq 로 빠져나오기 
   ```

2. merge commit 발생

3. push

   ```bash
   git push origin master
   
   $ git log --oneline
   # merge commit 발생!
   3bb716a (HEAD -> master, origin/master) Merge branch 'master' of <https://github.com/sammitako/practice>
   ```

#### • pull

> 원격 저장소의 변경 사항을 받아옴

```bash
git pull origin master
```

⚠️ 만약 충돌날 경우, `git log`와 `github`의 히스토리를 비교 후 `pull` → `push` 하면 THE END

#### • clone

> 원격 저장소를 복제하여 로컬에서 활용할 수 있도록 함

```bash
git clone 'HTTPS URL' # 원격 저장소 이름의 폴더가 생성되고, 해당 폴더로 이동하면 git을 활용할 수 있음
# 복제 완료 후, 해당 폴더에서 작업 시작~
```

#### [FAQ] clone과 Download ZIP의 차이점

- `clone`을 할 경우 DVCS 를 활용 (프로젝트 **이력들**을 모두 받아옴)
- 압축파일을 할 경우 CVCS 를 활용 (해당 프로젝트의 최신 시점의 파일만 받아옴)

#### [FAQ] clone과 init의 차이점

- `clone`: 원격 저장소를 로컬 저장소로 받아 오는 행위

- `init`: 로컬 저장소를 **새롭게 시작**하는 행위

  

### Branch에 대해 알아보자!

#### Branch 활용 명령어

##### • Branch 목록

```bash
git branch
```

##### • Branch 생성

```bash
git branch __브렌치 이름__
```

##### • Branch 삭제

```bash
git branch -d __브렌치 이름__
```

##### • Branch 이동

```bash
git checkout __브렌치 이름__
git checkout -b __브렌치 이름__ # 브렌치 생성 및 이동
```

##### • Branch 병합

```bash
## 현재 디렉토리 경로 확인
(master) git merge __브렌치 이름__ # master 브렌치에 __브렌치 이름__을 병합시킴
```



#### Branch 활용 예제

##### <예제 상황> 네이버 메인 페이지 개발

- master: 12/30일에 사용자가 보고 있는 버전 (개발자 입장에서 12/28일에 완료된 버전)
- feature branches: 로고팀, 페이팀, ...
- hotfix: 검색 이슈 발생 인지 후 긴급 패치

##### <예제 결론> Branch를 사용해야 하는 이유

- 만약 가지가 한개일 경우, 현재 진행 중인 모든 작업들이 같이 패치됨
- 따라서 각 가지들이 **독립적으로** 다양한 작업을 동시에 할 수 있고 따로 패치가 가능함



#### Branch 합칠 때 발생하는 3가지 상황

##### 일단, 준비물 😋

- `branch` 라는 폴더 이름 생성
- 첫 번째 커밋 남기기 ([README.md](http://README.md) 파일 생성)



##### **[상황 1] fast-foward**

> feature 브랜치 생성된 이후 **master 브랜치에 변경 사항이 없는 상황**

1. **feature/test** branch 생성 및 이동

   ```bash
   git branch feature/test # Branch 생성
   git branch # 목록 확인
   git checkout feature/test 
   ```

2. 작업 완료 후 commit

   ```bash
   touch test.txt
   git add .
   git commit -m 'Complete test'
   
   git log --oneline
   # feature/test Branch + **HEAD: 현재 있는 위치 정보**
   5ff4709 (HEAD -> feature/test) Complete test
   # master branch
   c6f5db0 (master) Add README
   ```

3. master 이동

   ```bash
   git checkout master
   ```

4. master에 병합

   ```bash
   git merge feature/teset
   # Fast-forward!!!
   # master에 변경사항 없어서 그냥 앞으로!!!
   ```

5. 결과 -> fast-foward (단순히 HEAD를 이동)

   ```bash
   git log --oneline 
   # 5ff4709 (HEAD -> master, feature/teset) Complete test
   ```

6. branch 삭제

   ```bash
   git branch -d feature/test
   ```

------



##### **[상황 2] merge commit**

> 서로 다른 이력(commit)을 병합(merge)하는 과정에서 다른 파일이 수정되어 있는 상황; git이 auto merging을 진행하고, commit이 발생된다.


1. **feature/data** branch 생성 및 이동

   ```bash
   git checkout -b feature/data
   ```

2. 작업 완료 후 commit

   ```bash
   touch data.txt
   git add .
   git commit -m 'Complete data'
   git log --oneline 
   # (HEAD -> feature/data) Complete data
   # 5ff4709 (master) Complete test
   # c6f5db0 Add README
   ```

3. master 이동

   ```bash
   git checkout master # Switched to branch 'master'
   git log --oneline # c6f5db0 (HEAD -> master) Add README
   ```

4. *master에 추가 commit 이 발생시키기!!*

   - **다른 파일을 수정 혹은 생성하세요!**

     ```bash
     touch hotfix.txt
     git add .
     git commit -m 'hotfix'
     git log --oneline # (HEAD -> master) hotfix
     ```

5. master에 병합

   ```bash
   git merge featrue/data
   # Merge made by the 'recursive' strategy.
   ```

6. 결과 -> 자동으로 *merge commit 발생*

   - vim 편집기 화면이 나타납니다. (vim: 터미널 용 문서 편집기 창)

   - 자동으로 작성된 커밋 메시지를 확인하고, `esc`를 누른 후 `:wq`를 입력하여 저장 및 종료를 합니다.

     - `w` : write
     - `q` : quit

   - 커밋이 확인 해봅시다.

     ```bash
     git log --oneline
     
     # 44515f8 (HEAD -> master) Merge branch 'feature/data'
     # 6930e34 hotfix
     # 6b0245e (feature/data) Complete data
     # 5ff4709 Complete test
     # c6f5db0 Add README
     ```

7. 그래프 확인하기

   ```bash
   git log --oneline graph
   
   *   44515f8 (HEAD -> master) Merge branch 'feature/data'
   |\\
   | * 6b0245e (feature/data) Complete data
   * | 6930e34 hotfix
   |/
   * 5ff4709 Complete test
   * c6f5db0 Add README
   ```

8. branch 삭제

   ```bash
   git branch -d feature/data
   # 가지 자체를 지우는 행위가 아니라, 가지의 정보가 없어짐
   ```

------



##### **[상황 3] merge commit 충돌**

> 서로 다른 이력(commit)을 병합(merge)하는 과정에서 **동일 파일이 수정되어 있는 상황;** git이 auto merging을 하지 못하고, 해당 파일의 위치에 라벨링을 해준다.원하는 형태의 코드로 직접 수정을 하고 merge commit을 발생 시켜야 한다.


1. **feature/web** branch 생성 및 이동

   ```bash
   git checkout -b feaeture/web
   ```

2. 작업 완료 후 commit

   ```bash
   # READE.md 파일 수정!!!
   touch README.md
   git add .
   git commit -m 'Update README and Complete web'
   ```

3. master 이동

   ```bash
   git checkout master
   ```

4. *master에 추가 commit 이 발생시키기!!*

   - **동일 파일(README.md)을 수정 혹은 생성하세요!**

     ```bash
     # README 파일을 수정
     git status
     # On branch master
     # Changes not staged for commit:
     # (use "git add <file>..." to update what will be committed)
     # (use "git restore <file>..." to discard changes in working directory)
     #      modified:   README.md
     
     # no changes added to commit (use "git add" and/or "git commit -a")
     
     git add .
     git commit -m 'Update README'
     ```

5. master에 병합

   ```bash
   git merge feature/web 
   # 충동을 고치고 결과를 커밋해야되네...
   # CONFLICT (content): Merge conflict in README.md
   # Automatic merge failed; fix conflicts and then commit the result.
   # (master|MERGING)
   ```

6. 결과 -> *merge conflict발생*

   ```bash
   git status 
   
   On branch master
   You have unmerged paths.
     (fix conflicts and run "git commit")
     (use "git merge --abort" to abort the merge)
   
   Changes to be committed:
           new file:   web.txt
   # 어디서 충돌이 난건지 충돌 파일 확인 필요
   Unmerged paths:
     (use "git add <file>..." to mark resolution)
           both modified:   README.md
   ```

7. 충돌 확인 및 해결

   ```bash
   # 해결전
   <<<<<<< HEAD
   # Project
   
   * data 프로젝트 
   =======
   # 프로젝트
   
   * web 개발
   >>>>>>> feature/web
   
   # 해결후
   # Project
   
   * data 프로젝트
   * web 개발
   ```

   ```bash
   git add .
   ```

8. merge commit 진행

   ```
   $ git commit
   ```

   - vim 편집기 화면이 나타납니다.

   - 자동으로 작성된 커밋 메시지를 확인하고, `esc` 를 누른 후 `:wq` 를 입력하여 저장 및 종료를 합니다.

     - `w` : write
     - `q` : quit
     
- 커밋이 확인 해봅시다.
  
9. 그래프 확인하기

   ```bash
   git log --oneline --graph
   *   1a08480 (HEAD -> master) Merge branch 'feature/web'
   |\\
   | * 156b027 (feature/web) Update README and Complete web
   * | 30c71d2 Update README
   |/
   *   44515f8 Merge branch 'feature/data'
   |\\
   | * 6b0245e Complete data
   * | 6930e34 hotfix
   |/
   * 5ff4709 Complete test
   * c6f5db0 Add README
   ```

10. branch 삭제

    ```bash
    git branch -d feature/data
    ```



### Github Flow 기본 원칙

#### pull request I

##### 1. Create a new branch

```bash
git checkout -b 02 # branch 만든 후 작업
git add .
git commit -m 'Complete 02'
git push origin 02 # 원격 저장소에 있는 branch 02로 push
```

##### 2. Merge a branch at github

- Open a pull request and Create pull request
- Merge pull reques



#### pull request II

##### <Open Source 참여 방법>

1. 원본 저장소에서 **Fork** 뜨기 (저장소에 push 권한이 없으므로)

2. 나의 github에서 해당 저장소 

   ```
   Clone
   ```

    받기

   - 폴더명이 원본 저장소의 폴더명과 동일하게 생성됨
   - 명령어는 반드시 폴더로 이동해서 입력
   - 절대 `init` 명령어 쓰지 마세요!

3. 작업 후 나의 로컬 저장소에서 `git push origin master`

4. 나의 github 저장소에서 **Pull request** 클릭

5. **Create pull request** 클릭 후 메세지 작성하면 THE END

   

### 추가적인 내용

#### • git status

> commit 하기 전 할일 들 체크

```bash
$ git status
On branch master
# 커밋될 변경사항들
# Staging Area O
**Changes to be committed:**
  (use "git restore --staged <file>..." to unstage)
  # a.txt 삭제된...
        **deleted:    a.txt**

# 변경사항인데 Staging 아닌것
# Working Directory O
**Changes not staged for commit:**
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
  # b.txt가 수정된...
        **modified:   b.txt**
# Untrack 
# Working Directory O
**Untracked files:**
  (use "git add <file>..." to include in what will be committed)
  # 새로운 파일..
        **c.txt**
```

##### git 저장소 내의 변경사항을 추적

- `untracked`: 한번도 git으로 관리된 적이 없거나 파일 생성 등

- ```
  tracked
  ```

  - ```
    modified
    ```

    - `modified`: 수정
    - `deleted`: 삭제

  - `unmodified`: `git status` 에 등장하지 않음

#### • git log

> merge 하기 전 이력들을 체크



### 거꾸로 돌려~ :cyclone:

#### • Add 취소

```bash
git restore --staged <file>
```


#### • WD 작업 내용 취소

> ‼️ 커밋되지 않은 변경 사항을 없애는 것으로, 명령어를 실행한 이후 다시 돌이킬 수 없음 
>
> [참고] 커밋되어 있다면 다 살릴 수 있어요 🥳

```bash
# Working Directory에 있는 파일의 변경 사항을 버림
git restore <file>
```

#### • commit 메시지 변경

> ‼️ 협업 시 역사를 바꾸려고 하지마라... 하늘이 노한다...

- 커밋 고유번호(해시값)가 변경됨
- 커밋이 아예 달라짐
- 공개된 저장소에 이미 `push` 가 된 경우 절대 변경을 하지 않는다!

```bash
git commit --amend

#vim
내용 수정
```

#### •  reset vs. revert (비권장...)

##### reset: 이전의 작업 내용을 삭제함

- `--hard`: 모든 작업 내용(변경사항)과 이력을 삭제 (매우 조심!!!)
- `--mixed`: 모든 작업 내용(변경사항)을 Staging Area에 보관
- `--soft`: Working Directory 내용까지도 보관

```bash
$ git log --oneline
0c330b4 (HEAD -> master) Add f.txt
d81c176 작업끝
57ad4ef Status
fb4ad8d Add b.txt
ec0574d Add a.txt

$ git reset --hard d81c176
HEAD is now at d81c176 작업끝

$ git log --oneline
d81c176 (HEAD -> master) 작업끝
57ad4ef Status
fb4ad8d Add b.txt
ec0574d Add a.txt
```



##### revert: 되돌린 내역을 남기고 삭제함

```bash
$ git log --oneline
0c330b4 (HEAD -> master) Add f.txt
d81c176 작업끝
57ad4ef Status
fb4ad8d Add b.txt
ec0574d Add a.txt

$ git revert 0c330b4
Removing f.txt
[master 56ff1b7] Revert "Add f.txt"
 1 file changed, 0 insertions(+), 0 deletions(-)
 delete mode 100644 f.txt

$ git log --oneline
56ff1b7 (HEAD -> master) Revert "Add f.txt"
0c330b4 Add f.txt
d81c176 작업끝
57ad4ef Status
fb4ad8d Add b.txt
ec0574d Add a.txt
```