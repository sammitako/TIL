# 웹 프로그래밍 용어 정리

- **네트워크**: 각종 통신 장비들이 서로 연결되어 데이터를 교환하는 통신망

- **인터넷**: 전세계적으로 구축된 네트워크 망

  -   network of network 
    
    -   물리적인 Framework


- **인터넷 서비스** (특정 프로토콜 이용)
-   **HTTP Web Service**
      
    -   SMTP Email Service
      
    -   FTP File Service


- **Web Client** (Program)

- **Web Server** (Program)

  
    -   정적인 Contents(가지고 있는 파일)를 서비스
    
    -   동적인 컨텐츠를 생성(서버에서 별도의 프로그램을 실행시켜 나온 결과)해서 서비스를 할 수는 없음
    
    - 동적인 컨텐츠를 서비스하기 위해서 WAS에게 작업 위임
  
    (WAS는 내부에 실행시킬 별도의 프로그램을 가지고 있음)


- **Server-side Web Application**: 서버에서 실행되는 별도의 웹 프로그램 (WAS에서 수행됨)

- **Framework**: 수행 체계, 구조가 구축되어 있음 → 수정해서 기능 구현

- **Library**: 특정 기능을 수행하는 코드 묶음(함수, 클래스), 알고리즘과 로직은 제공하지 않음

- **Platform**: 다른 프로그램을 실행시켜줄 수 있는 환경이 되는 프로그램

  ​				   예) OS 계열, Anaconda

## CASE 1 - Round Trip

Client-side web application의 코드는 Server-side web application에서 Response(HTML, CSS, JavaScript)를 통해 전달해준 HTML/CSS를 렌더링을하고, 클라이언트가 이벤트를 발생 시 Web Client 내부에서 JavaScript가 실행됨



**문제점**

- 서버쪽에서 모든 프로그램이 다 작성됨
- 데이터의 전달량이 많음

**실습**

Django를 통해 WAS에 Server-side Web Application 프로그램을 구현



## CASE 2 - Single Page Application (AJAX 방식)

Server-side Web Application과 Client-side Web Application을 분리해서 구현

- Web Client - HTML + CSS + JavaScript

- Web Server - 실행된 프로그램으로 도출된 결과를 데이터 표현 방식(CSV, XML, JSON)으로 Response를 생성

  

**실행순서**

1. Web client는 별도의 Web Server에 Request를 보내서 HTML, CSS, JavaScript로 구성된 별도의 웹 프로그램을 Response(Client-side Web program)로 받아옴
2. 이벤트가 발생 시, 그 때 Web server로 Request를 보내서 WAS로 부터 데이터를 받으면 Web client 내에 있던 JavaScript가 실행이 되서 결과값이 보여짐



**장점**

- 유지 보수가 용이
- 전송되는 데이터 양이 적어서 서버 쪽에 부하가 적음 (서버 트래픽 분산)
- 캐싱 기법을 사용하기 때문에 매번 Client program을 호출할 필요가 없어서 속도 차이는 없음

**실습**

Client-side Web Application을 직접 생성 후 Server-side Web Application은 Open API 사용해서 구현



## CASE 2 실습

별도의 Web Server에 **Client-side Web Application**을 구현해보자.



### 첫번째 방법 (비추천): HTML 파일 하나에 모든 코드를 씀

[**환경설정]**

- Project: FrontEndWeb
- File: boxoffice
- WebStorm에 내장된 Web Server 사용

[**Code]**

- boxoffice.html

  ```HTML
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <title>Rank of Boxoffice</title>
      <script>
          function hello() {
              alert('버튼이 눌렸네요');
          }
      </script>
  </head>
  <body>
  일일 박스 오피스 순위를 알아보자!
  <br><br>
  key: <input type="text" id="userKey">
  <br><br>
  날짜: <input type="text" id="userDate">
  <br><br>
  <input type="button" value="조회" onclick="hello()">
  
  </body>
  </html>
  ```

  

### 두번째 방법 (추천): 파일 분리 형태

[**환경설정]**

- Directory: js
- File: my_script.js

[**Code]**

- boxoffice.html

  ```HTML
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <title>Rank of Boxoffice</title>
      <!--  jQuery를 이용하기 위해 CDN 방식 이용  -->
      <script src="https://code.jquery.com/jquery-2.2.4.min.js" integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script>
      <script src="js/my_script.js"></script>
  </head>
  <body>
  일일 박스 오피스 순위를 알아보자!
  <br><br>
  key: <input type="text" id="userKey">
  <br><br>
  날짜: <input type="text" id="userDate">
  <br><br>
  <input type="button" value="조회" onclick="hello()">
  
  </body>
  </html>
  ```

  

- my_script.js

  - `location`: 브라우저의 위치 객체
  - `location.href`: 현재 브라우저의 url을 다른 url으로 변경해줄 때 사용

  ```javascript
  function hello() {
      alert('버튼이 눌렸네요');
      user_key = $('#userKey').val()
      user_date = $('#userDate').val()
      open_api = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json'
  
      //user_key = 나만의 key
      my_url = open_api + '?key=' + user_key + '&targetDt=' + user_date
  
      // location = 브라우저의 위치 객체
      // 현재 브라우저의 url을 변경해줄 때 사용
      location.href = my_url // my_url을 실행해서 결과를 location(현재 브라우저)에 표현됨
  
  }
  ```

  

## JavaScript

화면 제어, 프로그램 처리

- 간단한 문법

```javascript
// 1. 변수선언
let tmp1 = 'sample'; // string
let tmp2 = 10.34; // number
let tmp3 = true // boolean
let tmp4 = [1, 2, 3, 4] // array

// 2. 변수출력
// alert(tmp1) // blocking method: 여기에서 코드의 (수행이 확인 버튼 누를 때 까지)일시 중지
console.log('변수의 값: ' + tmp1);

// 3. 객체
let obj = {
    name : 'Sam',
    age : 28
}
console.log(obj.name);

// 4. 함수
function add(x, y) {
    return x + y;
}

alert(add(10, 11))
```

