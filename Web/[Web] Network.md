# 웹 프로그래밍

## Basics

- **용어**

  **웹 프로그램은 CS 구조로 되어 있다.**

  - C(client): 능동적으로 서비스를 요청
  - S(server): 대기하고 있다가 클라이언트 요청에 대해 서비스를 제공

  ------

  - Web client (program): 웹 서버에 접속하는 클라이언트 프로그램

    예) 웹 브라우저

  - Web server (program): 웹 클라이언트의 요청을 받아 서비스를 제공하는 프로그램

    예) Apache web server , IIS, Oracle Web server

- **웹 프로그램 정의**

  : **HTTP protocol**로 통신하는 서버프로그램과 클라이언트 프로그램을 개발

  - 웹 클라이언트 프로그램 (Front-end)
  - 웹 서버 프로그램 (Back-end)
  - 둘다 (Full-stack)
  - 웹 상에서 데이터 통신 방식 보기

- **프로토콜의 정의**

  : 데이터 통신을 위해 지켜야할 약속 혹은 규약

  - HTTP (웹 전용 프로토콜)
  - FTP (파일 전송 프로토콜)
  - SMTP (이메일 전송 프로토콜)

- **HTTP protocol의 특징**

  **= stateless protocol**

  - 클라이언트와 서버는 데이터 통신을 할 때만 잠깐 연결되고, 데이터 통신이 끝나면 해당 연결을 끊음

    → 서버의 부하를 줄임

  - 그러면 서버가 클라이언트를 구별할 수 없게됨 (stateless)

    → Session Tracking 필요

  - 채팅의 경우, 클라이언트의 정보를 항시 알아야하기 때문에 네트워크 연결 상태를 계속유지하게 됨

- **IP 주소**

  : Network에 연결되어 있는 각종 기기에 부여되는 논리적인 주소 (바뀔 수 있음)
  
  
    NIC(Network Interface Card) 카드에 할당되어 있는 주소
  
    `쩜`을 기준으로 *4자리*
  
  
    ￮  서브넷 주소
  
    ￮  리얼 IP
  
- **Mac Address**

  : Network에 연결되어 있는 각종 기기에 부여되는 물리적인 주소 (바꿀 수 없음)

  `쩜`을 기준으로 *6자리*

  따라서, IP 주소를 Mac Address로 내부적으로 변경해주어야 해당 컴퓨터를 찾아갈 수 있음

  Domain Name → IP 주소 → Mac Address

- **PORT**

  : 하나의 프로세서(프로그램)을 지칭하는 0 ~ 65535 사이의 숫자 (포트번호)

  - 0 ~ 1024: reserved(예약)

    예) 웹 서버의 포트번호: 80(생략가능)

  - 1025 ~ 65535: 사용자가 이용

  - PORT 예제

⇒ 네트워크 통신 시, **URL 웹주소:** `HTTP:// IP주소:포트번호`

## Web client application

- HTML
- CSS
- JavaScript

  

## Web server application

- Django (python)
- ~~Servlet (java)~~

  

## HTTP Protocol (**HyperText Transfer Protocol)**

웹 서버와 웹 클라이언트가 서로 데이터를 주고 받기 위해 사용하는 통신 규약

TCP/IP Protocol 스택 위에서 동작하는 IP 기반의 통신 프로토콜

(텍스트, 이미지, 동영상, pdf, ... 등 여러 종류의 데이터를 주고 받을 수 있음)

### HTTP Request message

> GET /book/readme.md  HTTP/1.1 
>
> Host: [www.example.com:80](http://www.example.com:80)
>
> 
>
>  body



- 요청방식 메소드 (4)

  ￮  **GET** - 서버의 리소스를 가져오고 싶을 때, 얻어올 때 사용

  ￮  **POST** - 새로운 데이터를 서버에 생성하는 요청을 할 때 사용

  ￮  PUT - 서버 쪽에 있는 데이터를 수정할 때 사용

  ￮  DELETE - 서버 쪽에 있는 데이터를 삭제할 때 사용

  ⇒ **REST 방식,**

  ​	그러나 주로 **GET, POST** 두 가지 방식으로 모든 CRUD 작업이 가능

  

- URL - 특정 리소스 지칭

- HTTP 버전 - 사용하는 현재 프로토콜 버전

- HOST - 어떤 호스트에 요청을 보내고 있는지에 대한 HOST 정보(이름)

  (한줄 띄고)

- body - 데이터를 서버에 전달 시 해당 데이터의 정보

  

→ 프로그램(웹 브라우저, 라이브러리 등)에서 내부적으로 request message를 이 형식으로 만들어줌

### GET 방식

------

전달할 데이터를 Query String 형식으로 URL 뒤에 붙여서 보냄

(추가적인 데이터를 포함하여 서버에 요청할 때 사용)

- 예

  일일 박스 오피스 순위를 알아보기 위해 Open API 사용 시, 키값과 해당 날짜 데이터를 포함하여 서버 쪽에 요청을 보냈음

  당시 WAS가 DB를 통해 결과를 산출한 뒤 JSON 형태로 response 해줬음

**장점**

- 사용하기 쉬움

**단점**

- 보완 취약

- 요청 URL은 길이제한이 있음

  → 서버에 전달할 데이터에 대한 길이 제약이 따름

### POST 방식

------

body 부분에 추가적 데이터를 포함하여 서버에 request message를 전송

**장점**

- 그나마 보완성이 있음

- 보내려는 데이터의 길이 제한이 없음

  → 새로운 데이터(파일 업로드)를 생성할 시 사용됨

**단점**

- 보완 취약

- 요청 URL은 길이제한이 있음

  → 서버에 전달할 데이터에 대한 길이 제약이 따름

## Web server vs. WAS(Web Application Server)

### Web server

정적 리소스에 대한 응답만 가능 (프로젝트 단위로 서비스 제공)

- 정적 리소스 - 이미 존재하는 것
- 동적 리소스 - 새로 만들어야 하는 것

서버에 특정 프로그램 호출 요청이 들어온다면?

그때 Web server는 WAS에 request 작업을 전달함 (위임)

### WAS (프로그램)

프로그램을 호출, 실행(프로그램 처리)해서 결과를 Web server로 리턴함

따라서 서버쪽 웹 프로그래밍(Back-end)이란, WAS 안에 프로그램들을 미리 만들어 두어서 클라이언트가 요청할 때마다 실행하는 것



⇒ 웹의 기본 구조: Web client, Web server, WAS, DB