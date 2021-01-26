# 사용자 이벤트 처리

## 1. HTML의 이벤트 관련 속성 이용

- `onclick=" "`: 모든 HTML element에 사용 할 수 있음
- `onmouseover=" "`: 특정 컴포넌트 위에 마우스가 올라감
- `onmouseenter=" "`: 특정 영역으로 마우스가 들어감
- `onmouseleave=" "`: 특정 영역에서 마우스가 나감

### [예제 코드]

- **HTML - jQuery_sample03.html**

  ```html
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Document</title>
      <script src="<https://code.jquery.com/jquery-2.2.4.min.js>"
          integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script>
  
      <script src="jQuery_sample03.js"></script>
  		<style>
  			.myStyle {
  	            background-color: yellow;
  	            color: red;
  	        }
  		</style>
  </head>
  <body>
      <!-- jQuery Event 처리 -->
      <h1 onclick="my_func()">여기는 H1 영역입니다.</h1>
  		<h1 onmouseover="set_style()" onmouseleave="release_style()">여기는 H1 영역입니다.</h1>
  	
      
  </body>
  </html>
  ```

- **JavaScript - jQuery_sample03.js**

  ```jsx
  function add_event() {
      // H1을 찾아서 해당 element에 event 처리 능력을 부여 (버튼을 눌렀을 때 부여)
      $('h1').on('click', function(event) {
          alert('h1이 클릭됨')
      })
  }
  ```

## 2. jQuery를 이용한 이벤트

(1) HTML의 element를 찾아서 이벤트 처리 능력을 부여

- `$('HTML element').on('정해진 이벤트 종류', 람다함수(event){})`
- `event`: 객체, 현재 발생한 이벤트에 대한 세부정보

### [예제 코드]

- **HTML - jQuery_sample03.html**

  ```html
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Document</title>
      <script src="<https://code.jquery.com/jquery-2.2.4.min.js>"
          integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script>
  
      <script src="jQuery_sample03.js"></script>
  		<style>
  			.myStyle {
  	            background-color: yellow;
  	            color: red;
  	        }
  		</style>
  </head>
  <body>
      <!-- jQuery Event 처리 -->
      <h1 onclick="my_func()">여기는 H1 영역입니다.</h1>
  		<h1 onmouseover="set_style()" onmouseleave="release_style()">여기는 H1 영역입니다.</h1>
  	
      
  </body>
  </html>
  ```

- **JavaScript - jQuery_sample03.js**

  ```jsx
  function my_func() {
      alert('Ouch!')
  }
  function set_style() {
      $('h1').addClass('myStyle')
  }
  function release_style() {
      $('h1').removeClass('myStyle')
  }
  ```

**(2) HTML 시행순서 고려 형태**

- `$(document).on('ready', 람다함수(event){})`

- `document`: 웹 브라우저의 흰색(바디) 영역

- `ready`: HTML에 대한 렌더링이 다 끝났을 때, 따라서 `document`(화면)이 준비가 다 되었으면

- **축약형태**

  `$(document).ready(function(){})`

### [예제 코드]

- **HTML - jQuery_sample03.html**

  ```html
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Document</title>
      <script src="<https://code.jquery.com/jquery-2.2.4.min.js>"
          integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script>
  
      <script src="jQuery_sample03.js"></script>
      <style>
          .myStyle {
              background-color: yellow;
              color: red;
          }
      </style>
  </head>
  <body>
      <h1>소리 없는 아우성</h1>
      <!-- <input type="button" value="클릭" onclick="add_event()"> -->
  
      
  
  </body>
  </html>
  ```

- **JavaScript - jQuery_sample03.js**

  ```jsx
  $(document).on('ready', function() {
      $('h1').on('click', function(event){
          alert('클릭');
      })
  })
  
  // 축약 표현
  $(document).ready(function() {
      
  })
  
  function add_event() {
      // H1을 찾아서 해당 element에 event 처리 능력을 부여 (버튼을 눌렀을 때 부여)
      $('h1').on('click', function(event) {
          alert('h1이 클릭됨')
      })
  }
  ```

**[주의] 이벤트 핸들러가 똑같은 태그에 붙을 경우?**

이벤트가 발생됐을 때, 어떤 element에서 이벤트가 발생됐는 지 파악 필요

- `this`: 현재 사용된 객체에 대한 참조 변수
- 이벤트가 발생한 element를 지칭하는 데에 사용됨
- 사용할 시 jQuery 형태로 변경, `$(this)`

### [예제 코드]

- **HTML - jQuery_sample03.html**

  ```html
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Document</title>
      <script src="<https://code.jquery.com/jquery-2.2.4.min.js>"
          integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script>
  
      <script src="jQuery_sample03.js"></script>
      <style>
          .myStyle {
              background-color: yellow;
              color: red;
          }
      </style>
  </head>
  <body>
  		<!-- this 사용하면 해결됨 -->
      <h1>사용자: Tom</h1>
      <h1>사용자: Sam</h1>
      
  
  </body>
  </html>
  ```

- **JavaScript - jQuery_sample03.js**

  ```jsx
  $(document).on('ready', function() {
      $('h1').on('click', function(event){
          // 이벤트가 발생했을 때, 어떤 element에서 이벤트가 발생했는 지 파악 필요
          alert($(this).text())
      })
  })
  ```

## 파이널 예제

- **HTML - dailyBoxOfficeSearch.html**

  ```jsx
  <!doctype html>
  <html lang="en">
  
  <head>
      <meta charset="utf-8">
      <title>BoxOffice Search</title>
  
      <!-- jQuary: 기본 라이브러리 -->
      <script src="<https://code.jquery.com/jquery-2.2.4.min.js>"
          integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script>
  
      <!-- Bootstrap core CSS -->
      <link href="<https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/css/bootstrap.min.css>" rel="stylesheet"
          integrity="sha384-giJF6kkoqNQ00vy+HMDP7azOuL0xtbfIcaT9wjKHr8RbDVddVHyTfAAsrekwKmP1" crossorigin="anonymous">
      <script src="<https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/js/bootstrap.bundle.min.js>"
          integrity="sha384-ygbV9kiqUc6oa4msXn9868pTtWMgiQaeYH7/t7LECLbyPA2x65Kgf80OJFdroafW"
          crossorigin="anonymous"></script>
      
  
      <!-- Favicons -->
  
      <style>
          .bd-placeholder-img {
              font-size: 1.125rem;
              text-anchor: middle;
              -webkit-user-select: none;
              -moz-user-select: none;
              user-select: none;
          }
  
          @media (min-width: 768px) {
              .bd-placeholder-img-lg {
                  font-size: 3.5rem;
              }
          }
      </style>
  
      <!-- Custom styles for this template -->
      <link href="css/dashboard.css" rel="stylesheet">
      <script src="dailyBoxOfficeSearch.js"></script>
  </head>
  
  <body>
  
      <header class="navbar navbar-dark sticky-top bg-dark flex-md-nowrap p-0 shadow">
          <a class="navbar-brand col-md-3 col-lg-2 me-0 px-3" href="#">Box Office</a>
          <button class="navbar-toggler position-absolute d-md-none collapsed" type="button" data-bs-toggle="collapse"
              data-bs-target="#sidebarMenu" aria-controls="sidebarMenu" aria-expanded="false"
              aria-label="Toggle navigation">
              <span class="navbar-toggler-icon"></span>
          </button>
          <input class="form-control form-control-dark w-100" type="text" placeholder="yyyymmdd 형식으로 날짜를 입력해주세요" 
                 id="userInputDate" aria-label="Search">
          <ul class="navbar-nav px-3">
              <li class="nav-item text-nowrap">
                  <a class="nav-link" href="#" onclick="my_func()">Search</a>
              </li>
          </ul>
      </header>
  
      <div class="container-fluid">
          <div class="row">
              <nav id="sidebarMenu" class="col-md-3 col-lg-2 d-md-block bg-light sidebar collapse">
                  <div class="position-sticky pt-3">
                      <ul class="nav flex-column">
                          <li class="nav-item">
                              <a class="nav-link active" aria-current="page" href="#">
                                  <span data-feather="home"></span>
                                  Ranking
                              </a>
                          </li>
                      </ul> 
                  </div>
              </nav>
  
              <main class="col-md-9 ms-sm-auto col-lg-10 px-md-4">
                  <h2>일일 박스오피스 검색 순위</h2>
                  <div class="table-responsive">
                      <table class="table table-striped table-sm">
                          <thead>
                              <tr>
                                  <th>순위</th>
                                  <th>영화제목</th>
                                  <th>누적관객수</th>
                                  <th>누적매출액</th>
                                  <th>삭제</th>
                              </tr>
                          </thead>
                          <tbody id="mv_tbody">
                              <tr>
                                  <td>1,001</td>
                                  <td>Lorem</td>
                                  <td>ipsum</td>
                                  <td>dolor</td>
                                  <td>sit</td>
                              </tr>
                          </tbody>
                      </table>
                  </div>
              </main>
          </div>
      </div>
  </body>
  
  </html>
  ```

- **JavaScript - dailyBoxOfficeSearch.js**

  ```jsx
  function my_func() {
      // 사용자가 입력한 날짜를 가져와서
      // 해당 날짜에 대항 BoxOffice 순위를 알려주는 서버 쪽 웹 프로그램을 호출
      // 그 결과를 화면에 출력
      let user_date = $('#userInputDate').val();
      let user_key = '2d8e579e31759d1dc24ccba95acb6553';
      let open_api = '<http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json>'
      // let my_url = open_api + '?key=' + user_key + '&targetDt=' + user_date;
  
      // 아래처럼 하면 안됨
      // location.href = my_url // 내 브라우저의 현재 상태를 my_url을 호출한 결과물로 대체 -> 내 웹 어플리케이션이 리프레쉬(갱신)됨
  
      // AJAX - 자바스크립트가 가지고 있는 데이터 통신 방식 사용
      // jQuery 이용해서 AJAX 구현 (자바스크립트 객체)
      $.ajax({
          url : open_api, // 호출할 서버쪽 프로그램의 URL
          type : 'GET', // 서버쪽 프로그램에 대한 request 방식 - api에 GET 방식으로 호출하라고 명시되어 있음
          dataType : 'json', // 서버 프로그램이 결과로 보내주는 데이터의 형식
          data : {            // 서버 프로그램 호출 시 넘어가는 데이터
              key : user_key, // 서버쪽 프로그램이 정해놓은 변수
              targetDt : user_date
          },
          success : function(result) {
  
              alert('서버 호출 성공'); // 확인용
              // 서버로부터 결과 json(문자열)을 매개변수를 받아옴
              // jQuery: json -> JavaScript 객체로 변환
              // console.log(result['boxOfficeResult']['boxofficeType']) // 확인용, 일별 박스오피스
              
              $('#mv_tbody').empty() // 데이터 재로드 시 데이터 미리 지워놓기
              let movie_list = result['boxOfficeResult']['dailyBoxOfficeList'] // array
              for(let i = 0; i < movie_list.length; i++) {
                  // 서버로부터 데이터 가져옴
                  // 최종 객체에 대해서 사용할 시 .속성으로 많이 사용함
                  let m_name = movie_list[i].movieNm;
                  let m_rank = movie_list[i].rank;
                  let m_sales = movie_list[i].salesAcc;
                  let m_audi = movie_list[i].audiAcc;
  
                  // 가져온 데이터를 HTML Element에 붙여넣기
                  // <tr>
                  //     <td>1,001</td>
                  //     <td>Lorem</td>
                  //     <td>ipsum</td>
                  //     <td>dolor</td>
                  //     <td>
                  //          <input type=button value=삭제>
                  //     </td>
                  // </tr>
  
                  let tr = $('<tr></tr>')
                  let rank_td = $('<td></td>').text(m_rank);
                  let name_td = $('<td></td>').text(m_name);
                  let sales_td = $('<td></td>').text(m_sales);
                  let audi_td = $('<td></td>').text(m_audi);
                  let delete_td = $('<td></td>')
                  let delete_btn = $('<input />').attr('type', 'button').attr('value', '삭제');
                  
                  // <tr></tr> 삭제
                  delete_btn.on('click', function(){
                      $(this).parent().parent().remove() // input -> 부모: td -> td의 부모: tr
                  })
                  
                  
                  $('#mv_tbody').append(tr)
                  tr.append(rank_td);
                  tr.append(name_td);
                  tr.append(sales_td);
                  tr.append(audi_td);
                  tr.append(delete_td);
                  delete_td.append(delete_btn);
  
              }
          },
          error : function() {
              alert('서버 호출 실패');
          }
      })
  
  }
  ```