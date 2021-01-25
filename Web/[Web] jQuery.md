## jQuery

JavaScript 라이브러리로, 화면 제어

CDN(Contents Delivery Network) 방식: 태그를 이용해서 라이브러리 사용 (네트워크가 가능하다는 가정 하에 물리적으로 떨어져 있는 사용자에게 컨텐츠를 더 빠르게 제공할 수 있는 기술)

```html
<!-- CDN 방식으로 사용 -->
<!-- 네트워크 사용 가능 환경에서만 작동 -->
<script src="<https://code.jquery.com/jquery-2.2.4.min.js>" integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script>
```

[**사용법]**

```
$('selector').method()
```

찾아서, 적용



1. **Selector**

   **HTML element**를 지칭하는 특수한 표기법

   - `*`: 전체 선택자
   - `'태그명'`: 태그 선택자
   - `#`: id 선택자
   - `.`: 클래스 선택자 (CSS 용도로 사용)
   - 구조 선택자 (HTML의 계층 구조 사용)
     - `>`: 자식
     - `<`: 부모
     - `공백`: 후손
     - `+`: 바로 다음에 나오는 형제
     - `~`: 에 나오는 모든 형제
   - `'태그[태그속성]'`: 속성 선택자 (속성을 이용해서 선택)
   - `:first`, `:last`: 순서 선택자
   - `:eq(순번)`: 순서 선택자

2. **Method**

   - `.val()`: 값
   - `.css()`: 보여지는 형태(스타일)을 변경 (전체 화면을 렌더링 → 자원 효율 떨어짐)
   - `.remove()`: 나 자신 포함 모두 지움
   - `.empty()`: 자신은 삭제하지 말고 자신의 후손을 모두 삭제
   - `.text(' ')`: 태그 사이에 들어가 있는 글자 가지고 오거나 변경할 수 있음
   - `.disable()`: 사용할 수 없게 만듦
   - `.attr( , )`: 속성에 대한 값을 가지고 오거나 변경할 수 있음
   - `.each(자바스크립트 람다함수)`: 반복 처리
   - `.addClass()`: CSS 스타일 적용
   - `.removeClass()`: CSS 스타일 제거
   - `.removeAttr()`: 속성 제거

- 코드예제

  1. JavaScript 파일: jQuery_exercise.js

     **jQuery 적용**

     ```jsx
     function my_func() {
         // 버튼을 누르면 호출됨
         // jQuery 사용법
         // 1. selector
     
         $('*').css('color', 'red'); // 전체 선택자
         $('span').remove(); // 태그 선택자
         $('li').css('background-color', 'yellow'); // 태그 선택자
     
         $('#inchon').text('소리 없는 아우성'); // id 선택자
         $('.region').css('color', 'blue'); // class 선택자
         $('ol > li').css('color', 'green')  // 구조 선택자
     		$('ol + div').css('color', 'pink'); // 구조 선택자
     		$('ol ~ div').css('color', 'gray'); // 구조 선택자
     		
     		// $('input[type]'): input tag를 찾아서 type이라는 속성이 있는 elemenet를 찾음
         $('input[type=button]').disable() // 속성 선택자
     
     }
     ```

  2. **HTML** 파일: jQuery_exercise.html

     **[HTML 용어정리]**

     * **element**: HTML 구성 요소로 시작 태그 부터 끝 태그 까지를 의미함

       예외적으로 `img` 태그는 닫는 태그가 없음 (inline element)


         ￮   block level element: element가 한 라인을 완전히 차지

         ￮   inline element: element가 해당 내용만 영역을 차지


         	⬩  영역 예제

       ​			1. `div` : block level element

       ​			2. `span` : inline element

       * **tag**: `<>` 로 구성되는 HTML 요소

       * **HTML의 계층 구조** - 후손, 부모, 형제, 자식

       * 하나의 HTML 파일에서 **id 값**은 unique 하다.

       * 반면 **클래스**는 중복값 허용 (CSS를 위해 사용되는 속성)

  ```jsx
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <title>Title</title>
      <script src="<https://code.jquery.com/jquery-2.2.4.min.js>" integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script>
      <script src="js/jQuery_exercise.js"></script>
  </head>
  <body>
      <h1>여기는 h1입니다.</h1>
      <!-- unordered list -->
      <!-- 계층 구조 -->
      <ul>
          <li class="region">서울</li>
          <li id="inchon">인천</li>
          <li>부산</li>
          <li>제주</li>
      </ul>
  
      <!-- ordered list -->
      <ol>
          <li>Sam</li>
          <li>Brian</li>
          <li>Jack</li>
      </ol>
  
      <div>이곳은 div 영역 입니다</div>
      <span class="region">이곳은 span 영역 입니다</span>
      <img src="img/car.jpg">
      <br><br>
      <input type="button" value="클릭" onclick="my_func()">
  </body>
  </html>
  ```

**[디자인 패턴] jQuery 요즘도 사용하나?!**

유지 보수 강화를 위해 패턴에 입각한 Framework 사용



## jQuery 연습문제 (1)

> \#1. 사과, 파인애플, 참외 출력 
>
> #2. 사용자 입력 값 출력 
>
> #3. 고양이 출력 
>
> #4. 호랑이 출력 
>
> #5. 여러개 반복 출력

[**사용자 입력 양식]**

**form 태그를 사용**

- 종류 - 글상자, radio button, check box, putdown list, select box, button

- 용도 - 사용자로 부터 받은 데이터를 서버에 전송

- form 속성 2가지

  - `action`: 서버 쪽 프로그램 (일일 박스 오피스 호출하는 프로그램)
  - `method`: 서버 쪽 요청 방식 (REST)

- HTML file: jQuery_sample01.html

  ```jsx
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Document</title>
      <script src="<https://code.jquery.com/jquery-2.2.4.min.js>"
          integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script>
      <script src="jQuery_sample01.js"></script>
  </head>
  <body>
      <div>
          <ul>
              <li id="apple">사과</li>
              <li id="pineapple">파인애플</li>
              <li class="myList">참외</li>
          </ul>
          <!-- 사용자 입력 양식 -->
          <form action="#" method="post">
              <input type="text" id="uId" size="20"> <!-- 한줄짜리 입력 상자 -->
          </form>
          <ol>
              <li class="myList">고양이</li>
              <li class="myList">호랑이</li>
              <li class="myList">강아지</li>
          </ol>
          <input type="button" value="클릭" onclick="my_func()">
      </div>
  </body>
  </html>
  ```

- JavaScript file: jQuery_sample01.js

  ```jsx
  function my_func() {
      console.log($('#apple').text()); // #1
      console.log($('#pineapple').text()); // #1
      console.log($('ul > li[class]').text()); // ul > .myList
  
      console.log($('#uId').val()); // #2
  
      // uId 라는 속성의 값?
      console.log($('input[type=text]').attr('id'));
  
      // 속성 변경
      $('input[type=text').attr('size', 30);
  
      console.log($('ol > li:first').text()); // #3
  
      console.log($('ol > li:first + li').text()); // #4
  
  		console.log($('ol > li:eq(1)').text()); // #4
  
  		// 반복 처리
  		$('ol > li').each(function(idx, item) { // 람다 함수 -> 매개변수
          // idx: 순서(0부터 시작), item: element 통째
          console.log($(item).text() + '입니다.') // jQuery 형태로 변경
      });
  
  }
  ```

## jQuery 연습문제 (2)

- HTML file: jQuery_sample02.html

  ```jsx
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Document</title>
  
      <script src="<https://code.jquery.com/jquery-2.2.4.min.js>"
          integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script>
      <script src="jQuery_sample02.js"></script>
      
      
      <style>
          
          .myStyle {
              width: 300px;
              height: 100px;
              background-color: yellow;
          }
  
      
      </style>
  
  </head>
  <body>
      <div>이것은 소리 없는 아우성!</div>
      
      <div class="myStyle">
          <ol>
              <li>홍길동</li>
              <li>김길동</li>
          </ol>
      </div>
      <input type="button" value="버튼 비활성화" disabled="disabled"> <!-- 속성 사용-->
      <input type="button" value="버튼 활성화" onclick="button_clicked()">
      
  
      <input type="button" value="스타일 변경" onclick="my_func()">
      <input type="button" value="스타일 제거" onclick="remove_func()">
  
      <input type="button" value="클래스 제거" onclick="remove_class()">
      <input type="button" value="나빼고제거" onclick="remove_notme()">
      
  </body>
  </html>
  ```

- JavaScript file: jQuery_sample02.js

  ```jsx
  function my_func() {
  	// 비추천
      $('div').css('color', 'red'); // 단점: 전체 화면 렌더링
      $('div').css('background-color', 'yellow');
  
  	// 추천
      $('div').addClass('myStyle');
  }
  
  function remove_func() {
      $('div').removeClass('myStyle');
  }
  
  function button_clicked() {
      $('input[type=button]:first').removeAttr('disabled');
  }
  
  function remove_class() {
      $('div.myStyle').remove();
  }
  
  function remove_notme() {
      $('div.myStyle').empty();
  }
  ```

## jQuery를 이용한 Element 생성, 수정, 삭제

### 없는 element 생성

1. `$('<태그명></태그명>').text('태그 사이 글자 입력')`
2. `$('<img />').attr('src', '값')`

### 생성한 또는 선택한 element를 원하는 위치에 붙이기

1. `aapend()`: 자식으로 붙이고, 맨 마지막 자식으로 붙임
2. `prepend()`: 자식으로 붙이고, 맨 처음 자식으로 붙임
3. `after()`: 형제로 붙이고, 바로 다음 형제로 붙임
4. `before()`: 형제로 붙이고, 바로 이전 형제로 붙임

- HTML file: jQuery_sample03.html

  ```jsx
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Document</title>
  
      <script src="<https://code.jquery.com/jquery-2.2.4.min.js>"
          integrity="sha256-BbhdlvQf/xTY9gja0Dq3HiwQF8LaCRTXxZKRutelT44=" crossorigin="anonymous"></script>
      <script src="jQuery_sample02.js"></script>
      
      
      <style>
          
          .myStyle {
              width: 300px;
              height: 100px;
              background-color: yellow;
          }
  
      
      </style>
  
  </head>
  <body>
      <div>이것은 소리 없는 아우성!</div>
      <ul>
          <li>Kim</li>
          <li>Park</li>
          <li>Lee</li>
      </ul>
  
      <input type="button" value="클릭" onclick="my_func()">
      
  </body>
  </html>
  ```

- JavaScript file: jQuery_sample03.js

  ```jsx
  function my_func() {
      // element 생셩
      let my_div = $('<div></div>').text('안녕:)'); // <div>안녕:)</div>
      let my_img = $('<img />').attr('src', 'img/car.jpg');
  
  		// 붙이기 함수
      // append(), prepend(), after(), before()
      let my_li = $('<li></li>').text('Sam')
      $('ul').append(my_li)
      $('ul').prepend(my_li)
      $('ul > li:eq(1)').after(my_li) // park
      $('ul > li:last').before(my_list)
      
  }
  ```