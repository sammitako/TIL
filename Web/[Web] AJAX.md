## AJAX(Asynchronous JavaScript and XML)

**페이지가 갱신되지 않는다**

📢 **jQuery**를 이용해서 AJAX 구현

```jsx
// 자바스크립트 객체
$.ajax({
        url : open_api,     // 호출할 서버쪽 프로그램의 URL
        type : 'GET',       // 서버쪽 프로그램에 대한 request 방식 - api에 GET 방식으로 호출하라고 명시되어 있음
        dataType : 'json',  // 서버 프로그램이 결과로 보내주는 데이터의 형식
        data : {            // 서버 프로그램 호출 시 서버에 넘겨주는 데이터
            key : user_key,      // 서버쪽 프로그램이 정해놓은 변수
            targetDt : user_date // 서버쪽 프로그램이 정해놓은 변수
        },
        success : function(result) {
            alert('서버 호출 성공');
						// 서버로부터 결과 json(문자열)을 매개변수로 받아옴
						// jQuery: json -> JavaScript 객체 { }로 변환
        },   
        error : function() {
            alert('서버 호출 실패');
        }
    })
```

