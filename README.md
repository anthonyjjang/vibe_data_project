
# 내 엑셀데이터와 대화하는 AI Agent

## 작동방식
![image](https://github.com/user-attachments/assets/e2f69ff8-ba16-498e-b6a8-48656369b838)


## 설치 방법

1. 필수 패키지 설치  
   ```bash
   uv add openai pandas streamlit python-dotenv openpyxl
   ```

2. 환경 변수 파일(.env)을 생성한 후, OpenAI API 키를 입력.
   ```env
   OPENAI_API_KEY=여기에_발급받은_키_입력
   ```

## 샘플 데이터

샘플 파일 다운로드 출처:  
https://www.data.go.kr/data/15083033/fileData.do#tab-layer-openapi

## 실행

Streamlit 앱 실행  
   ```bash
   uv run streamlit run app.py
   ```


