from openai import OpenAI
import streamlit as st
import pandas as pd
import json
import re
from dotenv import load_dotenv
import os
import requests
import re
load_dotenv()


#######################  llm 호출 함수 ########################

def llm_call(prompt: str) -> str:
    """
    주어진 프롬프트로 LLM을 동기적으로 호출합니다.
    이는 메시지를 하나의 프롬프트로 연결하는 일반적인 헬퍼 함수입니다.
    """
    model = "gpt-4.1" # o4-mini가 가능하면 제일 좋음
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    messages = [{"role": "user", "content": prompt}]
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    print(model, "완료")
    return chat_completion.choices[0].message.content



# 만약 ollama를 이용할 경우 활용


# def llm_call(prompt: str) -> str:
#     """
#     Ollama의 REST API를 사용하여 Qwen 모델을 호출합니다.
#     """

#     def remove_think_tags(text: str) -> str:
#         """
#         Removes all content enclosed in <think>...</think> tags from the input text.
#         """
#         return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

#     model = "qwen3:1.7b"
#     url = "http://localhost:11434/api/generate"
#     headers = {"Content-Type": "application/json"}
#     payload = {
#         "model": model,
#         "prompt": prompt,
#         "stream": False  
#     }

#     response = requests.post(url, headers=headers, data=json.dumps(payload))
#     if response.status_code != 200:
#         raise Exception(f"LLM 호출 실패: {response.status_code}, {response.text}")

#     result = response.json()
    
#     print(model, "완료")
#     return remove_think_tags(result["response"])

#######################  1단계 : code 생성 ########################
def generate_code_prompt(user_query: str, df_preview: dict, df_types: dict) -> str:
    print("📌 df 타입정보")
    print(json.dumps(df_types, ensure_ascii=False, indent=2))

    # dict → pretty JSON string
    preview_str = json.dumps(df_preview, ensure_ascii=False, indent=2)
    types_str = json.dumps(df_types, ensure_ascii=False, indent=2)
    prompt = f"""
    다음은 pandas DataFrame(df)의 미리보기입니다:
    {preview_str}

    각 컬럼의 데이터 타입은 다음과 같습니다:
    {types_str}

    다음 사용자 질의에 기반하여 관련 정보를 추출하는 Python 코드를 생성하세요:
    "{user_query}"

    코드는 `df`가 이미 로드되어 있다고 가정하고, 최종 결과는 새로운 DataFrame `final_df`로 반환되어야 합니다.

    단, 사용자 질의가 단일 값을 묻는 질문(예: 최대값, 최소값, 상위 1개 등)이라 하더라도,
    `final_df`에는 관련된 전체 맥락이 담겨야 합니다.
    예를 들어, "가장 층이 높은 행정구는?"이라는 질문이라면,
    해당 컬럼을 기준으로 정렬된 모든 행정구 정보를 포함한 DataFrame을 반환해야 합니다.

    즉, 단일 값만 추출하지 말고, 사용자의 질문에 대한 비교/정렬/비율 등의 추가적이고 관련 있는 정보를 함께 포함하세요.

    생성된 코드는 <result></result> XML 태그 안에 작성해주세요.
    import문이나 print문은 포함하지 마세요.

    ## 응답 예시
    <result>
    sorted_df = df.groupby("행정구")["층수"].max().reset_index()
    sorted_df = sorted_df.sort_values(by="층수", ascending=False)
    final_df = sorted_df
    </result>
    """
    return prompt

#######################  2단계 : code 추출 및 실행 ########################


def extract_code_from_response(response: str) -> str:
    # 1. <result> 태그 우선 추출
    match = re.search(r"<result>(.*?)</result>", response, re.DOTALL)
    if match:
        code_block = match.group(1)
        code_block = re.sub(r"```[a-zA-Z]*", "", code_block).strip()
        return code_block

    # 2. <result> 태그가 없으면, 마크다운 코드블럭에서 추출
    match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return ""

def execute_generated_code(code: str, df: pd.DataFrame, max_retries: int = 3):
    current_code = code
    error_history = []
    
    for attempt in range(max_retries):
        try:
            local_vars = {"df": df}
            exec(current_code, {}, local_vars)
            return local_vars.get("final_df", None)
        except Exception as e:
            error_message = str(e)
            error_history.append(f"Attempt {attempt + 1} failed: {error_message}")
            
            if attempt < max_retries - 1:  # 마지막 시도에는 새로운 코드 생성하지 않음
                # 코드 수정을 위한 새로운 프롬프트 생성
                error_prompt = f"""
                다음 코드에서 오류가 발생했습니다:
                {current_code}
                
                오류 메시지:
                {error_message}
                
                이전에 발생한 오류들:
                {chr(10).join(error_history[:-1])}
                
                1. 모든 데이터 타입 변환 문제 처리
                2. 누락되거나 유효하지 않은 값 처리
                3. 적절한 데이터 타입 처리 사용
                4. 'final_df'라는 이름의 필터링된 DataFrame 또는 피벗 테이블 반환
                5. 동일한 문제가 반복되지 않도록 이전 오류 고려
                
                수정된 코드는 <result></result> XML 태그 안에 작성해주세요.
                """
                
                # LLM에서 수정된 코드 추출
                corrected_response = llm_call(error_prompt)
                corrected_code = extract_code_from_response(corrected_response)
                
                if corrected_code:
                    current_code = corrected_code
                else:
                    return f"코드 수정에 실패했습니다. 오류 기록: {chr(10).join(error_history)}"
            else:
                return f"최대 시도 횟수({max_retries})에 도달했습니다. 오류 기록: {chr(10).join(error_history)}"
    
    return f"예상치 못한 오류: {chr(10).join(error_history)}"


#######################  3단계 : 답변 생성 ########################


def generate_final_prompt(user_query: str, filtered_df: pd.DataFrame) -> str:
    try:
        filtered_json = json.dumps(json.loads(filtered_df.to_json()), ensure_ascii=False)
    except Exception as e:
        filtered_json = "{}"  # fallback in case of an error
    
    context_json = json.dumps({"query": user_query, "data": filtered_json}, ensure_ascii=False)
    prompt = f"""
    다음 컨텍스트가 주어졌습니다:
    {context_json}
    주어진 데이터를 기반으로 질문에 대한 답변을 제공해주세요. 답변은 명확하고 간결해야 하며, 불필요한 포맷팅이나 인코딩 문제가 없어야 합니다.
    - 답변은 한국어로 작성해주세요.
        """
    return prompt


def main():
    st.title("내 엑셀데이터와 대화하기")
    uploaded_file = st.file_uploader("파일 업로드", type=["xls", "xlsx", "csv"])
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'csv':
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        with st.expander("데이터 미리보기(사람용)"):
            st.dataframe(df.head(5))

        df_preview = df.head(5).to_dict(orient="records")
        with st.expander("데이터 미리보기(LLM용)"):
            st.json(df_preview)
        
        df_types = df.dtypes.apply(lambda x: str(x)).to_dict()
        with st.expander("데이터타입 미리보기(LLM용)"):
            st.json(df_types)
        
        questions = [
            "서울에서 업종별(대분류)로 가장 많은 상점이 있는 구는 어디인가요?",
            "서울에서 카페가 위치한 평균 층수가 가장 높은 구는 어디인가요?", 
            "서울에서 부동산 중개업이 전체 상가에서 차지하는 비중이 가장 높은 지역은 어디인가요?",
            "성동구에서 업종별(중분류) 상점 비중은 어떻게 되나요?",
            "서울에서 음식점 업종이 각 행정구별로 층별 분포가 어떻게 되어 있는지 알고 싶어요. 모든 행정구에 대해서 궁금해요. 예를 들어, 성동구는 1층이 몇프로 2층이 몇프로, 강남구는 3층이 몇프로 등."
        ]

        if "user_query" not in st.session_state:
            st.session_state["user_query"] = ""

        user_input = st.text_input(
            "데이터에 대해 질문해주세요:",
            key="input_box"
        )

        if not user_input:
            sample = st.selectbox("또는 예시 질문을 선택해주세요:", questions, key="sample_box")
            user_query = sample
        else:
            user_query = user_input

        if st.button("질문하기"):
            st.session_state["user_query"] = user_query
            if user_query:
                code_prompt = generate_code_prompt(user_query, df_preview, df_types)
                print("생성된 코드 프롬프트")
                print(code_prompt)
                generated_response = llm_call(code_prompt)
                print("생성된 코드")
                print(generated_response)   
                generated_code = extract_code_from_response(generated_response)
                print("생성된 코드 추출")
                print(generated_code)
                
                filtered_df = execute_generated_code(generated_code, df)
                
                if isinstance(filtered_df, pd.DataFrame):
                    final_prompt = generate_final_prompt(user_query, filtered_df)
                    final_response = llm_call(final_prompt)
                    
                    st.write("### 답변")
                    st.write(final_response)
                    
                    with st.expander("답변 근거"):
                        st.write("### 생성된 코드")
                        st.code(generated_code, language="python")

                        st.write("### 필터링된 데이터") 
                        st.dataframe(filtered_df)
                                                
                        st.write("### 최종 질문 프롬프트")
                        st.code(final_prompt, language="json")

                else:
                    st.error(filtered_df)
            else:
                st.warning("질문을 입력해주세요.")

if __name__ == "__main__":
    main()
