
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model



# langchain 메시지 객체 임포트
from langchain.messages import HumanMessage, SystemMessage

# .env 파일에서 환경 변수 로드
load_dotenv()

# API Key 가져오기
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    # 보안을 위해 키의 일부만 출력
    masked_key = f"{api_key[:8]}...{api_key[-4:]}"
    print(f"API Key가 성공적으로 로드되었습니다: {masked_key}")
else:
    print("API Key를 찾을 수 없습니다. .env 파일을 확인해주세요.")

model = init_chat_model("gemini-2.0-flash",
  model_provider="google_genai",
  api_key=os.getenv("GEMINI_API_KEY"))



# SystemMessage와 HumanMessage를 사용해 대화 구성


# 메시지를 딕셔너리로 정의 (role, content)
messages_data = [
  {"role": "system", "content": "You are a value investment investor."},
  {"role": "human", "content": "Broadcom price will be increased?"},
  {"role": "human", "content": "My favorite color is blue."},
  {"role": "human", "content": "What color did I say is my favorite?"},
  {"role": "human", "content": "Remind me what my first question was."}
]



response = model.invoke(messages_data)
print(response)


