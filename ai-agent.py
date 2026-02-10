
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent

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



# 현재 날씨를 알려주는 tool 함수 예시
@tool
def get_current_weather(location: str) -> str:
  """
  주어진 위치(location)의 현재 날씨 정보를 반환합니다.
  실제 구현에서는 외부 API 호출이 필요합니다.
  """
  # 예시 응답 (실제 구현 시 API 연동 필요)
  return f"{location}의 현재 날씨는 맑음, 22도입니다."
  


agent = create_agent(
    model=model,
    tools=[get_current_weather],)


result = agent.invoke({"messages": [
  {"role": "user", "content": "서울 날씨 어때"},
]})

print("Agent의 응답:", result)
print(result['messages'][-1].content)