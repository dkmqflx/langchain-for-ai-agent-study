
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



# 사칙연산 tool 함수 정의
@tool
def add(a: int, b: int) -> float:
  """두 수의 합을 반환합니다."""
  return a + b

@tool
def divide(a: float, b: float) -> float:
  """a를 b로 나눈 값을 반환합니다. (b가 0이면 예외 발생)"""
  if b == 0:
    raise ValueError("0으로 나눌 수 없습니다.")
  return a / b

@tool
def multiply(a: float, b: float) -> float:
  """두 수의 곱을 반환합니다."""
  return a * b



# AI는 확률 기반으로 추론하는데, tool를 사용하지 않고도 답을 할 수 있다고 판단하게 되면 tools를 사용하지 않을 수 있다
# 이런 경우를 방지하려면, system 메시지에 반드시 tool를 사용하도록 지시하는 내용을 추가해야 합니다.
agent = create_agent(
    model=model,
    tools=[add, divide, multiply],
    system_prompt="당신은 계산기입니다. 반드시 도구를 사용하여 계산을 수행해야 합니다. 절대 도구를 사용하지 않고 대답하지 마세요.")


result = agent.invoke({"messages": [
  {"role": "user", "content": "32 + 5 * 100를 계산해줘."},
]})

print("Agent의 응답:", result) 

