
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent


# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-2.5-flash-lite")

# 현재 날씨를 알려주는 tool 함수 예시
@tool
def get_current_weather(location: str) -> str:
  """
  주어진 위치(location)의 현재 날씨 정보를 반환합니다.
  실제 구현에서는 외부 API 호출이 필요합니다.
  """
  # 예시 응답 (실제 구현 시 API 연동 필요)
  return f"{location}의 현재 날씨는 맑음, 22도입니다."
  
agent_with_weather = create_agent(
    model=model,
    tools=[get_current_weather],)


result = agent_with_weather.invoke({"messages": [
  {"role": "user", "content": "서울 날씨 어때"},
]})

print("Agent의 응답:", result)
print(result['messages'][-1].content)


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
agent_with_calculator = create_agent(
    model=model,
    tools=[add, divide, multiply],
    system_prompt="당신은 계산기입니다. 반드시 도구를 사용하여 계산을 수행해야 합니다. 절대 도구를 사용하지 않고 대답하지 마세요.")


result = agent_with_calculator.invoke({"messages": [
  {"role": "user", "content": "32 + 5 * 100를 계산해줘."},
]})

print("Agent의 응답:", result) 