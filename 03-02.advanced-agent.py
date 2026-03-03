
import os
from typing import Any, List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
import requests
from langchain.agents.structured_output import ToolStrategy

from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import LLMToolEmulator
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import PIIMiddleware 
from langchain.agents.middleware import before_model
from dataclasses import dataclass
from langchain.agents.middleware import wrap_model_call

# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-2.5-flash-lite")


@dataclass
class Context:
  user_name: str

@wrap_model_call
def wrap_model_call_hook(request, handler):
  print("request", request)
  print("handler", handler)

  user_name = request.runtime.context.user_name
  print("user_name", user_name)

  if user_name:
    system_message = f"사용자의 이름은 {user_name} 입니다."
    request = request.override(system_prompt=system_message)
  return handler(request)

agent = create_agent(
  model=model,
  tools=[],
  middleware=[wrap_model_call_hook],
  context_schema=Context
)

result = agent.invoke({"messages": [{"role": "user", "content": "안녕하세요, 제 이름은 무엇일까요?"}]}, context=Context(user_name="John Doe"))
print('result', result)
# 이번에는 이름을 제대로 추측하는 것을 확인할 수 있다


# ## 1. Node-style Hook
# 입력 파라미터: state, runtime
# **"노드 실행 중 특정 지점에서 순차적으로 동작하는 방식"**

# 공식 문서에서 로깅, 검증, 상태 업데이트에 이 방식을 추천하는 이유는 노드 내부의 **상세한 맥락(Context)**에 직접 접근할 수 있기 때문입니다.

# * **동작 원리:** 노드 함수가 실행되는 도중, 코드에 명시된 특정 지점(Execution Points)에서 훅이 호출됩니다.
# * **주요 특징:**
# * **순차 실행:** 노드 로직과 함께 위에서 아래로 흐름을 같이 합니다.
# * **내부 접근:** 노드 안의 로컬 변수나 인자값을 즉시 검사(Validation)하고 기록(Logging)하기 좋습니다.


# * **공식 문서 권장 용도:**
# * **Validation:** 입력 데이터가 비즈니스 로직에 들어가기 직전 검사.
# * **Logging:** 노드 내부에서 발생하는 상세 작업 단계 기록.
# * **State Updates:** 실행 결과에 따라 즉각적으로 상태를 변경해야 할 때.



# ## 2. Wrap-style Hook
# 입력 파라미터: request, handler
# model이나 tool을 호출될 때 실행과 제어를 가로채서 원하는 작업을 수행할 수 있습니다.

# 공식 문서에서 정의한 Wrap-style의 용도는 크게 세 가지입니다.

# ① 단락 실행 (Short-circuit / Zero times)

# 실제 도구나 노드를 단 한 번도 실행하지 않고 결과를 돌려주는 기능입니다.
# 적용 사례: Caching (캐싱). 이미 똑같은 질문에 대한 답이 메모리에 있다면, 비싼 LLM이나 도구를 호출하지 않고 저장된 값을 즉시 반환합니다.
# 적용 사례: Emulation. 실제 API를 호출하는 대신 가짜 데이터를 반환할 때 사용합니다.


# ② 정상 흐름 (Normal flow / Once)

# 일반적인 실행이지만, 입력값이나 출력값을 가공(Transformation)할 때 사용합니다.
# 적용 사례: Transformation (변환). 도구에 들어가기 전 인자값을 보안 처리하거나, 도구가 내뱉은 복잡한 JSON을 에이전트가 읽기 쉬운 텍스트로 변환합니다.


# ③ 반복 실행 (Retry logic / Multiple times)

# 도구가 실패했을 때, 성공할 때까지 여러 번 다시 시도하게 만드는 기능입니다.
# 적용 사례: Retries (재시도). 네트워크 오류로 API 호출이 실패하면, 노드 내부 로직과 상관없이 미들웨어 수준에서 3번 더 시도하도록 강제합니다


# wrap_model_call - Around each model call
# wrap_tool_call - Around each tool call


# handler 란 

#  handler는 "실제 모델(LLM) 호출을 실행하는 함수" 또는 "다음 실행 단계로 넘겨주는 함수"를 의미합니다.
# @wrap_model_call 데코레이터가 붙은 미들웨어(hook)는 모델이 호출되기 직전에 실행을 가로챕니다. 이때 handler를 인자로 받는데, 이 handler를 호출해야만 비로소 실제 LLM(여기서는 Gemini 모델)이 실행됩니다.


# @wrap_model_call
# def wrap_model_call_hook(request, handler):
  # 1. 모델 호출 전 처리 (예: 프롬프트 수정)
  # ...
  
  # 2. 실제 모델 호출 실행 (handler 호출)
  # result = handler(request) 
  
  # 3. 모델 호출 후 처리 (예: 결과 가공)
  # return result


# request 란 

# request는 "모델(LLM)이나 도구(Tool)를 호출하기 위해 필요한 모든 정보를 담고 있는 데이터 꾸러미"입니다.