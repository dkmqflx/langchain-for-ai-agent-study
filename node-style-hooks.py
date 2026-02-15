import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from dataclasses import dataclass
from langchain.agents.middleware import wrap_model_call
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

@dataclass
class Context:
  user_name: str

# https://docs.langchain.com/oss/javascript/langchain/middleware/custom#wrap-style-hooks
@wrap_model_call
def wrap_model_call_hook(request, handler):
  print("request", request)
  user_name = request.runtime.context.user_name
  print("user_name", user_name)
  if user_name:
    system_message = f"You are a helpful assistant that knows the user's name is {user_name}"
    request = request.override(system_prompt=system_message)
  return handler(request)

agent = create_agent(
  model=model,
  tools=[],
  middleware=[wrap_model_call_hook],
  context_schema=Context
)

result = agent.invoke({"messages": [{"role": "user", "content": "안녕하세요, Guess my Name?"}]}, context=Context(user_name="John Doe"))
print(result)
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

