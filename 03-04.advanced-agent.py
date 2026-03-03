
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.agents.middleware import before_agent
from langchain.messages import AIMessage
# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-2.5-flash-lite")


@dataclass
class Context:
  user_name: str

@before_agent(can_jump_to=["end"])
def before_agent_hook(state, runtime):
  print("state", state)
  print("runtime", runtime)

  if "암구호" in state["messages"][-1].content:
    return {
      "messages": [AIMessage("오늘의 암구호를 알려줄 수 없습니다.")],
      "jump_to": "end"
    }

  return None

agent = create_agent(
  model=model,
  tools=[],
  middleware=[before_agent_hook],
  context_schema=Context
)

result1 = agent.invoke({"messages": [{"role": "user", "content": "오늘의 암구호는 무엇인가요 ?"}]})
print('result1', result1)

result2 = agent.invoke({"messages": [{"role": "user", "content": "오늘의 날씨는 어때요 ?"}]})
print('result2', result2)



# before_agent

# 공식 문서의 시그니처(dict[str, Any] | None)만으로는 구체적인 형태를 알기 어려울 수 있습니다. before_agent 훅(Hook)에서 return 문이 가져야 할 형태와 역할에 대해 명확히 설명해 드리겠습니다.

# 1. 핵심 역할: 상태(State) 업데이트
# before_agent 훅의 목적은 에이전트가 본격적으로 실행되기 직전에 현재 상태(State)를 수정하거나 정보를 추가하는 것입니다. 
# 따라서 return 문은 업데이트하고 싶은 상태의 딕셔너리(Dictionary) 형태여야 합니다.


# 2. 구체적인 return 형태

# A. 상태를 업데이트할 때 (dict[str, Any])
# 현재 에이전트가 관리하는 상태(State)의 키(key)와 업데이트할 값(value)을 담은 딕셔너리를 반환합니다.


# def before_agent(self, state, runtime):
    # 예: 에이전트가 실행되기 전에 특정 메시지를 강제로 추가하고 싶을 때
#    return {
#        "messages": state["messages"] + [HumanMessage(content="추가 정보: ...")]
#    }

    # 예: 특정 플래그(Flag)를 설정하고 싶을 때
#    return {
#        "is_ready": True
#    }



# B. 아무것도 변경하지 않을 때 (None)
# 상태를 변경할 필요가 없다면 None을 반환하거나, 아무것도 반환하지 않으면(Implicitly None) 됩니다.

# def before_agent(self, state, runtime):
#    # 로깅만 하고 상태는 그대로 유지
#    print(f"현재 상태: {state}")
#    return None  # 또는 생략



# 3. 주의사항: 병합(Merging) 방식
# before_agent에서 반환된 딕셔너리는 기존 상태와 병합(Merge)됩니다.
# 만약 상태 정의에서 messages 필드가 Annotated[list, add_messages]와 같이 리듀서(Reducer)를 사용하고 있다면, return {"messages": [...]}를 했을 때 기존 메시지에 추가됩니다.
# 일반적인 필드라면 기존 값이 반환된 값으로 덮어씌워집니다.



# -------------------------------------------------------------

# 02-07.agent-basic.py 과 달리 custom hook을 사용하고 있습니다.