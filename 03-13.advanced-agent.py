
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from langchain.messages import AIMessage
from langchain.messages import HumanMessage
from langchain.messages import SystemMessage
from dataclasses import dataclass
from langgraph.store.memory import InMemoryStore

# re는 Regular Expression(정규 표현식)의 약자로, 파이썬의 표준 라이브러리

# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-2.5-flash-lite")

### Tool 기반 장기 메모리 관리


# 실행 컨텍스트 정의 (누가 실행하는지 식별)
@dataclass
class Context:
    user_id: str
    app_name: str

# Store 초기화
store = InMemoryStore()



from typing import TypedDict, Annotated, Any

# LLM이 추출해야 할 정보의 구조를 정의 (사용자의 request로 부터 추출해야 할 정보의 구조를 정의)
# 즉, LLM이 request로 부터 정보를 추출한다음, 해당 정보를 Tool에 전달해야하는 경우 전달한다
class UserInfo(TypedDict):
    personal_info: str
    preference: str



from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langchain.tools import tool


# Tool 정의: 사용자의 정보를 조회
@tool
def get_user_info(runtime: Annotated[Any, InjectedToolArg]) -> str:
    """
    현재 사용자의 정보 조회 (시스템 내부용 도구)
    """
    user_id = runtime.context.user_id
    app = runtime.context.app_name

    # 해당 네임스페이스의 모든 메모리 검색
    memories = runtime.store.search((user_id, app))

    if not memories:
        return "기록된 정보 없음"

    results = []
    for item in memories:
        # 저장된 데이터 구조(UserInfo)에 맞춰 필드 확인
        data = item.value
        # 저장할 때 사용한 Key들을 확인해서 문자열로 변환
        if "personal_info" in data:
            results.append(f"- 개인정보: {data['personal_info']}")
        if "preference" in data:
            results.append(f"- 선호도: {data['preference']}")

    return "\n".join(results) if results else "데이터 형식 불일치로 읽을 수 없음"


# LangChain의 @tool 데코레이터를 사용할 때, 함수의 인자가 어떻게 결정되고 전달되는지에 대해 설명해 드리겠습니다.

# 결론부터 말씀드리면, LLM(모델)이 함수의 "이름", "설명(Docstring)",
# 그리고 "인자의 타입 힌트"를 보고 어떤 값을 넣을지 스스로 판단하여 전달합니다.

# 1. 명시적 인자가 있는 경우 (02-08.agent-basic.py)
# @tool
# def get_weather(city: str) -> str:
#    """Get the current weather for a city."""
#    return f"The weather in {city} is sunny and 72°F"

# LLM의 판단: 모델은 이 도구의 설명을 보고 "도시 이름을 넣어야겠구나"라고 이해합니다.
# 전달 방식: 사용자가 "서울 날씨 어때?"라고 물으면, LLM은 내부적으로 {"city": "Seoul"}이라는 JSON 데이터를 생성하고, LangChain은 이를 get_weather(city="Seoul") 형태로 호출합니다.
# 핵심: city: str이라는 타입 힌트와 Get the current weather for a city라는 설명이 LLM에게 가이드 역할을 합니다.


# 2. 특수한 인자가 있는 경우 (위 get_user_info 함수)
# 이 runtime 인자에 무엇이 전달될지는 프레임워크의 '인자 주입(Injection)' 규칙에 의해 결정됩니다.

# 1) LLM은 이 인자의 존재를 모름: @tool 데코레이터가 LLM에게 보내는 도구 설명서에서 'runtime'을 자동으로 제외합니다.
# 2) 시스템이 자동 주입: 프레임워크(LangGraph/LangChain)가 도구를 실행할 때, 인자 이름이 'runtime'인 것을 보고 
#    자신이 관리하는 시스템 객체(State, Store 등)를 자동으로 끼워 넣어 호출합니다.



import uuid

@tool
def save_user_info(user_info: UserInfo, runtime: Annotated[Any, InjectedToolArg]):
    """
    사용자의 정보를 저장하거나 업데이트
    """
    # 1. 실행 컨텍스트에서 user_id 가져오기
    user_id = runtime.context.user_id
    app = runtime.context.app_name
    store = runtime.store

    # 2. Store에 데이터 저장 (put(namespace, key, value))
    memory_key = str(uuid.uuid4())
    store.put((user_id, app),memory_key, user_info)

    return f"정보가 안전하게 저장되었습니다. (ID: {memory_key})"



from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[get_user_info, save_user_info],
    store=store, # 에이전트에 store 연결
    context_schema=Context
)

response = agent.invoke(
    {"messages": [
        {"role": "system", "content": "당신은 사용자의 개인 비서입니다. 사용자가 자신의 이름, 선호도, 개인정보를 공유하면 즉시 save_user_info 도구를 호출하여 저장하세요. 저장 후 확인 메시지를 전달하세요."},
        # 모델 차이로 인해 위 코드가 필요할 수 있음
        # 강의에서 사용하는 gpt의 경우 위 system 메세지가 없어도 알아서 tool을 호출함
        {"role": "user", "content": "내 이름은 이제 'Alice'야. 커피보단 차를 좋아해"},
    ]},
    context=Context(user_id="user_001", app_name="personal_assistant")
)

print("response1", response)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "나에 대해 아는 정보 말해줘"}]},
    context=Context(user_id="user_001", app_name="personal_assistant")
)

print("response2", response)