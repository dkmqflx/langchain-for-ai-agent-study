
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from dataclasses import dataclass
from langchain.agents.middleware import wrap_model_call
from langchain.agents import create_agent

# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-2.5-flash-lite")

# 실행 컨텍스트 정의 (누가 실행하는지 식별)
@dataclass
class Context:
    user_id: str
    app_name: str

# store 생성
store = InMemoryStore()

user_id = "user_001"
application_context = "personal_assistant"


# namespace 정의
namespace = (user_id, application_context)


# store에 데이터 저장
store.put(
    namespace,
    "memory_001",
    {
        "facts": [
            "사용자는 커피보다 차를 선호함",
            "사용자는 매일 아침 6시에 일어남",
        ],
        "language": "Korean",
    },
)

store.put(
    namespace,
    "memory_002",
    {
        "facts": [
            "사용자는 그림 회화 작품을 좋아함",
            "빈센트 반 고흐의 작품을 특히 좋아함",
        ]
    },
)

@wrap_model_call
def inject_memory(request, handler):
    current_user = request.runtime.context.user_id
    current_app = request.runtime.context.app_name
    memories = request.runtime.store.search((current_user, current_app))

    memory_content = "기록된 정보 없음"

    if memories:
        # 검색된 메모리들을 텍스트 변환
        extracted_facts = []
        for item in memories:
            if "facts" in item.value:
                extracted_facts.extend(item.value["facts"])
        memory_content = "\n- ".join(extracted_facts)

    system_message=f"사용자 관련 장기 메모리 : {memory_content}"
    request = request.override(system_prompt=system_message)
    # 해당 로직에서 장기 메모리를 system_prompt로 처리하는 이유는 "AI에게 사용자에 대한 배경지식을 주입하여 개인화된 답변을 유도하기 위함

    return handler(request)



agent = create_agent(
    model=model,
    store=store,          # store 연결
    context_schema=Context,
    middleware=[inject_memory]  # 메모리 주입 미들웨어
)


response = agent.invoke(
    {"messages": [{"role": "user", "content": "나에 대해 알고있는 정보 알려줘"}]},
    context=Context(user_id="user_001", app_name="personal_assistant") # user_id가 002로 바뀌면 사용자 정보를 알 수 없다고 답변한다 
)

print("response", response)