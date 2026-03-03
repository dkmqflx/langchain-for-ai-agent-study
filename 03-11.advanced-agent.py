
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore

# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-2.5-flash-lite")


# store 생성
store = InMemoryStore()

user_id = "user_001"
application_context = "personal_assistant"


# namespace 정의
namespace = (user_id, application_context)

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

item = store.get(namespace, "memory_001")

print(f"저장된 메모리: {item.value}")

items = store.search(namespace)
print(f"저장된 모든 메모리: {items}")



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

item = store.get(namespace, "memory_002")
print("저장된 메모리:", item.value)


items = store.search(namespace)
print("저장된 모든 메모리:", items)