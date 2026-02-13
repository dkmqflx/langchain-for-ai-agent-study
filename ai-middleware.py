"""
================================================================================
LangChain Built-in Middleware 정리
================================================================================

[랭체인 빌트 인 미들웨어 종류]
- Summarization (대화 요약): 긴 대화 히스토리를 요약하여 컨텍스트 크기 최적화
- Human-in-the-loop (사용자 개입): 에이전트 실행 중 사용자 승인 필요 -> 작업 전 사용자의 의사를 묻는 미들웨어
- Model call limit (모델 호출 제한): API 호출 횟수 제어
- Tool call limit (도구 호출 제한): 각 도구 사용 횟수 제어
- Model fallback (대체 모델): 주 모델 실패 시 다른 모델 사용
- PII detection (개인정보 탐지): 민감한 정보 감지 및 필터링
- Todo list (할 일 목록): 작업 추적
  - https://docs.langchain.com/oss/python/langchain/middleware/built-in#to-do-list

[고급 Middleware]
- LLM tool selector (LLM 기반 도구 선택): 상황에 맞는 최적 도구 자동 선택
- Tool retry (도구 재시도): 실패한 도구 자동 재실행
- LLM tool emulator (LLM 도구 에뮬레이터): 실제 도구 없이 LLM으로 시뮬레이션, 
  - https://docs.langchain.com/oss/python/langchain/middleware/built-in#llm-tool-emulator
  - API 호출 비용 절감 및 개발/테스트 용이
- Context editing (컨텍스트 편집): 프롬프트/상태 동적 수정

[현재 코드에서 사용 중인 Middleware]
- InMemorySaver (Checkpointer): 메모리에 대화 상태와 스레드 정보를 저장하여 
  대화 히스토리 관리. 개발/테스트용으로 적합하며, 프로덕션에서는 
  SQLiteSaver 또는 PostgresSaver 권장.
================================================================================
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langchain.agents.middleware import PIIMiddleware 
from langchain.agents.middleware import SummarizationMiddleware


from langgraph.checkpoint.memory import InMemorySaver


# .env 파일에서 환경 변수 로드
load_dotenv()

# API Key 가져오기
api_key = os.getenv("GEMINI_API_KEY")
aladin_api_key = os.getenv("ALADIN_API_KEY")

if api_key:
    # 보안을 위해 키의 일부만 출력
    masked_key = f"{api_key[:8]}...{api_key[-4:]}"
    print(f"API Key가 성공적으로 로드되었습니다: {masked_key}")
else:
    print("API Key를 찾을 수 없습니다. .env 파일을 확인해주세요.")

model = init_chat_model("gemini-2.0-flash",
  model_provider="google_genai",
  api_key=os.getenv("GEMINI_API_KEY"))


checkpointer=InMemorySaver()





# Summarization Middleware
# This middleware is used to summarize long conversation histories to optimize context size.
# It is particularly useful when working with models that have a limited context window.
# By summarizing previous messages, the middleware ensures that the most relevant information
# is retained while staying within the model's context limit.
summarization_middleware = SummarizationMiddleware(
    model="gpt-4.1-mini",
    trigger=("tokens", 4000),
    keep=("messages", 20),
)

# Add the summarization middleware to the agent
agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        LLMToolEmulator(model=model),               # LLM Tool Emulator
        summarization_middleware        # Updated Summarization Middleware
    ],
)

# Summarization Middleware Test Example
print("\n4. Summarization Middleware Test:")
print("Original conversation:")
conversation = [
    {"role": "user", "content": "안녕하세요, 오늘 날씨가 참 좋네요."},
    {"role": "assistant", "content": "안녕하세요! 네, 오늘 날씨가 정말 좋습니다. 어떻게 도와드릴까요?"},
    {"role": "user", "content": "오늘 저녁에 뭐 먹을지 고민이에요. 추천해 주실 수 있나요?"},
    {"role": "assistant", "content": "물론이죠! 한식, 중식, 일식 중에 어떤 걸 드시고 싶으신가요?"},
    {"role": "user", "content": "한식이요!"},
    {"role": "assistant", "content": "그렇다면 비빔밥이나 불고기를 추천드려요."}
]

print("\nOriginal Messages:")
for msg in conversation:
    print(f"{msg['role']}: {msg['content']}")

result_summary = agent.invoke({"messages": conversation}, {"configurable": {"thread_id": 13}})
print("\nSummarized Conversation:")
print(result_summary['messages'][-1].content)
print("(Summarization Middleware applied to optimize context size)\n")



