
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




@tool
def save_book_info():
    """주어진 책 제목을 저장하는 도구"""
    # 실제로는 데이터베이스나 파일에 저장하는 로직이 들어감
    return f"책  정보가 저장되었습니다."

# ================================================================================
# PII Detection Middleware 예시
# ================================================================================
# PII(Personal Identifiable Information) 감지 및 필터링

# 1. 이메일 주소 감지 및 redact 전략
email_pii_middleware = PIIMiddleware("email", strategy="redact", apply_to_input=True)

# 2. 신용카드 번호 감지 및 mask 전략
credit_card_pii_middleware = PIIMiddleware("credit_card", strategy="mask", apply_to_input=True,)

# 3. 사용자정의 타입 - 회원번호 형식 (USER-12345)
custom_pii_middleware = PIIMiddleware(
    pii_type="user_id",  # 커스텀 타입명
    detector=r"USER-\d{5}",  # 감지 패턴
    strategy="mask",  # 필터링 전략
    apply_to_input=True
)


agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        LLMToolEmulator(model=model),               # LLM Tool Emulator
    email_pii_middleware,           # 이메일 감지 (redact)
    credit_card_pii_middleware,     # 신용카드 감지 (mask)
    custom_pii_middleware,          # 회원번호 감지 (mask)
    ],

)




print("\n" + "="*80)
print("PII Detection Middleware 예시")
print("="*80)

# 테스트: 이메일 주소 포함 메시지
print("\n1. 이메일 감지 테스트 (redact):")
print("원본 메시지: 제 이메일은 john.doe@gmail.com 입니다")
result_email = agent.invoke({"messages": [
    {"role": "user", "content": "제 이메일은 john.doe@gmail.com입니다"}]},
    {"configurable": {"thread_id": 10}})
print(result_email)
print("결과:", result_email['messages'][-1].content)
print("(PII Detection으로 이메일이 필터링됨)\n")

# 테스트: 신용카드 번호 포함 메시지
print("2. 신용카드 번호 감지 테스트 (mask):")
print("원본 메시지: 제 카드번호는 4532-1234-5678-9010입니다")
result_card = agent.invoke({"messages": [
    {"role": "user", "content": "제 신용카드 번호는 4532-1234-5678-9010 입니다"}]},
    {"configurable": {"thread_id": 11}})
print(result_card)
print("결과:", result_card['messages'][-1].content)
print("(PII Detection으로 신용카드 번호가 필터링됨)\n")

# 테스트: 커스텀 타입 (회원번호) 포함 메시지
print("3. 커스텀 타입 감지 테스트 (mask) - 회원번호:")
print("원본 메시지: 회원번호는 USER-12345입니다")
result_user_id = agent.invoke({"messages": [
    {"role": "user", "content": "회원번호는 USER-12345입니다"}]},
    {"configurable": {"thread_id": 12}})
print(result_user_id)
print("결과:", result_user_id['messages'][-1].content)
