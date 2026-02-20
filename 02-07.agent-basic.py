
import os
from typing import Any, List, Dict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
import requests


from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import LLMToolEmulator
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import PIIMiddleware 

# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-3-flash-preview")


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
email_pii_middleware = PIIMiddleware("email", strategy="redact", apply_to_input=True,)

# 2. 신용카드 번호 감지 및 mask 전략
credit_card_pii_middleware = PIIMiddleware("credit_card", strategy="mask", apply_to_input=True,)

# 3. 사용자정의 타입 - 회원번호 형식 (USER-12345)
custom_pii_middleware = PIIMiddleware(
    pii_type="user_id",  # 커스텀 타입명
    detector=r"USER-\d{5}",  # 감지 패턴
    strategy="mask",  # 필터링 전략
    apply_to_input=True
)

checkpointer=InMemorySaver()


agent = create_agent(
    model=model,
    tools=[save_book_info],
    middleware=[email_pii_middleware, credit_card_pii_middleware, custom_pii_middleware],
    )

# # 테스트: 이메일 주소 포함 메시지
print("\n1. 이메일 감지 테스트 (redact):")
result_email = agent.invoke(
    {"messages": [{"role": "user", "content": "제 이메일은 john.doe@gmail.com입니다.",}]})
print(result_email)
# print("결과:", result_email['messages'][-1].content)


# 테스트: 신용카드 번호 포함 메시지
print("2. 신용카드 번호 감지 테스트 (mask):")
result_card = agent.invoke({"messages": [
    {"role": "user", "content": "제 신용카드 번호는 4532-1234-5678-9010 입니다"}]},
    {"configurable": {"thread_id": 11}})
print(result_card)
print("결과:", result_card['messages'][-1].content)


# 테스트: 커스텀 타입 (회원번호) 포함 메시지
print("3. 커스텀 타입 감지 테스트 (mask) - 회원번호:")
result_user_id = agent.invoke({"messages": [
    {"role": "user", "content": "회원번호는 USER-12345입니다"}]},
    {"configurable": {"thread_id": 12}})
print(result_user_id)
print("결과:", result_user_id['messages'][-1].content)


## 참고, 
# apply_to_output=True 옵션을 사용하면, 출력 메시지에도 PII가 필터링 되는 것 확인할 수 있지만
# email, credit_card,타입에 대해서는 필터링 되지 않는 것을 확인할 수 있다.
# 커스텀 타입에 대해서는 필터링 되는 것을 확인할 수 있다.