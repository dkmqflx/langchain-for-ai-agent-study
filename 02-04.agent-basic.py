
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

# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-3-flash-preview")


@tool
def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """
    가상의 이메일 전송 도구
    """
    return f"이메일이 {recipient}에게 전송되었습니다. 제목: {subject}"

@tool
def read_email_tool(email_id: str) -> str:
    """
    가상의 이메일 읽기 도구
    """
    return f"이메일 {email_id}의 내용: 안녕하세요, 이것은 테스트 이메일입니다."


agent = create_agent(
    model=model,
    tools=[send_email_tool, read_email_tool],
    # tool emulator: 실제 도구 호출 대신 LLM으로 시뮬레이션, 즉, 도구가 만든 결과를 모킹하는것. gemini-2.5-flash 모델 사용
    middleware=[LLMToolEmulator(model="google_genai:gemini-2.5-flash-lite")],
    )

# # 이메일 읽기 예시
result = agent.invoke({"messages": [
  {"role": "user", "content": "최근 도착한 이메일을 읽고 답장해줘"}]},
  {"configurable": {"thread_id": 1}})

print("=== 이메일 읽고 쓰기 결과 ===")
print(result)
print(result['messages'][-1].content)

