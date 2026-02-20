
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


checkpointer=InMemorySaver()


agent = create_agent(
    model=model,
    tools=[send_email_tool, read_email_tool],
    checkpointer=checkpointer,
    middleware=[LLMToolEmulator(model=model), HumanInTheLoopMiddleware(
        interrupt_on={
            "send_email_tool": {'allowed_decisions': ['approve', 'reject', 'edit']}, 
            "read_email_tool": False}
    ) ],
    )

# https://docs.langchain.com/oss/python/langchain/human-in-the-loop    

# # 이메일 읽기 예시
# result1 = agent.invoke({"messages": [
#   {"role": "user", "content": "최근 도착한 메일을 확인해줘"}]},
#   {"configurable": {"thread_id": 1}})

# print("=== 이메일 읽기 결과 ===")
# print(result1)
# print(result1['messages'][-1].content)




# # 이메일 쓰기 예시 1
result2 = agent.invoke({"messages": [
  {"role": "user", "content": "내일 방문할 예정인 고객에게 이메일을 작성해줘"}]},
  {"configurable": {"thread_id": 2}})

print("=== 이메일 쓰기 결과 ===")
print(result2) 
# 이메일 읽기와 달리 tool_call이 호출되지 않은 것을 확인할 수 있다 
# 사용자가 입력한 정보가 충분하지 않기 때문에 LLM이 tool을 호출하지 않는다
print(result2['messages'][-1].content)


# # 이메일 쓰기 예시 2
result3 = agent.invoke({"messages": [
  {"role": "user", "content": "내일 방문할 Jasper의 이메일 주소는 jasper@gmail.com 입니다. subject: 오늘 방문 예정, body: 오늘 방문 예정입니다. 이메일을 작성해줘"}]},
  {"configurable": {"thread_id": 2}})

print("=== 이메일 쓰기 결과 ===")
print(result3) 
# 위 예시와 달리, 상세한 정보를 전달해주자 tool_call이 호출되는 것을 확인할 수 있다 
print(result3['messages'][-1].content)

# invoke는 바로 종료되고, 결과에 __interrupt__가 들어 있는 상태로 반환됩니다. 사용자 입력을 기다리며 blocking 되지 않습니다.
# 그래서:
# 첫 번째 invoke: 에이전트가 send_email_tool을 호출하려다 HumanInTheLoop에서 멈추고, __interrupt__와 함께 바로 결과를 돌려줌
# 지금 상태: send_email_tool은 아직 실행되지 않았고, action_requests에 승인 대기 중인 도구 호출만 담겨 있음
# 실제 도구 실행을 하려면: 같은 thread_id로 사용자 결정(approve/reject/edit)을 포함해서 invoke를 한 번 더 호출해야 함