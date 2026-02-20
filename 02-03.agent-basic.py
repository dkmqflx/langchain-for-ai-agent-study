
import os
from typing import Any, List, Dict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
import requests

from langgraph.checkpoint.memory import InMemorySaver

# .env 파일에서 환경 변수 로드
load_dotenv()
aladin_api_key = os.getenv("ALADIN_API_KEY")

# 모델 선언
model = init_chat_model("google_genai:gemini-2.5-flash-lite")

# 메모리 종류 

# short term memory: Checkpointer를 통해 대화 스레드(세션) 저장
# long term memory: Store를 통해 장기 메모리 저장 

# 주의할 점은 invoke할 때 thread_id = 1를 명시적으로 지정해야 한다는 점입니다. 

agent = create_agent(
    model=model,
    tools=[],
    checkpointer=InMemorySaver()
    
    )

result = agent.invoke({"messages": [
  {"role": "user", "content": "안녕하세요, 나는 자유인입니다"}]},
  {"configurable": {"thread_id": 1}})

print(result['messages'][-1].content)


result = agent.invoke({"messages": [
  {"role": "user", "content": "제가 뭐라고 했죠 ?"}]},
  {"configurable": {"thread_id": 1}}) # thread_id가 다르면, 새로운 대화로 인식합니다.


print(result['messages'][-1].content)

