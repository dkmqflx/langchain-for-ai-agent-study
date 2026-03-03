
import os
from typing import Any, List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
import requests
from langchain.agents.structured_output import ToolStrategy

from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import LLMToolEmulator
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import PIIMiddleware 
from langchain.agents.middleware import before_model
from dataclasses import dataclass

# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-2.5-flash-lite")


@dataclass
class Context:
  user_name: str

@before_model
def log_before_model(state, runtime):
  print("state", state)
  print("runtime", runtime)
  return None

agent = create_agent(
  model=model,
  tools=[],
  middleware=[log_before_model],
  context_schema=Context
)

result = agent.invoke({"messages": [{"role": "user", "content": "안녕하세요, 내 이름은 무엇인가요"}]}, context=Context(user_name="John Doe"))
# print(result)
# {'messages': [HumanMessage ... 전에 log_before_model 함수가 실행된 것을 확인할 수 있다 
# 그리고 node style hook으로, 단순히 로깅용으로 사용되고 있기 때문에,
# 언어모델은 내 이름을 모른다는 것도 알 수 있다