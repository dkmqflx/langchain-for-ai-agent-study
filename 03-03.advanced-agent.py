
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

from langchain.agents.middleware import wrap_model_call


# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-2.5-flash-lite")


@dataclass
class Context:
  user_name: str
  is_premium: bool

@wrap_model_call
def dynamic_model_selection_hook(request, handler):
  print("request", request)
  user_name = request.runtime.context.user_name
  is_premium = request.runtime.context.is_premium
  print("user_name", user_name)
  print("is_premium", is_premium)

  if is_premium:
    model_name = "google_genai:gemini-2.5-flash-lite"
    print("Using premium model")

  else:
    model_name = "google_genai:gemini-2.0-flash"
    print("Using free model")

  new_model = init_chat_model(model_name)
  new_request = request.override(model=new_model)

  return handler(new_request)

agent = create_agent(
  model=model,
  tools=[],
  middleware=[dynamic_model_selection_hook],
  context_schema=Context
)

result = agent.invoke({"messages": [{"role": "user", "content": "안녕하세요?"}]}, context=Context(user_name="John Doe", is_premium=True))
print('result', result) # is_premium 값에 따라 모델이 달라진다