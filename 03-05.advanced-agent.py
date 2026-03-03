
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.agents.middleware import before_agent
from langchain.messages import AIMessage
from langchain.agents.middleware import wrap_model_call

# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-2.5-flash-lite")


@dataclass
class Context:
  user_name: str

@wrap_model_call
def dynamic_model_selection_hook(request, handler):

  last_message = request.messages[-1].content if request.messages else ""
  message_len = len(last_message)

  if message_len > 10:
    new_model = init_chat_model("google_genai:gemini-2.5-flash-lite")
    request = request.override(model=new_model)
  else:
    new_model = init_chat_model("google_genai:gemini-2.0-flash")
    request = request.override(model=new_model)

  print(f"Using model: {new_model.model}, message length: {message_len}")

  return handler(request)

agent = create_agent(
  model=model,
  tools=[],
  middleware=[dynamic_model_selection_hook],
  context_schema=Context
)

result1 = agent.invoke({"messages": [{"role": "user", "content": "이것은 10자가 넘는 텍스트입니다."}]})
print('result1', result1)





