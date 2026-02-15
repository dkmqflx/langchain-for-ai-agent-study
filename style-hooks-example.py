import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from dataclasses import dataclass
from langchain.agents.middleware import wrap_model_call
# .env 파일에서 환경 변수 로드
load_dotenv()

# API Key 가져오기
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    # 보안을 위해 키의 일부만 출력
    masked_key = f"{api_key[:8]}...{api_key[-4:]}"
    print(f"API Key가 성공적으로 로드되었습니다: {masked_key}")
else:
    print("API Key를 찾을 수 없습니다. .env 파일을 확인해주세요.")

model = init_chat_model("gemini-2.0-flash",
  model_provider="google_genai",
  api_key=os.getenv("GEMINI_API_KEY"))

@dataclass
class Context:
  user_name: str
  is_premium: bool = False

@wrap_model_call
def dynamic_model_selection_hook(request, handler):
  print("request", request)
  user_name = request.runtime.context.user_name
  is_premium = request.runtime.context.is_premium
  print("user_name", user_name)
  print("is_premium", is_premium)

  if is_premium:
    model_name = "gemini-2.0-flash"
    print("Using premium model")

  else:
    model_name = "gemini-2.0-flash"
    print("Using free model")

  new_model = init_chat_model(
    model_name,
    model_provider="google_genai",
    api_key=os.getenv("GEMINI_API_KEY")
  )
  new_request = request.override(model=new_model)

  return handler(new_request)

agent = create_agent(
  model=model,
  tools=[],
  middleware=[dynamic_model_selection_hook],
  context_schema=Context
)

result = agent.invoke({"messages": [{"role": "user", "content": "안녕하세요, Guess my Name?"}]}, context=Context(user_name="John Doe", is_premium=True))
print(result)