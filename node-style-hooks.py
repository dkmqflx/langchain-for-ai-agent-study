import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from dataclasses import dataclass
from langchain.agents.middleware import before_model
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

result = agent.invoke({"messages": [{"role": "user", "content": "안녕하세요, Guess my Name?"}]}, context=Context(user_name="John Doe"))
print(result)
# {'messages': [HumanMessage ... 전에 log_before_model 함수가 실행된 것을 확인할 수 있다 
# 그리고 node style hook이기 때문에 로그를 보면 내 이름을 모른다는 것도 알 수 있다