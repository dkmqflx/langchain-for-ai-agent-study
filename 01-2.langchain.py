
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage


# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-2.5-flash-lite")

# SystemMessage와 HumanMessage를 사용해 대화 구성
messages = [
  SystemMessage(content="You are a value investment investor."),
  HumanMessage(content="Broadcom price will be increased?")
]
response = model.invoke(messages)
print(response)

# SystemMessage와 HumanMessage를 사용해 대화 구성

messages = [
  SystemMessage(content="You are a value investment investor."),
  HumanMessage(content="Broadcom price will be increased?"),
  
  HumanMessage(content="My favorite color is blue."),
  HumanMessage(content="What color did I say is my favorite?"),
  HumanMessage(content="Remind me what my first question was.")
]

response = model.invoke(messages)
print(response)



# 메시지를 딕셔너리로 정의 (role, content)
messages_data = [
  {"role": "system", "content": "You are a value investment investor."},
  {"role": "human", "content": "Broadcom price will be increased?"},
  {"role": "human", "content": "My favorite color is blue."},
  {"role": "human", "content": "What color did I say is my favorite?"},
  {"role": "human", "content": "Remind me what my first question was."}
]

response = model.invoke(messages_data)
print(response)
