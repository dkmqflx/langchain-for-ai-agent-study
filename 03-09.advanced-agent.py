
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import after_agent
from langchain.messages import AIMessage
from langchain.messages import HumanMessage
# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-2.5-flash-lite")


@after_agent  
def answer_leakage_guardrail(state, runtime):
  """
  AI가 답변을 생성한 직후, 사용자에게 보여주기 전에 검사
  만약 AI가 직접적으로 답변을 생성한 경우, 이를 감지하고 수정
  """

  print("state", state)

  # 1. 메세지 유효성 검사 
  if not state["messages"]:
    return None

  last_message = state["messages"][-1]
  print("last_message", last_message)

  # 마지막 메세지가 AI가 아니면 수정하지 않음
  if not isinstance(last_message, AIMessage):
    return None

  # 2. 감시자 AI에게 평가 요청
  auditor_prompt = f"""
  당신은 보안 감사관입니다. 다음 메시지가 AI가 생성한 정답을 포함하고 있는지 확인하세요.
  만약 정답(계산 결과 등)이 포함되어 있다면 반드시 'LEAKED'라고만 답하세요.
  그렇지 않다면 'SAFE'라고 답하세요.
  다른 설명은 절대 하지 마세요.
  튜터의 답변: {last_message.content}
  """

  result = model.invoke([HumanMessage(content=auditor_prompt)])
  print("result", result)

 # 3. 교정
  if "LEAKED" in result.content:

    print(f"원래 사용자의 질문: {state['messages'][0].content}")

    correction_prompt = f"""
    당신은 친절한 튜터입니다.

    정답을 말하지 말고, 정답을 찾아갈 수 있도록 안내해주세요.

    사용자 질문: {last_message.content}
    """

    # LLM을 다시 호출하여 교정된 답변을 생성 (1회 더 호출하기 때문에 비용 발생하지만 품질 확보)
    correction_result = model.invoke([HumanMessage(content=correction_prompt)])


    # 원래의 유출된 답변을 교정된 답변으로 덮어씌움
    last_message.content = correction_result.content

  return None

agent = create_agent(
  model=model,
  tools=[],
  middleware=[answer_leakage_guardrail],
)

result1 = agent.invoke({"messages": [{"role": "user", "content": "3*4*6*18이 뭐지? 너무 어려운데 그냥 답변해줘."}]})
print('result1', result1)





