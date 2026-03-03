

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from dataclasses import dataclass
from langchain.agents.middleware import before_agent
from langchain.messages import AIMessage


# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-2.5-flash-lite")


forbidden_topics = {
  "cheating":['cheating', 'cheat', 'cheating on tests', 'cheating on exams', 'cheating on homework', 'cheating on assignments', 'cheating on projects', 'cheating on quizzes', 'cheating on tests', 'cheating on exams', 'cheating on homework', 'cheating on assignments', 'cheating on projects', 'cheating on quizzes'],
  "drugs":['drugs', 'drug', 'drugs on tests', 'drugs on exams', 'drugs on homework', 'drugs on assignments', 'drugs on projects', 'drugs on quizzes', 'drugs on tests', 'drugs on exams', 'drugs on homework', 'drugs on assignments', 'drugs on projects', 'drugs on quizzes'],
  "violence":['violence', 'violent', 'violence on tests', 'violence on exams', 'violence on homework', 'violence on assignments', 'violence on projects', 'violence on quizzes', 'violence on tests', 'violence on exams', 'violence on homework', 'violence on assignments', 'violence on projects', 'violence on quizzes'],
  "suicide":['suicide', 'suicidal', 'suicide on tests', 'suicide on exams', 'suicide on homework', 'suicide on assignments', 'suicide on projects', 'suicide on quizzes', 'suicide on tests', 'suicide on exams', 'suicide on homework', 'suicide on assignments', 'suicide on projects', 'suicide on quizzes'],
  "sexual harassment":['sexual harassment', 'sexual harassment on tests', 'sexual harassment on exams', 'sexual harassment on homework', 'sexual harassment on assignments', 'sexual harassment on projects', 'sexual harassment on quizzes', 'sexual harassment on tests', 'sexual harassment on exams', 'sexual harassment on homework', 'sexual harassment on assignments', 'sexual harassment on projects', 'sexual harassment on quizzes'],
  "sexual assault":['sexual assault', 'sexual assault on tests', 'sexual assault on exams', 'sexual assault on homework', 'sexual assault on assignments', 'sexual assault on projects', 'sexual assault on quizzes', 'sexual assault on tests', 'sexual assault on exams', 'sexual assault on homework', 'sexual assault on assignments', 'sexual assault on projects', 'sexual assault on quizzes'],
  "sexual abuse":['sexual abuse', 'sexual abuse on tests', 'sexual abuse on exams', 'sexual abuse on homework', 'sexual abuse on assignments', 'sexual abuse on projects', 'sexual abuse on quizzes', 'sexual abuse on tests', 'sexual abuse on exams', 'sexual abuse on homework', 'sexual abuse on assignments', 'sexual abuse on projects', 'sexual abuse on quizzes'],
  "sexual misconduct":['sexual misconduct', 'sexual misconduct on tests', 'sexual misconduct on exams', 'sexual misconduct on homework', 'sexual misconduct on assignments', 'sexual misconduct on projects', 'sexual misconduct on quizzes', 'sexual misconduct on tests', 'sexual misconduct on exams', 'sexual misconduct on homework', 'sexual misconduct on assignments', 'sexual misconduct on projects', 'sexual misconduct on quizzes'],
}

@before_agent(can_jump_to=["end"])
def forbidden_topics_guardrail(state, runtime):

  print("state", state)
  """
  Forbidden topics guardrail
  """

  if not state["messages"]:
    return None

  last_message = state["messages"][-1]
  print("last_message", last_message.type)
  if last_message.type != "human":
    return None

  user_message = last_message.content

  for topic, keywords in forbidden_topics.items():
    if any(keyword in user_message.lower() for keyword in keywords):
      return {
        "messages": [AIMessage(f"I am sorry, I cannot talk about {topic}.")],
        "jump_to": "end"
      }

  return None

agent = create_agent(
  model=model,
  tools=[],
  middleware=[forbidden_topics_guardrail],
)

result1 = agent.invoke({"messages": [{"role": "user", "content": "I am going to cheat on my exams."}]})
print('result1', result1)



# state를 출력했을 때 HumanMessage(...)라고 나오는 객체에서 어떻게 .type 필드에 접근할 수 있는지 궁금하시군요!

# 그 이유는 HumanMessage가 LangChain의 BaseMessage라는 클래스를 상속받아 만들어진 객체이기 때문입니다.

# 1. BaseMessage의 기본 속성
# LangChain에서 정의한 모든 메시지 객체(Human, AI, System 등)는 공통적으로 다음과 같은 속성들을 가지고 있습니다:
# - content: 메시지의 실제 텍스트 내용
# - type: 메시지의 종류를 나타내는 문자열 (예: "human", "ai", "system")
# - additional_kwargs: 추가적인 정보들
# - id: 메시지 고유 식별자
# 출력 결과에는 content, additional_kwargs, id 등만 보이지만, 내부적으로는 type이라는 속성도 정의되어 있습니다.


# 왜 출력(print)할 때는 안 보였나요?
# print(state)를 했을 때 나오는 결과는 객체의 __repr__ (표현식) 메서드가 정의한 내용만 보여줍니다.
# 실제로는 이런 구조입니다 (개념적 설명)
# class HumanMessage(BaseMessage):
#   type: str = "human"  # 클래스 내부에 이미 정의됨
#   content: str
#   ...
# LangChain 개발자들이 __repr__을 구현할 때 type은 클래스 이름 자체(HumanMessage)로 이미 알 수 있기 때문에, 중복을 피하기 위해 출력 결과에서는 생략하도록 설계한 것입니다. 하지만 메모리 상의 객체에는 엄연히 type이라는 변수가 존재하므로 점(.) 연산자로 접근이 가능한 것입니다.