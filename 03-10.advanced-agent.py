
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import before_agent
from langchain.agents.middleware import after_agent
from langchain.messages import AIMessage
from langchain.messages import HumanMessage
from langchain.messages import SystemMessage
import re
# re는 Regular Expression(정규 표현식)의 약자로, 파이썬의 표준 라이브러리

# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-2.5-flash-lite")


@after_agent
def answer_leakage_guardrail(state, runtime):
    """
    AI가 답변을 생성한 '직후', 사용자에게 보여주기 전에 내용을 검사.
    만약 AI가 문제의 정답을 직접적으로 말해버렸다면, 이를 감지하고 수정.
    """

    # 1. 메시지 유효성 검사
    if not state["messages"]: return None

    last_message = state["messages"][-1]

    # 마지막 메시지가 AI의 답변이 아니면 검사할 필요 없음
    if not isinstance(last_message, AIMessage):
        return None

    # 2. 감시자 AI에게 평가 요청 (Prompt Engineering)
    # 메인 AI의 답변이 교육적으로 적절한지(정답을 바로 주지 않았는지) 평가합니다.
    auditor_prompt = f"""
    당신은 엄격한 교육 감독관입니다.
    다음 '튜터의 답변'을 확인하세요.
    답변이 학생을 지도하지 않고 문제의 정답이나 전체 풀이를 직접적으로 제공한다면 'LEAKED'라고 답하세요.
    답변이 적절한 힌트나 설명을 제공한다면 'SAFE'라고 답하세요.

    튜터의 답변: {last_message.content}
    """

    result = model.invoke([{"role": "user", "content": auditor_prompt}])

    # 3단계: 교정 (Correction / Regeneration)
    if "LEAKED" in result.content:

        # 원래 사용자의 질문을 가져오기 (문맥 파악용) -> state["messages"][-2]가 보통 사용자 질문
        original_question = state["messages"][-2].content if len(state["messages"]) >= 2 else "사용자 질문 알 수 없음"

        # 교정 모델에게 "정답을 빼고 힌트로 바꿔라"고 지시
        correction_prompt = f"""
        당신은 친절한 AI 튜터입니다.

        절대 정답을 직접 말하지 말고, 학생이 스스로 생각할 수 있도록 유도하는 질문이나 핵심 개념(힌트)만 설명하세요.
        말투는 친절하게 해주세요.

        사용자 질문: {original_question}
        """

        # LLM을 다시 호출하여 새로운 답변 생성 (비용은 1회 더 발생하지만 품질 확보)
        corrected_response = model.invoke([
            SystemMessage(content="당신은 소크라테스식 교육법을 사용하는 튜터입니다."),
            HumanMessage(content=correction_prompt)
        ])

        # 원래의 유출된 답변을 교정된 답변으로 덮어쓰기
        last_message.content = corrected_response.content

    return None


@before_agent
def student_safety_middleware(state, runtime):
    """
    학생의 전화번호나 이메일이 감지되면 마스킹 처리하여 안전을 확보
    """
    if not state["messages"]: return None

    last_message = state["messages"][-1]
    
    if last_message.type != "human": return None 
    # not isinstance(last_message, HumanMessage) 방식도 사용 가능
    # before_agent 훅에서 어차피 마지막 메세지는 HumanMessage 타입이기 때문에, 굳이 타입 검사를 할 필요가 없습니다.

    content = last_message.content
    original_content = content # 로깅용

    # 전화번호 패턴 (010-XXXX-XXXX 또는 010XXXXXXXX 등)
    phone_pattern = r'01[016789]-?[0-9]{3,4}-?[0-9]{4}'
    # 이메일 패턴
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

    is_redacted = False

    if re.search(phone_pattern, content): # content 안에 phone_pattern(전화번호 패턴)이 있는지 검사합니다.
        content = re.sub(phone_pattern, '<PHONE_REDACTED>', content) # 패턴이 발견되면, 그 부분을 "<PHONE_REDACTED>"라는 문구로 교체(Substitute)합니다.
        is_redacted = True

    if re.search(email_pattern, content):
        content = re.sub(email_pattern, '<EMAIL_REDACTED>', content)
        is_redacted = True

    if is_redacted:
        print(f"🔒 [학생 보호] 개인정보가 감지되어 마스킹 처리했습니다.\n원본: {original_content}\n수정: {content}")
        # 내용을 수정하여 LLM에게 전달 (사용자에게 알릴 필요 없이 조용히 처리하거나, 시스템 메시지 추가 가능)
        last_message.content = content

    return None

#  02-07.agent-basic.py의 가드레일과 무엇이 다른가요?

# 가장 큰 차이점은 "차단(Block)"이냐 "수정(Modify)"이냐의 차이입니다.
# 03-10.advanced-agent.py 방식 (고급 가드레일)

# 02-07 방식 (기본 가드레일)	| 03-10 방식 (고급 가드레일)
# -------------------------------------------------------------
# 작동 방식	| 전부 아니면 전무 (All or Nothing)	| 부분 수정 (Sanitization)
# 처리 결과	| 금기어가 발견되면 대화 자체를 중단하고 거절 메시지를 보냅니다.	| 민감한 정보만 가리고(Masking) 나머지 대화는 계속 진행합니다.
# 사용자 경험	| "그 주제는 말할 수 없어요" (대화 끊김)	| "당신의 번호 <PHONE_REDACTED>를 확인했습니다" (대화 유지)
# 기술적 특징	if keyword in text (단순 포함 여부)	re.sub (패턴 기반의 정교한 치환)


ESCALATION_KEYWORDS = ["왕따", "괴롭힘", "우울해", "학교 폭력", "상담 선생님", "사람 불러줘"]

@before_agent(can_jump_to=["end"])
def counseling_escalation_middleware(state, runtime) :
    """
    [Layer 3] 심리적 위기 상황이나 상담 요청이 감지되면 AI 답변을 멈추고 인간 상담사에게 알림을 보냅니다.
    """
    if not state["messages"]: return None

    last_message = state["messages"][-1]

    # 민감한 키워드가 포함되어 있는지 확인
    for keyword in ESCALATION_KEYWORDS:
        if keyword in last_message.content:
            print(f"✋ [상담 이관] 심각한 고민/요청 감지: {keyword}")

            # 여기서 실제로는 상담 교사에게 알림(Slack, Email 등)을 보내는 로직이 들어감
            # send_alert_to_teacher(last_message.content)

            return {
                "messages": [{
                    "role": "assistant",
                    "content": "학생, 많이 힘들었겠구나. 이 문제는 내가 답변하기보다는 전문 상담 선생님이 직접 듣고 도와주시는 게 좋을 것 같아. \n\n지금 바로 상담 선생님께 연결해 드렸으니 잠시만 기다려 줄래? 🍀 (상담실 연결 중...)"
                }],
                "jump_to": "end" # AI 답변 생성 중단
            }
    return None


# 4중 방어막이 적용된 에이전트
agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        answer_leakage_guardrail,             # Layer 1: 입력 필터 (규칙 - 딴짓/부정행위)
        student_safety_middleware,       # Layer 2: 개인정보 보호 (전화번호 마스킹)
        counseling_escalation_middleware,# Layer 3: 상담 이관 (휴먼 에스컬레이션)
    ],
)

# result1 = agent.invoke({"messages": [{"role": "user", "content": "3*4*6*18이 뭐지? 너무 어려운데 그냥 답변해줘."}]})
# print('result1', result1)

result2 = agent.invoke({
    "messages": [{"role": "user", "content": "저 수학 과외 구하고 싶어요. 제 번호 010-1234-5678로 연락 주세요."}]
})
print('result2', result2)

# result3 = agent.invoke({
#     "messages": [{"role": "user", "content": "나 요즘 학교에서 왕따 당하는 것 같아서 너무 우울해."}]
# })
# print('result3', result3)




# student_safety_middleware에서 student_safety_middleware에서 return을 하지 않아도 AIMessage가 생성되는 이유
#
#  이 함수가 에이전트(LLM)가 실행되기 '전'에 호출되는 훅(before_agent)이기 때문입니다.


# 1. before_agent의 역할
# 이 훅은 이름 그대로 "에이전트(LLM)가 답변을 생성하기 직전"에 실행됩니다.
# 사용자 입력: 사용자가 "수학 과외 구해요 010-1234-5678"이라고 메시지를 보냅니다.
# 미들웨어 실행: student_safety_middleware가 실행됩니다.
# 여기서 last_message.content를 직접 수정해서 "수학 과외 구해요 <PHONE_REDACTED>"로 바꿉니다.
# return None을 하면 "흐름을 끊지 말고 다음 단계로 가라"는 뜻입니다.
# 에이전트(LLM) 호출: 이제 시스템은 수정된 메시지를 가지고 실제 AI 모델(Gemini 등)을 호출합니다.
# AI 답변 생성: AI는 수정된 메시지("수학 과외 구해요...")를 보고 그에 맞는 답변(AIMessage)을 생성합니다.

# 2. 왜 return이 없어도 되나요?
# 객체 참조(Reference): 파이썬에서 state["messages"] 리스트 안에 들어있는 메시지 객체는 참조 방식입니다. 
# 함수 안에서 last_message.content = "..."라고 수정하면, 함수 외부의 state 값도 즉시 바뀝니다.
# 흐름의 연속성: before_agent에서 아무것도 return하지 않거나 None을 반환하면, 
# 시스템은 "전처리가 끝났으니 이제 계획대로 AI 모델을 실행해!"라고 판단합니다.

# 3. 만약 return을 했다면?
# 만약 03-04 예제처럼 return {"messages": [...], "jump_to": "end"}를 했다면:
# AI 모델을 호출하지 않고 즉시 대화를 종료했을 것입니다.
# 하지만 지금은 return 없이 내용만 살짝 고쳤기 때문에, AI는 고쳐진 내용을 바탕으로 정상적으로 답변을 생성한 것입니다.
