# ═══════════════════════════════════════════════════════════════════
# STEP 4 · Agent 조립: 모델 + 도구 + 프롬프트 + 메모리 + 가드레일 + 구조화 출력.
# 이 파일이 앞으로 관측(STEP 6) · 모델선택(STEP 7)을 하나씩 얹어 가며 자라날 자리다.
# ═══════════════════════════════════════════════════════════════════
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import before_agent, PIIMiddleware   # ← 추가(STEP 4)
from langchain.agents.structured_output import ToolStrategy           # ← 추가(STEP 4)
from langchain.messages import AIMessage                              # ← 추가(STEP 4)
from langgraph.checkpoint.memory import InMemorySaver   # ← 추가(STEP 3)
from pydantic import BaseModel, Field                                 # ← 추가(STEP 4)

from tools import get_today, calculate, web_search, search_internal_docs   # ← search_internal_docs 추가(STEP 2)

load_dotenv()                                        # .env의 OPENAI_API_KEY 로드

# 시스템 프롬프트 = Agent의 '행동 지침서'. 매 요청마다 대화 맨 앞에 조용히 붙는다.
# 도구를 쥐여 줬다고 알아서 쓰지는 않는다 — '언제 써야 하는지'까지 여기서 못 박아야 한다.
SYSTEM_PROMPT = """너는 정확성을 최우선으로 하는 업무 비서다.

규칙:
1. 오늘 날짜·현재 시각이 필요하면 반드시 get_today를 호출한다. 날짜를 추측하지 않는다.
2. 숫자 계산은 반드시 calculate를 호출한다. 암산하지 않는다.
3. 약관·규정·계약 조건에 관한 질문은 web_search보다 search_internal_docs를 먼저 쓴다.
   우리 문서에 있는 내용을 인터넷에서 찾으면 틀린 답이 나온다.
4. search_internal_docs의 결과를 쓸 때는 답변에 [근거]의 문서명과 쪽수를 반드시 적는다.
5. 최신 정보가 필요하면 web_search를 호출하고, 답변에 출처 URL을 함께 적는다.
6. 도구로도 확인할 수 없는 내용은 모른다고 말한다. 지어내지 않는다."""

# ─────────────────────────────────────────────────────────────
# 가드레일 ① — 입력 검문소 (STEP 4)
# ─────────────────────────────────────────────────────────────
# 막을 주제와, 그 주제를 나타내는 신호 단어들.
# 완벽한 목록은 만들 수 없다 — 목적은 '흔한 경로를 싸게 막는 것'이지 '전부 막는 것'이 아니다.
BLOCKED = {
    # ① 프롬프트 탈취·조작: 시스템 지침을 빼내거나 무력화하려는 시도
    "prompt_injection": ["시스템 프롬프트", "이전 지시 무시", "규칙 무시", "너의 지침", "system prompt"],
    # ② 투자 권유: 금융 도메인에서 함부로 답하면 규제 위반이 될 수 있는 영역
    "investment_advice": ["어디에 투자", "수익 보장", "원금 보장", "얼마 벌 수", "추천 종목"],
    # ③ 자격증명·민감정보: 애초에 우리가 알지도 못하고 알아서도 안 되는 것
    "credential": ["비밀번호 알려", "계좌 비밀번호", "주민등록번호 알려", "카드번호 알려"],
}

# 차단할 때 돌려줄 말. 사용자가 '왜 막혔는지' 알 수 있게 카테고리별로 다르게 쓴다.
# "처리할 수 없습니다" 한 마디로 뭉뚱그리면 사용자는 서비스가 고장 난 줄 안다.
REFUSALS = {
    "prompt_injection": "죄송합니다. 시스템 설정이나 내부 지침에 관한 요청은 처리할 수 없습니다.",
    "investment_advice": "저는 약관·규정 안내만 도와드릴 수 있습니다. 투자 판단이나 수익에 관한 답변은 드릴 수 없어요.",
    "credential": "비밀번호·주민등록번호 같은 민감정보는 다루지 않습니다. 해당 기관에 직접 문의해 주세요.",
}


# @before_agent = Agent가 일을 시작하기 '직전' 딱 한 번 실행되는 훅.
# can_jump_to=["end"] 는 "필요하면 곧장 끝으로 점프할 권한을 주세요"라는 선언이다.
@before_agent(can_jump_to=["end"])
def input_guardrail(state, runtime):
    """사용자 입력이 LLM에 닿기 전에 검사한다. 막을 게 없으면 None을 돌려준다."""
    if not state["messages"]:
        return None
    last = state["messages"][-1]
    if last.type != "human":              # 사람이 방금 한 말이 아니면 검사할 게 없다
        return None

    text = last.content
    for category, keywords in BLOCKED.items():
        for kw in keywords:
            if kw in text:
                # STEP 6에서 이 print 자리를 Langfuse 기록으로 바꾼다.
                # 차단은 '아무 일도 안 일어난 것'처럼 보여서, 기록하지 않으면 존재 자체가 안 보인다.
                print(f"⛔ 차단 [{category}] — '{kw}' 감지")
                return {
                    "messages": [AIMessage(content=REFUSALS[category])],
                    "jump_to": "end",     # 모델 호출 없이 여기서 끝낸다
                }

    return None                            # None = "이상 없음, 평소대로 진행하라"


# ─────────────────────────────────────────────────────────────
# 구조화 출력 — 채점할 수 있게 '칸'을 만든다 (STEP 4)
# ─────────────────────────────────────────────────────────────
class ChatReply(BaseModel):
    """사용자에게 돌려줄 최종 답변의 형식."""

    answer: str = Field(description="사용자에게 보여줄 답변 본문")
    sources: list[str] = Field(
        default_factory=list,
        description="근거 출처 목록. 사내 문서면 '문서명 p.쪽수', 웹이면 URL. "
                    "도구를 쓰지 않았거나 근거가 없으면 빈 목록.",
    )
    confident: bool = Field(
        description="근거로 확인한 내용이면 true, 모르거나 확인하지 못했으면 false",
    )


# 대화 기록 보관함. Agent에게 '수첩과 펜'을 쥐여 주는 것과 같다.
# InMemorySaver는 이름 그대로 '메모리(RAM)에' 저장한다 → 서버를 끄면 전부 사라진다.
# 운영에서는 Postgres/SQLite 기반 checkpointer로 교체해야 한다(STEP 9 README에 명시).
checkpointer = InMemorySaver()

# 모델은 "provider:model" 문자열로도 넘길 수 있다. STEP 7에서 이 부분을 갈아끼운다.
agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[get_today, calculate, web_search, search_internal_docs],   # ← 추가
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,                                        # ← 추가(STEP 3)
    middleware=[
        input_guardrail,                                          # ① 내가 만든 키워드 검문소
        # ② 카드번호가 섞여 들어오면 마지막 4자리만 남기고 가린다.
        #    사용자가 실수로 붙여넣은 번호가 OpenAI 서버까지 가는 걸 막는다.
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
    ],
    response_format=ToolStrategy(ChatReply),                          # ← 추가(STEP 4)
)


def ask(message: str, session_id: str = "default") -> dict:
    """질문 하나를 처리해 {answer, sources, used_tools, blocked} 형태로 돌려준다.

    session_id가 같으면 앞 대화가 이어진다. 두 번째 인자(config)의 thread_id가
    '대화방 번호'다. checkpointer를 달아 놓고 이 값을 안 넘기면 에러가 난다 —
    어느 차트를 꺼낼지 모르니까.
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        {"configurable": {"thread_id": session_id}},                  # ← 추가(STEP 3)
    )
    messages = result["messages"]

    # 이번 턴에 쓴 도구만 센다. messages에는 지난 대화까지 다 들어 있으므로
    # '마지막 사용자 발화' 이후 구간만 봐야 이번 턴의 도구가 나온다.
    last_human = max(i for i, m in enumerate(messages) if m.type == "human")
    # ToolStrategy는 '이 형식으로 제출하라'는 도구(=ChatReply)를 내부적으로 하나 더 만든다.
    # 그건 우리가 쥐여 준 도구가 아니므로 뺀다 — STEP 8의 채점 대상은 진짜 도구뿐이다.
    used_tools = [
        call["name"]
        for m in messages[last_human:]
        if getattr(m, "tool_calls", None)
        for call in m.tool_calls
        if call["name"] != ChatReply.__name__
    ]

    # 가드레일에 막혔으면 structured_response가 없다 → .get()으로 안전하게 꺼낸다
    reply = result.get("structured_response")
    if reply is None:
        return {
            "answer": messages[-1].content,   # 가드레일이 넣어 둔 거절 문구
            "sources": [],
            "used_tools": [],
            "blocked": True,
        }

    return {
        "answer": reply.answer,
        "sources": reply.sources,
        "used_tools": used_tools,
        "blocked": False,
    }


if __name__ == "__main__":
    import sys

    # 실행할 때 세션 이름을 줄 수 있다:  uv run python agent.py 나의세션
    session = sys.argv[1] if len(sys.argv) > 1 else "demo"
    print(f"세션 '{session}' — 빈 줄을 입력하면 종료합니다.")

    while True:
        try:
            msg = input("\n나 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            break

        print(ask(msg, session))
