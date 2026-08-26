# ═══════════════════════════════════════════════════════════════════
# STEP 4 · Agent 조립: 모델 + 도구 + 프롬프트 + 메모리 + 가드레일 + 구조화 출력.
# STEP 6 · 관측성(Langfuse): 요청 하나가 처리되는 과정 전체를 트레이스 하나로 남긴다.
# STEP 7 · 모델 선택: 모델 이름을 코드 밖(.env·요청)으로 꺼낸다.
# ═══════════════════════════════════════════════════════════════════
from dataclasses import dataclass                                     # ← 추가(STEP 7)

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (                             # ← 추가(STEP 4)
    before_agent, wrap_model_call, PIIMiddleware, ToolCallLimitMiddleware,
)                                                                     # ← wrap_model_call 추가(STEP 7)
from langchain.messages import AIMessage                              # ← 추가(STEP 4)
from langgraph.checkpoint.memory import InMemorySaver   # ← 추가(STEP 3)
from pydantic import BaseModel, Field                                 # ← 추가(STEP 4)
# STEP 6 관측성 — 랭체인 실행을 지켜보다 Langfuse로 흘려보내는 참관인 둘
from langfuse.langchain import CallbackHandler                        # ← 추가(STEP 6)
# propagate_attributes = 지금 열려 있는 상자와 그 자식들에 이름표(세션·태그)를 붙이는 도구
from langfuse import get_client, propagate_attributes                 # ← 추가(STEP 6)

from providers import get_model, DEFAULT_PROVIDER                     # ← 추가(STEP 7)
from tools import get_today, calculate, web_search, search_internal_docs   # ← search_internal_docs 추가(STEP 2)

load_dotenv()                                        # .env의 OPENAI_API_KEY·LANGFUSE_* 로드

# ═══════════════════════════════════════════════════════════════════
# STEP 6 · 관측성(Langfuse): 어떤 도구를 몇 번 불렀고, 토큰을 얼마나 썼고, 어디서 오래 걸렸는지 남긴다.
# 키를 인자로 넘기지 않는 점에 주목 — .env의 정해진 이름(LANGFUSE_*)을 알아서 찾아 읽는다.
# 그래서 이 두 줄은 반드시 load_dotenv() '뒤'에 있어야 한다(앞에 두면 키가 아직 없어 못 읽는다).
# ═══════════════════════════════════════════════════════════════════
langfuse_handler = CallbackHandler()
langfuse = get_client()                              # 콜백과 같은 .env 키를 읽는 클라이언트

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
                print(f"⛔ 차단 [{category}] — '{kw}' 감지")
                # 차단된 요청은 모델도 도구도 안 썼으니 토큰 0·지연 0 — 대시보드에서
                # '아무 일도 없었던 것'처럼 보인다. 그런데 운영에서 정말 세고 싶은 숫자가
                # "하루에 몇 번 막혔나"다. 그래서 일부러 태그를 달아 보이게 만든다(STEP 6).
                # with에 들어가는 순간 지금 열려 있는 상자에 태그가 붙고, 태그는 트레이스 단위로
                # 합쳐지므로 ask()가 나중에 다는 태그를 덮어쓰지 않고 나란히 남는다.
                with propagate_attributes(
                    tags=["blocked", category],
                    metadata={"blocked_keyword": kw},
                ):
                    return {
                        "messages": [AIMessage(content=REFUSALS[category])],
                        "jump_to": "end",     # 모델 호출 없이 여기서 끝낸다
                        # 같은 세션의 직전 답이 남아 있지 않게 비운다.
                        # (랭체인은 이 값을 모델 노드가 돌 때 비우는데, 여기서는 모델을 안 부르고 끝내므로 직접 비운다)
                        "structured_response": None,
                    }

    return None                            # None = "이상 없음, 평소대로 진행하라"


# ─────────────────────────────────────────────────────────────
# 모델 선택 — 요청마다 다른 모델을 쓸 수 있게 (STEP 7)
# ─────────────────────────────────────────────────────────────
@dataclass
class Context:
    """요청마다 달라지는 '설정'을 담는 상자.

    대화 내용(state)과는 별개다 — checkpointer에 저장되지 않고,
    요청 한 번 도는 동안만 살아 있다. "이 요청을 어떻게 처리할지"만 담는다.
    """

    provider: str | None = None            # None이면 기본 모델(DEFAULT_PROVIDER)


# @wrap_model_call = 모델을 부르기 '직전'에 끼어들어, 들어가는 요청을 고칠 수 있는 훅.
# 고친 request를 handler에 넘기면 그대로 반영된다.
@wrap_model_call
def select_provider(request, handler):
    """요청에 provider가 지정돼 있으면 그 모델로 갈아끼운다."""
    name = getattr(request.runtime.context, "provider", None)   # 상자에서 provider를 꺼내
    if name:
        request = request.override(model=get_model(name))       # 요청의 모델만 바꿔치기
    return handler(request)                                     # 바뀐 요청으로 실제 호출 진행


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

# Agent가 쥐고 있는 진짜 도구들. 이름만 따로 모아 두는 이유는 ask()의 주석 참고.
TOOLS = [get_today, calculate, web_search, search_internal_docs]      # ← 추가(STEP 2)
TOOL_NAMES = {t.name for t in TOOLS}                                  # ← 추가(STEP 7)

agent = create_agent(
    model=get_model(),                    # ← "openai:gpt-5-mini" 대신 콘센트(STEP 7)
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,                                        # ← 추가(STEP 3)
    context_schema=Context,               # ← 이 Agent가 받을 상자의 모양(STEP 7)
    middleware=[
        input_guardrail,                                          # ① 내가 만든 키워드 검문소
        select_provider,                                          # ② 요청별 모델 교체(STEP 7)
        # ③ 카드번호가 섞여 들어오면 마지막 4자리만 남기고 가린다.
        #    사용자가 실수로 붙여넣은 번호가 모델 제공사 서버까지 가는 걸 막는다.
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        # ④ 요청 하나당 도구 호출 상한. 평소엔 1~3번이면 끝나므로 8이면 넉넉하다.
        #    넘으면 도구 대신 "한도 초과" 결과를 돌려주고(continue), 모델은 그때까지 모은 것으로 답을 마무리한다.
        ToolCallLimitMiddleware(run_limit=8, exit_behavior="continue"),
    ],
    # 구조화 출력. 클래스만 넘기면 랭체인이 모델의 능력을 보고 방식을 고른다(STEP 7) —
    # OpenAI는 JSON 스키마 모드(답 본문이 곧 ChatReply), Gemini 2.5는 '형식 제출용 도구' 방식.
    # 모델을 갈아끼울 수 있게 만들었으니 형식을 받아 내는 방법도 모델에 맡기는 것이다.
    response_format=ChatReply,                                        # ← 추가(STEP 4)
)


def ask(message: str, session_id: str = "default", provider: str | None = None) -> dict:
    """질문 하나를 처리해 {answer, sources, confident, used_tools, blocked} 형태로 돌려준다.

    session_id가 같으면 앞 대화가 이어진다. 두 번째 인자(config)의 thread_id가
    '대화방 번호'다. checkpointer를 달아 놓고 이 값을 안 넘기면 에러가 난다 —
    어느 차트를 꺼낼지 모르니까.

    provider를 주면 이 요청만 그 모델로 처리한다. 안 주면 기본 모델(STEP 7).
    처리 과정 전체는 Langfuse에 트레이스(상자) 하나로 남는다(STEP 6).
    """
    # with = 상자 열기. 이 블록 안에서 생긴 기록은 모두 이 상자의 자식으로 들어가,
    # 대시보드에 '요청 1개 = 트리 1개'로 그려진다. 없으면 모델 호출 3번이 서로 남남인
    # 상자 3개로 흩어져서 "이 질문 하나에 총 얼마 들었나"를 볼 수 없다.
    with langfuse.start_as_current_observation(
        name="agent-chat", as_type="span", input=message,
    ) as trace, propagate_attributes(
        session_id=session_id,             # 세션 이름표 → 같은 대화가 한 줄로 묶인다
        # 어느 모델로 돌렸는지 이름표를 달아 둔다. 두 모델의 비용·지연을 나란히 비교할 때
        # 대시보드에서 이 값으로 걸러 본다(STEP 7).
        metadata={"provider": provider or DEFAULT_PROVIDER},           # ← 추가(STEP 7)
    ):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            {
                "configurable": {"thread_id": session_id},             # ← 추가(STEP 3)
                # 참관인을 들여보내는 줄. 핸들러를 만들어 둔다고 자동으로 기록되지 않는다 —
                # 이 config를 넘긴 호출만 Langfuse에 남는다.
                "callbacks": [langfuse_handler],                       # ← 추가(STEP 6)
            },
            context=Context(provider=provider),                        # ← 상자를 채워 넘긴다(STEP 7)
        )
        messages = result["messages"]

        # 이번 턴에 쓴 도구만 센다. messages에는 지난 대화까지 다 들어 있으므로
        # '마지막 사용자 발화' 이후 구간만 봐야 이번 턴의 도구가 나온다.
        #   예) [Human, AI, Tool, Human, AI, Tool] → last_human=3, 그 뒤 3개가 이번 턴
        # 주의: last_human에 담기는 건 '메시지'가 아니라 그 메시지의 '번호(인덱스)'다.
        # 마지막 사용자 발화의 내용을 읽으려는 게 아니라, messages[last_human:] 슬라이싱의
        # 시작점으로 쓰려고 위치만 구하는 것이다.
        last_human = max(i for i, m in enumerate(messages) if m.type == "human")
        # 세는 건 '우리가 쥐여 준 도구'뿐이다(TOOL_NAMES). 모델에 따라 구조화 출력이
        # ChatReply라는 이름의 도구 호출로 들어오기도 하는데(STEP 7의 Gemini), 그건 답변을
        # 형식에 맞춰 제출하는 절차지 Agent가 고른 도구가 아니다. 모델을 바꿔도 이 목록의
        # 의미가 같아야 STEP 8에서 두 모델의 '도구 선택 정확도'를 같은 자로 잴 수 있다.
        used_tools = [
            call["name"]
            for m in messages[last_human:]           # 이번 턴 구간만 훑고
            # tool_calls는 AIMessage에만 있다. m.tool_calls로 바로 꺼내면 Human에서 터지니
            # getattr(..., None)으로 안전하게 본다. 도구를 안 쓴 AI는 []라서 같이 걸러진다.
            if getattr(m, "tool_calls", None)
            for call in m.tool_calls                 # 한 메시지가 도구를 여러 개 부를 수도 있다
            if call["name"] in TOOL_NAMES
        ]

        # 가드레일에 막히면 모델을 아예 호출하지 않고 끝난다(jump_to="end").
        # 모델이 안 돌았으니 structured_response도 없다 → []로 꺼내면 KeyError, .get()으로 받는다.
        reply = result.get("structured_response")
        # 막힌 경우와 정상인 경우 모두 같은 모양의 dict를 돌려준다.
        # 호출하는 쪽은 blocked만 보면 되고 예외 처리를 따로 하지 않아도 된다(STEP 8 채점이 쉬워진다).
        if reply is None:
            out = {
                "answer": messages[-1].content,   # 가드레일이 넣어 둔 거절 문구
                "sources": [],
                "confident": False,               # 근거로 확인한 게 아니라 문 앞에서 돌려보낸 것
                "used_tools": [],
                "blocked": True,
            }
        else:
            out = {
                "answer": reply.answer,
                "sources": reply.sources,
                "confident": reply.confident,   # ChatReply에 정의해 둔 칸을 그대로 꺼낸다
                "used_tools": used_tools,
                "blocked": False,
            }

        # 결과도 상자에 붙여 둔다. 태그는 대시보드의 필터 손잡이 —
        # "search_internal_docs를 쓴 요청만" 같은 식으로 걸러 볼 수 있다.
        # 도구를 하나도 안 썼으면 태그가 비어 필터에서 사라지므로 no_tool을 대신 단다.
        with propagate_attributes(tags=used_tools or ["no_tool"]):
            trace.update(output=out)
            return out


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

    # Langfuse는 기록을 모아 뒀다가 뒤에서 보낸다. 서버는 계속 떠 있으니 알아서 나가지만,
    # 이렇게 끝나는 스크립트는 다 보내기 전에 프로세스가 죽어 기록이 사라진다 → 끝에서 밀어 보낸다.
    langfuse.flush()                                                  # ← 추가(STEP 6)
