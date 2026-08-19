# ═══════════════════════════════════════════════════════════════════
# STEP 3 · Agent 조립: 모델 + 도구 + 시스템 프롬프트 + 메모리(checkpointer).
# 이 파일이 앞으로 가드레일(STEP 4) · 관측(STEP 6) · 모델선택(STEP 7)을
# 하나씩 얹어 가며 자라날 자리다.
# ═══════════════════════════════════════════════════════════════════
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver   # ← 추가(STEP 3)

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
)

def ask(message: str, session_id: str = "default"):
    """질문 하나를 Agent에 던지고, 오간 메시지 전체를 돌려준다.

    최종 답변만이 아니라 messages 전체를 돌려주는 게 지금 단계에서는 중요하다.
    '어떤 도구를 왜 골랐는지'가 그 안에 다 들어 있고, 그걸 봐야 학습이 된다.

    session_id가 같으면 앞 대화가 이어진다. 두 번째 인자(config)의 thread_id가
    '대화방 번호'다. checkpointer를 달아 놓고 이 값을 안 넘기면 에러가 난다 —
    어느 차트를 꺼낼지 모르니까.
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        {"configurable": {"thread_id": session_id}},                  # ← 추가(STEP 3)
    )
    return result["messages"]


if __name__ == "__main__":
    import sys

    # 실행할 때 세션 이름을 줄 수 있다:  uv run python agent.py 나의세션
    session = sys.argv[1] if len(sys.argv) > 1 else "demo"
    print(f"세션 '{session}' — 빈 줄을 입력하면 종료합니다.")

    seen = 0                                  # 지금까지 화면에 찍은 메시지 개수
    while True:
        try:
            msg = input("\n나 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg:
            break

        messages = ask(msg, session)          # 항상 '대화 전체'가 돌아온다
        new, seen = messages[seen:], len(messages)   # 이번 턴에 새로 생긴 것만 골라 찍는다

        for m in new:
            if m.type == "human":
                continue                      # 내가 방금 친 말은 다시 안 찍는다
            if getattr(m, "tool_calls", None):
                print(f"  🔧 {[(c['name'], c['args']) for c in m.tool_calls]}")
            elif m.type == "tool":
                print(f"  ↩︎ ({m.name}) {m.content[:80]}...")
            elif m.content:
                print(f"\n🤖 {m.content}")
