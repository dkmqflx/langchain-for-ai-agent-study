# ═══════════════════════════════════════════════════════════════════
# STEP 1 · Agent 조립: 모델 + 도구 + 시스템 프롬프트를 하나로 묶는다.
# 이 파일이 앞으로 메모리(STEP 3) · 가드레일(STEP 4) · 관측(STEP 6) · 모델선택(STEP 7)을
# 하나씩 얹어 가며 자라날 자리다. 지금은 가장 앙상한 뼈대만 세운다.
# ═══════════════════════════════════════════════════════════════════
from dotenv import load_dotenv
from langchain.agents import create_agent

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

# 모델은 "provider:model" 문자열로도 넘길 수 있다. STEP 7에서 이 부분을 갈아끼운다.
agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[get_today, calculate, web_search, search_internal_docs],   # ← 추가
    system_prompt=SYSTEM_PROMPT,
)

def ask(message: str):
    """질문 하나를 Agent에 던지고, 오간 메시지 전체를 돌려준다.

    최종 답변만이 아니라 messages 전체를 돌려주는 게 지금 단계에서는 중요하다.
    '어떤 도구를 왜 골랐는지'가 그 안에 다 들어 있고, 그걸 봐야 학습이 된다.
    """
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    return result["messages"]


if __name__ == "__main__":
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else "오늘 며칠이야? 그리고 1234 곱하기 5678은?"

    for m in ask(question):
        # 메시지 종류(human / ai / tool)와 내용을 한 줄씩 찍는다.
        # AIMessage는 도구를 부를 때 content가 비어 있고 tool_calls에만 내용이 있다 —
        # 그래서 둘 다 찍어야 무슨 일이 있었는지 보인다.
        print(f"\n[{m.type}] {m.content}")
        if getattr(m, "tool_calls", None):
            print(f"    ↳ 호출: {[(c['name'], c['args']) for c in m.tool_calls]}")
