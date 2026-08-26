# ═══════════════════════════════════════════════════════════════════
# STEP 8 · 프롬프트 A/B 평가 하네스: 고정된 코스(eval_cases.json)를 프롬프트마다 돌리고
#          같은 스톱워치(채점 규칙)로 재서 표 한 장으로 만든다.
#
# 점수는 셋으로 나눠 잰다 — ① 도구 선택 ② 형식 준수는 코드로 세고(공짜),
# 사람의 판단이 필요한 ③ 답변 품질만 LLM에게 맡긴다(LLM-as-judge).
#
# 실행: uv run python evaluate.py
#   ※ search_internal_docs가 P1을 부르므로 P1 서버(8000)가 떠 있어야 한다.
# ═══════════════════════════════════════════════════════════════════
import json

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langfuse import get_client

from agent import ask               # 우리 Agent를 부르는 창구
from prompts import PROMPTS         # {"A": ..., "B": ...}

load_dotenv()
langfuse = get_client()


# ─────────────────────────────────────────────────────────────
# ④-1 채점자(judge) 준비 — 점수와 이유를 정해진 모양으로 받는다
# ─────────────────────────────────────────────────────────────
class Judgement(BaseModel):
    """채점 결과의 형식. 점수만 받지 않고 이유까지 받는다.

    이유를 함께 받는 이유는 둘이다 — 나중에 우리가 "왜 이 점수인지" 읽어야 하고,
    이유를 쓰게 하면 채점자가 답변을 실제로 들여다보게 되어 점수가 덜 흔들린다.
    """

    score: int = Field(description="1~5점", ge=1, le=5)
    reason: str = Field(description="그 점수를 준 이유 한 문장")


# STEP 4에서 ChatReply로 한 것과 같은 일 — "이 칸에 맞춰 채워라"라고 모델을 묶어 둔다.
# 그래서 judge.invoke(...)는 문자열이 아니라 Judgement 객체를 돌려준다.
# 채점자는 피평가 모델과 다르게 두는 편이 낫다(자기 답에 후한 경향).
# init_chat_model("google_genai:gemini-2.5-flash")로 바꿔 두 채점자를 비교해 보세요.
judge = init_chat_model("openai:gpt-5-mini").with_structured_output(Judgement)

# 척도의 뜻을 못 박아 두는 게 핵심이다. "1~5점으로 평가하라"고만 하면 채점자는
# 대부분 3~4점만 주고, 그러면 A와 B의 평균이 붙어 버려 비교가 안 된다.
JUDGE_TEMPLATE = """너는 엄격한 채점자다. 아래 답변을 채점 기준에 따라 1~5점으로 평가하라.

[질문]
{question}

[채점 기준]
{rubric}

[답변]
{answer}

[제시된 근거]
{sources}

점수 척도 — 반드시 이 기준을 따르라:
5 = 기준을 완전히 충족하고, 근거가 적절하다
4 = 기준을 충족하나 사소한 누락이 있다
3 = 절반쯤 맞다. 중요한 내용이 빠졌다
2 = 대체로 빗나갔다. 일부만 관련 있다
1 = 완전히 틀렸거나, 문서에 없는 내용을 지어냈다

지어낸 내용이 하나라도 있으면 다른 게 아무리 좋아도 1점이다."""


# ─────────────────────────────────────────────────────────────
# ④-2 답변 하나를 채점받는다
# ─────────────────────────────────────────────────────────────
def judge_answer(case: dict, result: dict) -> Judgement:
    """case = eval_cases.json의 케이스 하나, result = ask()가 돌려준 dict."""
    prompt = JUDGE_TEMPLATE.format(
        question=case["question"],
        rubric=case["rubric"],
        answer=result["answer"],
        # 빈 대괄호 []만 보내면 채점자가 근거가 없다는 사실을 흘려보내기 쉽다.
        sources=result["sources"] or "(없음)",
    )
    return judge.invoke(prompt)


# ─────────────────────────────────────────────────────────────
# ④-3~6 케이스 하나를 돌리고 채점해 '표 한 줄'로 만든다
# ─────────────────────────────────────────────────────────────
def run_case(case: dict, label: str, system_prompt: str) -> dict:
    # 세션을 케이스마다 다르게 준다. 같으면 STEP 3의 메모리 때문에 앞 케이스의 대화가
    # 뒤 케이스에 섞이고, 그건 우리가 재려던 상황이 아니다.
    # 덤으로 이 이름은 Langfuse에 세션 이름으로 남아 'eval-'로 걸러 볼 수 있다(STEP 6).
    result = ask(
        case["question"],
        session_id=f"eval-{label}-{case['id']}",
        system_prompt=system_prompt,               # agent.py에 뚫어 둔 구멍
    )

    # ① 도구 선택 — 기대한 도구를 썼는가. 기대가 없으면(None) '아무 도구도 안 썼는가'.
    #    ==이 아니라 in인 이유: 날짜를 확인한 뒤 약관을 찾아보는 건 잘못된 판단이 아니다.
    #    여기서 재는 건 "딱 하나만 불렀나"가 아니라 "필요한 도구를 골라냈나"다.
    expect = case.get("expect_tool")
    used = result["used_tools"]
    tool_ok = (expect in used) if expect else (len(used) == 0)

    # ② 형식 준수 — 근거를 요구한 케이스에서 sources가 채워졌는가 + 차단 여부가 기대와 같은가.
    #    차단은 양방향으로 맞아야 한다. 막아야 할 게 통과한 것도, 평범한 질문이 막힌 것도 틀린 거다.
    sources_ok = bool(result["sources"]) if case.get("expect_sources") else True
    block_ok = result["blocked"] == case.get("expect_blocked", False)
    format_ok = sources_ok and block_ok

    # ③ 답변 품질 — 차단이 정답인 케이스는 채점할 '내용'이 없으므로 성공/실패만 점수로 넣는다.
    #    비워 두면 가드레일이 뚫려도 품질 평균이 꿈쩍하지 않아 표만 보고는 눈치채지 못한다.
    #    덤으로 채점 호출도 그만큼 아낀다.
    if case.get("expect_blocked"):
        score, reason = (5 if block_ok else 1), "차단 케이스(내용 채점 생략)"
    else:
        verdict = judge_answer(case, result)
        score, reason = verdict.score, verdict.reason

    # 이 dict 하나가 표의 한 줄이다. 케이스 8개 × 프롬프트 2개 = 16줄.
    return {
        "id": case["id"], "prompt": label,
        "tool_ok": tool_ok, "format_ok": format_ok, "score": score,
        # 리스트는 표의 한 칸에 넣기 어려워 문자열로 만든다. 빈 문자열은 표에서 안 보이니 "-".
        "expect_tool": expect, "used_tools": ",".join(used) or "-",
        "blocked": result["blocked"], "reason": reason,
    }


# ─────────────────────────────────────────────────────────────
# ④-7 전체 실행 — 돌리고, 표로 모으고, 요약과 비교를 찍는다
# ─────────────────────────────────────────────────────────────
def main():
    cases = json.load(open("eval_cases.json", encoding="utf-8"))
    rows = []

    # (1) 프롬프트마다 케이스를 전부 돌린다. 몇 분 걸리는 작업이라 진행 상황을 찍어 준다.
    for label, system_prompt in PROMPTS.items():          # A 한 바퀴, B 한 바퀴
        print(f"\n=== 프롬프트 {label} · {len(cases)}개 케이스 ===")
        for case in cases:
            row = run_case(case, label, system_prompt)
            rows.append(row)
            mark = "✅" if (row["tool_ok"] and row["format_ok"]) else "❌"
            print(f"  {mark} {row['id']:10s} 도구={row['used_tools']:24s} 품질={row['score']}")

    # (2) 줄들을 표로 바꾸고 원본을 남긴다. 요약표에는 reason이 안 남으므로
    #     "왜 이 점수였지?"를 나중에 확인하려면 이 CSV가 필요하다.
    #     엑셀에서 한글이 깨지지 않게 utf-8-sig로 저장한다.
    df = pd.DataFrame(rows)
    df.to_csv("eval_results.csv", index=False, encoding="utf-8-sig")

    # (3) 프롬프트별 요약 — 이게 목표였던 표.
    #     mean()이 True를 1, False를 0으로 세므로 tool_ok의 평균이 곧 정확도다.
    summary = df.groupby("prompt").agg(
        도구선택_정확도=("tool_ok", "mean"),
        형식준수율=("format_ok", "mean"),
        답변품질_평균=("score", "mean"),
    ).round(3)

    print("\n" + "=" * 52)
    print(summary.to_string())
    print("=" * 52)

    # (4) 어느 케이스에서 갈렸는지 — 요약표보다 이쪽이 훨씬 쓸모 있다.
    #     pivot은 긴 표를 눕혀 케이스 하나를 한 줄로, A와 B를 나란히 놓는다.
    pivot = df.pivot(index="id", columns="prompt", values="score")
    diff = pivot[pivot["A"] != pivot["B"]]
    if not diff.empty:
        print("\n[A와 B의 점수가 갈린 케이스]")
        print(diff.to_string())

    print("\n케이스별 원본: eval_results.csv (reason 열을 읽어 보세요)")

    langfuse.flush()   # 스크립트가 끝나기 전에 기록을 모두 전송한다 (STEP 6 참고)


if __name__ == "__main__":
    main()
