# ═══════════════════════════════════════════════════════════════════
# STEP 1 · 도구(Tool) 정의: Agent가 골라 쓸 수 있는 능력들을 함수로 만든다.
# 여기서 중요한 건 '함수가 잘 도는가'보다 'LLM이 언제 이걸 써야 하는지 아는가'다.
# → 그 정보는 오직 함수 이름 · 타입 힌트 · docstring 세 가지로만 전달된다.
# ═══════════════════════════════════════════════════════════════════
import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo          # 파이썬 3.9+ 내장. 시간대를 다룰 때 쓴다

from langchain.tools import tool       # 평범한 함수를 '도구'로 등록해 주는 데코레이터
from ddgs import DDGS                  # DuckDuckGo 검색 클라이언트 (API 키 불필요)


# ── 도구 1 · 오늘 날짜 ────────────────────────────────────────────
# LLM은 '오늘'이 며칠인지 모른다. 학습 데이터가 끝난 시점까지만 알고,
# 지금 이 순간이 언제인지는 알 방법이 없다 → 알려 주는 도구가 필요하다.
@tool
def get_today() -> str:
    """오늘 날짜와 현재 시각(한국 시간)을 알려준다.

    '오늘', '지금', '이번 주', '며칠 남았는지' 처럼 현재 시점을 알아야 답할 수 있는
    질문에는 다른 무엇보다 먼저 이 도구를 호출한다.
    """
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    weekday = "월화수목금토일"[now.weekday()]      # weekday()는 월=0 … 일=6
    return now.strftime(f"%Y-%m-%d %H:%M ({weekday}요일)")


# ── 도구 2 · 계산기 ──────────────────────────────────────────────
# 아래 세 줄은 '안전한 계산기'를 만들기 위한 재료다.
# 파이썬의 eval()을 쓰면 한 줄로 끝나지만, 그건 절대 하면 안 된다 (아래 경고 참고).
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,   # USub = -5 같은 단항 마이너스
}

def _eval_node(node):
    """수식을 해석한 나무(AST)를 타고 내려가며 계산한다. 허용한 연산만 통과시킨다."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value                                        # 숫자면 그대로
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:     # a + b 같은 2항 연산
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:   # -a 같은 1항 연산
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("허용되지 않은 식")                           # 그 외는 전부 거부

@tool
def calculate(expression: str) -> str:
    """사칙연산과 거듭제곱으로 이루어진 수식을 계산한다.

    예: "1234 * 5678", "(120 - 45) / 3", "2 ** 10"
    숫자 계산이 필요하면 직접 암산하지 말고 반드시 이 도구를 사용한다.
    """
    try:
        # ast.parse(..., mode="eval") = 문자열을 '실행'하지 않고 '구조로 분해'만 한다.
        # .body가 그 구조의 뿌리 노드이고, 우리가 만든 _eval_node가 이걸 타고 내려가며 계산한다.
        return str(_eval_node(ast.parse(expression, mode="eval").body))
    except Exception:
        # 도구는 예외를 밖으로 던지지 않는다 — 자세한 이유는 아래 경고 참고
        return f"계산할 수 없는 식입니다: {expression}"


# ── 도구 3 · 웹 검색 ─────────────────────────────────────────────
@tool
def web_search(query: str, max_results: int = 3) -> str:
    """웹에서 최신 정보를 검색한다.

    뉴스, 시세, 최근에 일어난 일처럼 모델이 알 수 없는 '지금'의 정보가 필요할 때 사용한다.
    사내 문서나 약관의 내용은 이 도구로 찾을 수 없다.
    """
    try:
        hits = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return f"웹 검색에 실패했습니다: {e}"
    if not hits:
        return "검색 결과가 없습니다."
    # LLM이 읽을 문자열로 정리한다. 출처(url)를 함께 넣어야 답변에 근거를 달 수 있다.
    return "\n\n".join(
        f"- {h.get('title')}\n  {h.get('body')}\n  ({h.get('href')})" for h in hits
    )
