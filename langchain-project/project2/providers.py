# ═══════════════════════════════════════════════════════════════════
# STEP 7 · 모델 창구(provider factory): 이름 하나 → 모델 객체.
#
# Agent는 "모델 하나 주세요"라고만 하고, 어떤 회사의 어떤 모델이 오는지는 이 파일이 정한다.
# 모델 이름이 agent.py 한복판에 박혀 있으면 모델을 바꿀 때마다 코드를 고치고 다시 배포해야 하지만,
# 여기로 빼 두면 .env 한 줄(LLM_PROVIDER) 또는 요청 한 칸(provider)으로 갈아탈 수 있다.
#
# P1의 get_retriever(mode)와 같은 모양이다 — 문자열을 받아 객체를 돌려주고, 만든 건 보관한다.
# ═══════════════════════════════════════════════════════════════════
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()                            # OPENAI_API_KEY·GOOGLE_API_KEY를 읽는다

# 이름 → "회사:모델". 모델을 늘리는 일은 여기 한 줄 추가가 전부다.
# 모델 이름은 몇 달이면 새 버전이 나오고 옛 이름이 사라진다 — 그래서 코드 곳곳이 아니라
# 딕셔너리 한곳에 모아 둔다. model not found가 뜨면 이 표만 고치면 된다.
PROVIDERS = {
    "openai": "openai:gpt-5-mini",
    "gemini": "google_genai:gemini-2.5-flash",
    # "claude": "anthropic:claude-sonnet-4-5",   ← 키만 있으면 이 한 줄로 늘어난다
}

# 평소에 쓸 모델. .env에 LLM_PROVIDER=gemini 라고 쓰면 서버가 뜰 때 갈아탄다.
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

_models: dict = {}                       # 한 번 만든 모델은 보관해 두고 재사용한다


def get_model(name: str | None = None):
    """provider 이름을 받아 모델 객체를 돌려준다. 이름을 안 주면 기본값(DEFAULT_PROVIDER).

    모르는 이름은 조용히 기본 모델로 넘기지 않고 바로 알린다.
    오타(gpt5)를 기본값으로 처리해 버리면 "분명 gemini로 보냈는데 왜 openai 요금이 나오지"
    같은 문제를 나중에 대시보드에서 찾아야 한다.
    """
    name = name or DEFAULT_PROVIDER
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider: {name} (가능: {list(PROVIDERS)})")
    if name not in _models:
        # "회사:모델" 문자열을 보고 알맞은 패키지(langchain-openai·langchain-google-genai)를 고른다.
        _models[name] = init_chat_model(PROVIDERS[name])
    return _models[name]
