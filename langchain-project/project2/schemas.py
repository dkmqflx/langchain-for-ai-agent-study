# ═══════════════════════════════════════════════════════════════════
# STEP 5 · 스키마(schema): API가 주고받을 데이터의 '모양'만 모아 둔 파일.
# 로직은 한 줄도 없다 — "요청은 이런 모양이어야 하고, 응답은 이런 모양으로 주겠다"는 계약서다.
#
# 창구(router.py)와 계약서(schemas.py)를 나눠 두면, 이 API가 뭘 받고 뭘 주는지
# 이 파일 하나만 열어 확인할 수 있다. P1의 schemas.py와 같은 자리·같은 이름이다.
# ═══════════════════════════════════════════════════════════════════
from pydantic import BaseModel              # 요청/응답의 형태(스키마)를 선언하는 클래스


class ChatIn(BaseModel):                 # 요청 본문 스키마
    message: str                         # 필수. 없으면 FastAPI가 422로 자동 거절
    # 대화방 번호. 클라이언트가 보관하고 매 요청에 실어 보낸다.
    # 기본값을 둔 건 curl로 시험하기 편하라고 둔 것이고, 실제 서비스라면
    # 로그인 사용자 ID 등에서 서버가 직접 만들어 내는 게 맞다.
    session_id: str = "default"
    # 이 요청만 다른 모델로 돌리고 싶을 때 쓴다. 안 보내면 서버의 기본 모델(STEP 7).
    # 여기서는 모델을 비교해 보려고 클라이언트에 열어 두지만, 공개 API라면
    # 서버가 사용자 등급·실험 그룹을 보고 정하는 게 맞다.
    provider: str | None = None          # ← 추가(STEP 7)


class ChatOut(BaseModel):                # 응답 스키마 — STEP 4의 ChatReply를 HTTP로 옮긴 모양
    answer: str
    sources: list[str] = []
    confident: bool = False              # STEP 4에서 꺼낸 값. 여기 안 적으면 response_model이 조용히 잘라 낸다
    used_tools: list[str] = []
    blocked: bool = False                # 가드레일에 막혔는지 (클라이언트가 다르게 표시할 수 있게)
    session_id: str                      # 클라이언트가 다음 요청에 그대로 실어 보내라는 뜻
