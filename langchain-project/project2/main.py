# ═══════════════════════════════════════════════════════════════════
# STEP 5 · FastAPI 서빙: ask()를 HTTP(POST /chat)로 노출한다.
# Agent 로직은 한 줄도 새로 쓰지 않는다 — STEP 1~4의 결과를 HTTP로 감싸기만 한다.
#
# 실행: uv run uvicorn main:app --reload --port 8001
#   ※ 포트 8001! P1이 8000을 쓰고 있으므로 겹치면 안 된다.
# 자동 문서: http://localhost:8001/docs
# ═══════════════════════════════════════════════════════════════════
from fastapi import FastAPI
from pydantic import BaseModel

from agent import ask                    # STEP 1~4에서 만든 핵심 함수 그대로 재사용

app = FastAPI(title="P2 Agent API")


class ChatIn(BaseModel):                 # 요청 본문 스키마
    message: str                         # 필수. 없으면 FastAPI가 422로 자동 거절
    # 대화방 번호. 클라이언트가 보관하고 매 요청에 실어 보낸다.
    # 기본값을 둔 건 curl로 시험하기 편하라고 둔 것이고, 실제 서비스라면
    # 로그인 사용자 ID 등에서 서버가 직접 만들어 내는 게 맞다(아래 경고 참고).
    session_id: str = "default"


class ChatOut(BaseModel):                # 응답 스키마 — STEP 4의 ChatReply를 HTTP로 옮긴 모양
    answer: str
    sources: list[str] = []
    confident: bool = False              # STEP 4에서 꺼낸 값. 여기 안 적으면 response_model이 조용히 잘라 낸다
    used_tools: list[str] = []
    blocked: bool = False                # 가드레일에 막혔는지 (클라이언트가 다르게 표시할 수 있게)
    session_id: str                      # 클라이언트가 다음 요청에 그대로 실어 보내라는 뜻


@app.get("/health")                      # 서버가 살아있는지 확인하는 운영 기본 엔드포인트
def health():
    return {"status": "ok"}              # Agent를 안 태우므로 빠르고 공짜다


@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn):
    result = ask(body.message, session_id=body.session_id)   # Agent 실행
    return {**result, "session_id": body.session_id}         # 결과에 세션 번호를 얹어 돌려준다
