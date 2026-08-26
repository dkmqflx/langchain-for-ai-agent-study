# ═══════════════════════════════════════════════════════════════════
# STEP 5 · 라우터(router): HTTP 창구(엔드포인트)들을 모아 둔 파일.
# 요청을 받아 ask()에 넘기고 결과를 돌려주는 것까지가 전부다 —
# Agent 로직은 한 줄도 없다. STEP 1~4의 결과를 HTTP로 감싸기만 한다.
# ═══════════════════════════════════════════════════════════════════
from fastapi import APIRouter

from agent import ask                    # STEP 1~4에서 만든 핵심 함수 그대로 재사용
from schemas import ChatIn, ChatOut      # 요청·응답의 '모양' — 선언은 schemas.py 한곳에 모아 둔다

# APIRouter = 창구 묶음. 여기에 붙인 창구들이 main.py의 app에 통째로 등록된다(include_router).
# @app.get 대신 @router.get을 쓰는 것 말고는 창구를 적는 방식이 똑같다.
router = APIRouter()


@router.get("/health")                   # 서버가 살아있는지 확인하는 운영 기본 엔드포인트
def health():
    return {"status": "ok"}              # Agent를 안 태우므로 빠르고 공짜다


@router.post("/chat", response_model=ChatOut)
def chat(body: ChatIn):
    # provider까지 그대로 전달한다. 창구가 하는 일은 여전히 '받아서 넘기기'뿐이다(STEP 7).
    result = ask(body.message, session_id=body.session_id, provider=body.provider)
    return {**result, "session_id": body.session_id}         # 결과에 세션 번호를 얹어 돌려준다
