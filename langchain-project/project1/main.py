# ═══════════════════════════════════════════════════════════════════
# STEP 5 · FastAPI 서빙: answer()를 HTTP API(POST /query)로 노출한다.
# pydantic 모델로 요청/응답 '스키마'를 선언 → 잘못된 입력은 자동 거부, /docs에 문서가 자동 생성.
# 이 순간 '노트북 코드'가 '서비스'가 된다.
#
# RAG 로직은 한 줄도 새로 쓰지 않는다 — STEP 2~4의 answer()를 HTTP로 감싸기만 한다.
# 실행: uv run uvicorn main:app --reload   /   자동 문서: http://localhost:8000/docs
# ═══════════════════════════════════════════════════════════════════
from typing import Literal                  # 값을 '이 목록 중 하나'로 좁히는 타입 표기
from fastapi import FastAPI                 # 파이썬 함수를 HTTP 창구로 바꿔 주는 웹 프레임워크
from pydantic import BaseModel              # 요청/응답의 형태(스키마)를 선언하는 클래스
from rag import answer                      # STEP 2~4에서 만든 핵심 함수 그대로 재사용

# 앱 객체 — 앞으로 만들 창구(엔드포인트)들이 여기에 등록된다.
# title은 /docs에 자동 생성되는 문서 페이지의 제목으로 표시된다.
app = FastAPI(title="P1 RAG Search Service")

class QueryIn(BaseModel):                   # 요청 본문 스키마
    question: str                           # 반드시 있어야 한다. 없으면 FastAPI가 422로 자동 거절
    # 그냥 str로 두면 "hybird" 같은 오타도 통과해 rag.py까지 흘러가 500이 된다.
    # 허용 목록으로 좁혀 두면 오타는 내 코드가 실행되기 전에 422로 거절되고, /docs에는 드롭다운으로 표시된다.
    mode: Literal["baseline", "hybrid", "rerank"] = "rerank"   # 검색 구성도 API로 고를 수 있게(실험용)

class Citation(BaseModel):                  # 근거 1건의 스키마
    source: str | None = None               # 파일 경로 (없을 수 있어 None 허용)
    page: int | None = None                 # 쪽 번호 (PDF에 따라 안 잡히는 경우가 있어 None 허용)

class QueryOut(BaseModel):                  # 응답 스키마: 답변 + 근거 목록
    answer: str
    citations: list[Citation]

@app.get("/health")                         # 헬스체크: 서버가 살아있는지 확인하는 운영 기본 엔드포인트
def health():
    return {"status": "ok"}                 # RAG를 태우지 않으므로 빠르고 공짜 — 몇 초마다 찔러볼 수 있다

@app.post("/query", response_model=QueryOut)   # POST /query. 응답을 QueryOut 형태로 검증·문서화
def query(body: QueryIn):                   # body는 요청 JSON이 QueryIn으로 자동 파싱·검증된 결과
    result = answer(body.question, mode=body.mode)   # RAG 실행(검색 + 생성 + 근거)
    return {"answer": result["answer"], "citations": result["citations"]}   # contexts는 응답에서 제외
