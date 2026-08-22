# ═══════════════════════════════════════════════════════════════════
# STEP 5 · FastAPI 서빙: answer()를 HTTP API(POST /query)로 노출한다.
# 주고받을 데이터의 '스키마'는 schemas.py에 선언해 두고, 여기서는 '창구'만 연다.
# 스키마 덕분에 잘못된 입력은 자동 거부되고, /docs에 문서가 자동 생성된다.
# 이 순간 '노트북 코드'가 '서비스'가 된다.
#
# RAG 로직은 한 줄도 새로 쓰지 않는다 — STEP 2~4의 answer()를 HTTP로 감싸기만 한다.
# 실행: uv run uvicorn main:app --reload   /   자동 문서: http://localhost:8000/docs
# ═══════════════════════════════════════════════════════════════════
from fastapi import FastAPI                 # 파이썬 함수를 HTTP 창구로 바꿔 주는 웹 프레임워크
from schemas import QueryIn, QueryOut       # 요청·응답의 '모양' — 선언은 schemas.py 한곳에 모아 둔다
from rag import answer                      # STEP 2~4에서 만든 핵심 함수 그대로 재사용

# 앱 객체 — 앞으로 만들 창구(엔드포인트)들이 여기에 등록된다.
# title은 /docs에 자동 생성되는 문서 페이지의 제목으로 표시된다.
app = FastAPI(title="P1 RAG Search Service")

@app.get("/health")                         # 헬스체크: 서버가 살아있는지 확인하는 운영 기본 엔드포인트
def health():
    return {"status": "ok"}                 # RAG를 태우지 않으므로 빠르고 공짜 — 몇 초마다 찔러볼 수 있다

@app.post("/query", response_model=QueryOut)   # POST /query. 응답을 QueryOut 형태로 검증·문서화
def query(body: QueryIn):                   # body는 요청 JSON이 QueryIn으로 자동 파싱·검증된 결과
    result = answer(body.question, mode=body.mode)   # RAG 실행(검색 + 생성 + 근거)
    return {"answer": result["answer"], "citations": result["citations"]}   # contexts는 응답에서 제외
