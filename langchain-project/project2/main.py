# ═══════════════════════════════════════════════════════════════════
# STEP 5 · FastAPI 서빙: ask()를 HTTP(POST /chat)로 노출한다.
# 이 파일이 하는 일은 둘뿐이다 — 앱을 만들고, 창구 묶음을 등록한다.
#   schemas.py … 주고받을 데이터의 모양(계약서)
#   router.py  … 창구와 그 안에서 하는 일
#
# 실행: uv run uvicorn main:app --reload --port 8001
#   ※ 포트 8001! P1이 8000을 쓰고 있으므로 겹치면 안 된다.
# 자동 문서: http://localhost:8001/docs
# ═══════════════════════════════════════════════════════════════════
from fastapi import FastAPI

from router import router                # 창구 묶음 (router.py에서 만든 APIRouter)

# 앱 객체 — 이 변수 이름(app)을 실행 명령(uvicorn main:app)에서 쓴다.
# title은 /docs에 자동 생성되는 문서 페이지의 제목으로 표시된다.
app = FastAPI(title="P2 Agent API")

# 창구 묶음을 앱에 등록한다. 이 한 줄로 /health와 /chat이 열린다.
app.include_router(router)
