# ═══════════════════════════════════════════════════════════════════
# STEP 5 · 스키마(schema): API가 주고받을 데이터의 '모양'만 모아 둔 파일.
# 로직은 한 줄도 없다 — "요청은 이런 모양이어야 하고, 응답은 이런 모양으로 주겠다"는 계약서다.
#
# main.py에서 왜 떼어 냈나: 지금 이 규모(모델 3개)라면 main.py에 그대로 둬도 아무 문제 없다.
# 그럼에도 나눠 두는 건, 엔드포인트가 늘어나면 스키마는 창구 코드보다 훨씬 빨리 불어나고
# 테스트·클라이언트 코드 등 여러 곳에서 같이 참조되는 부품이 되기 때문이다. 그때
# '계약서(schemas.py)'와 '창구(main.py)'가 나뉘어 있으면, API가 뭘 받고 뭘 주는지
# 이 파일 하나만 열어 확인할 수 있다. 서빙 코드의 흔한 관례라 미리 익혀 둘 만하다.
# ═══════════════════════════════════════════════════════════════════
from typing import Literal                  # 값을 '이 목록 중 하나'로 좁히는 타입 표기
from pydantic import BaseModel              # 요청/응답의 형태(스키마)를 선언하는 클래스

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
