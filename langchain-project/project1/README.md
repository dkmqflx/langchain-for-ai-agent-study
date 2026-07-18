# P1 · 프로덕션급 RAG 문서 검색 서비스

`project1.html`의 STEP 0~9를 따라 만드는 실습 프로젝트입니다.
문서에 질문하면 **근거(citation)와 함께** 답하는 RAG를 FastAPI로 서빙하고,
RAGAS로 검색 품질을 정량 측정한 뒤 Hybrid + Reranker로 개선합니다.

> STEP 9에서 문제정의 · 아키텍처 · 지표 before/after · 배운 점으로 이 README를 완성합니다.

## 진행 상태
- [x] **STEP 0** · 프로젝트 셋업 & 준비물 (환경/의존성/키)
- [ ] STEP 1 · 문서 인제스천 (`ingest.py`)
- [ ] STEP 2 · 검색 + 생성 + 근거 (`rag.py`)
- [ ] STEP 3 · FastAPI 서빙 (`main.py`)
- [ ] STEP 4 · Langfuse 관측성 (`rag.py`)
- [ ] STEP 5 · 골든셋 + RAGAS 베이스라인 (`eval_data.json`, `evaluate.py`)
- [ ] STEP 6 · Hybrid 검색 (BM25 + Dense)
- [ ] STEP 7 · Reranker (2-stage)
- [ ] STEP 8 · 파라미터 튜닝 (선택)
- [ ] STEP 9 · README & 포트폴리오화

## 실행 방법 (셋업 후)
```bash
uv sync                       # 의존성 설치
# data/ 에 PDF 3~10개 넣기
uv run python ingest.py       # STEP 1: 인덱싱
uv run python rag.py          # STEP 2: 검색+생성 확인
uv run uvicorn main:app --reload   # STEP 3: API 서버
```

## 준비물
- `.env` — `OPENAI_API_KEY`(필수), `LANGFUSE_*`(STEP 4에서 채움)
- `data/*.pdf` — 질문할 도메인 문서 (직접 넣기)
