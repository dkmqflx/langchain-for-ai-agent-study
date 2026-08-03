# P1 · 프로덕션급 RAG 문서 검색 서비스

[`docs/index.html`](docs/index.html)의 STEP 0~9를 따라 만드는 실습 프로젝트입니다.
먼저 랭체인만으로 검색기를 **baseline → Hybrid → Reranker**로 발전시키고(STEP 1~4),
그 다음 FastAPI로 서빙 + Langfuse 추적 + RAGAS로 세 구성을 정량 비교합니다(STEP 5~7).

> STEP 9에서 문제정의 · 아키텍처 · 지표 before/after · 배운 점으로 이 README를 완성합니다.

## 진행 상태
- [x] **STEP 0** · [프로젝트 셋업 & 준비물](docs/0.html) (환경/의존성/키)
- [ ] STEP 1 · [문서 인제스천](docs/1.html) (`ingest.py`)
- [ ] STEP 2 · [검색 + 생성 + 근거](docs/2.html) (`rag.py`)
- [ ] STEP 3 · [Hybrid 검색](docs/3.html) — BM25 + Dense, mode 팩토리 (`rag.py`)
- [ ] STEP 4 · [Reranker](docs/4.html) — 2-stage retrieval (`rag.py`)
- [ ] STEP 5 · [FastAPI 서빙](docs/5.html) (`main.py`)
- [ ] STEP 6 · [Langfuse 관측성](docs/6.html) (`rag.py`)
- [ ] STEP 7 · [골든셋 + RAGAS 세 구성 비교](docs/7.html) (`eval_data.json`, `evaluate.py`)
- [ ] STEP 8 · [파라미터 튜닝](docs/8.html) (선택)
- [ ] STEP 9 · [README & 포트폴리오화](docs/9.html)

## 실행 방법 (셋업 후)
```bash
uv sync                       # 의존성 설치
# data/ 에 PDF 3~10개 넣기
uv run python ingest.py       # STEP 1: 인덱싱
uv run python rag.py          # STEP 2: 검색+생성 확인
uv run uvicorn main:app --reload   # STEP 5: API 서버
```

## 준비물
- `.env` — `OPENAI_API_KEY`(필수), `LANGFUSE_*`(STEP 6에서 채움)
- `data/*.pdf` — 질문할 도메인 문서 (직접 넣기)
