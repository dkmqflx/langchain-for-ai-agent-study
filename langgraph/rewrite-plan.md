# langgraph 강의 HTML 슬림화 플랜

목표: 14개 강의 HTML의 파일 구성과 **모든 코드(`<pre>` 블록)는 그대로 유지**하면서,
설명 텍스트의 중복을 걷어내 각 파일을 대략 절반 분량으로 줄인다.
처음 읽는 사람의 이해 순서는 유지·강화한다. (설계 승인: 2026-08-15)

## 재작성 규칙

1. **개념 설명은 "처음 등장한 곳"이 소유한다.**
   뒤 파일에서 재등장하면 한 문장 요약 + 소유 파일 링크만 남긴다. (아래 소유권 지도 참고)
2. **복습 섹션은 다리 역할만.** "여기까지 온 길", "앞 장이 남긴 구멍" 류는
   2~3문장 연결 문단(+링크)으로 압축한다.
3. **산문 압축, 비유는 유지.** 새 개념을 처음 소개하는 비유는 남기고,
   같은 말을 굵은 글씨로 반복하는 재서술만 제거. "자주 걸리는 함정"·"📚 한 장 정리"는 유지하되
   파일 간 겹치는 함정은 처음 나온 곳에만 남긴다(단, 코드가 딸린 함정은 유지).
4. **`<pre>` 블록 전부 보존** (STEP별 코드·실행 결과·전체 코드). 순서와 내용 그대로.
5. **CSS는 공통 `langgraph/styles.css`로.** 각 파일의 `<style>` 블록을
   `<link rel="stylesheet" href="../styles.css" />`로 교체.
6. 사이드바 nav는 슬림화된 섹션 구조에 맞게 갱신하고, 모든 `#앵커`가 실제 `id`와 일치해야 한다.

## 개념 소유권 지도 (뒤 파일에서는 링크만)

| 개념 | 소유 파일 |
|---|---|
| LangGraph란 무엇인가, LangChain vs LangGraph | `01.Intro/1.html` |
| State·Node·Edge, `@tool`·`bind_tools`, `add_messages`, 조건부 엣지, `tool_calls`·`ToolMessage`, compile/invoke, Gemini vs OpenAI 응답 차이 | `02.LangGraph 기초/1.html` |
| 기획 3단계(요구사항→그래프→State), State 설계 원칙 | `02.LangGraph 기초/2.html` |
| Prompt Chaining, Parallelization(fan-out/in, reducer 충돌) | `03. LangGraph 응용/1-1.html` |
| Routing, Structured Outputs 활용 | `03. LangGraph 응용/1-2.html` |
| Evaluator-Optimizer, 루프·무한루프 방지 | `03. LangGraph 응용/2-1.html` |
| Orchestrator-Worker, Send API, Workflow vs Agent | `03. LangGraph 응용/2-2.html` |
| checkpointer·`thread_id`, `get_state_history`·스냅샷 | `04. Memory/1.html` |
| Time Travel(`update_state`), 정적 Interrupt(`interrupt_before`) | `04. Memory/2.html` |
| Store·namespace·put/get/search, 장기+단기 병행 | `04. Memory/3.html` |
| `interrupt()`·`Command(resume=)`, interrupts in tools, 노드 재실행 특성 | `05.Multi Agent/1.html` |
| 입력 검증 루프(human node 내부 검증) | `05.Multi Agent/1-2.html` |
| `Command(goto=)`, 조건부 엣지 vs Command | `05.Multi Agent/2.html` |
| Subgraph(부모/자식), State 분리·브릿지 노드 | `05.Multi Agent/3.html` |

## 검증 (챕터마다)

- `<pre>` 블록: 원본(git HEAD)과 개수·내용·순서 동일 — `verify_rewrite.py`
- nav의 모든 `href="#..."`가 파일 내 `id`와 매칭
- 브라우저 렌더링 확인, `index.html` 링크 유효

## 진행 체크리스트 (챕터마다 커밋) — 2026-08-15 전부 완료

- [x] 준비: `styles.css` 추출 + 본 플랜 문서
- [x] 01.Intro (763→461줄) — 기준 샘플
- [x] 02.LangGraph 기초 (1696→1279, 2322→1795)
- [x] 03. LangGraph 응용 (1681→1308, 1562→1113, 1889→1272, 1950→1455)
- [x] 04. Memory (1432→1037, 1839→1328, 1891→1307)
- [x] 05.Multi Agent (1982→1500, 1406→1060, 1524→1175, 1982→1634)
- [x] 검증: 전 파일 pre 블록 원본 대비 바이트 동일(총 430개), 파일 내 앵커·index.html·파일 간 링크 전부 유효

기타 기록: 04/1에서 복습 슬라이드 이미지 1장, 04/3에서 10장 제거(내용이 헤딩·코드와 중복).
02/2의 잘못된 다음 장 예고 문구를 실제 순서(3-1)에 맞게 수정. 문법 강조 `<script>`는 각 파일에 유지
(토큰 색상만 styles.css로 이동).
