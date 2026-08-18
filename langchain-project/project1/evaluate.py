# ═══════════════════════════════════════════════════════════════════
# STEP 7 · RAGAS 평가: '좋아진 것 같다'를 숫자로 바꾼다.
# 골든셋(eval_data.json)에 answer(question, mode=...)를 돌려, 4개 지표로 채점한다:
#   faithfulness(환각 여부) · answer_relevancy(질문 적합) · context_precision(문맥 정확) · context_recall(누락 없음)
#
# RAGAS는 채점을 LLM에게 시킨다(LLM-as-a-judge). 예를 들어 faithfulness는 우리 답을
# '주장 단위'로 쪼갠 뒤 주장마다 "이게 검색된 문맥에 실제로 적혀 있나?"를 심판 LLM에게 묻고,
# 근거 있는 주장의 비율을 점수로 낸다 → 그래서 점수가 0~1 소수이고, 평가에도 API 비용이 든다.
#
# 실행:  uv run python evaluate.py baseline   → eval_baseline.csv
#       uv run python evaluate.py hybrid     → eval_hybrid.csv
#       uv run python evaluate.py rerank     → eval_rerank.csv
# 세 번 실행해 나온 지표를 한 표에 모으면 '개선 스토리'가 완성된다.
# ═══════════════════════════════════════════════════════════════════
import json, sys
from datasets import Dataset                                   # RAGAS에 넣을 표(데이터셋) 형식
from ragas import evaluate                                     # 채점 실행 함수
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall  # 지표 4종
from ragas.llms import LangchainLLMWrapper                     # RAGAS가 LangChain LLM을 쓰게 감싸는 어댑터
from ragas.embeddings import LangchainEmbeddingsWrapper        # RAGAS가 쓸 임베딩 어댑터
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from rag import answer                                         # 평가 대상: 우리 RAG 파이프라인

mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"        # 어떤 검색 구성(mode)을 채점할지
rows = json.load(open("eval_data.json", encoding="utf-8"))     # 골든셋 로드: [{question, ground_truth}, ...]

records = []
for i, r in enumerate(rows, 1):                                # 질문마다 실제로 RAG를 돌려 채점용 행을 만든다
    print(f"  [{mode}] {i}/{len(rows)} {r['question']}")       # 몇 번째를 돌고 있는지 (질문당 검색+LLM 호출이 일어난다)
    out = answer(r["question"], mode=mode)                     # STEP 3~4의 mode 팩토리가 여기서 쓰인다
    records.append({
        "question": r["question"],                             # 질문
        "answer": out["answer"],                               # 우리 시스템이 낸 답
        "contexts": out["contexts"],                           # 검색된 문맥(정확/누락 지표 계산에 필요)
        "ground_truth": r["ground_truth"],                     # 사람이 만든 정답(기준)
    })

result = evaluate(                                             # 4개 지표로 일괄 채점
    Dataset.from_list(records),
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    # 채점을 수행하는 심판 LLM. bypass_temperature=True가 필요한 이유:
    # RAGAS는 채점을 일관되게 만들려고 심판 LLM의 temperature를 0.01로 낮춰 호출하는데,
    # gpt-5 계열은 temperature 기본값(1) 외의 값을 거부한다 → 400 에러로 전 지표가 NaN이 된다.
    # 이 플래그를 켜면 RAGAS가 temperature를 건드리지 않고 모델 기본값 그대로 호출한다.
    llm=LangchainLLMWrapper(init_chat_model("gpt-5-mini"), bypass_temperature=True),
    embeddings=LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small")),  # 유사도 계산용 임베딩
)
print(f"[{mode}]", result)                                     # 지표 평균을 화면에 출력
result.to_pandas().to_csv(f"eval_{mode}.csv", index=False)     # 문항별 상세 점수를 CSV로 저장 → 구성 간 비교
# 평균만 보면 '왜' 떨어졌는지 모른다. 어떤 질문이 점수를 깎아먹었는지는 이 CSV를 열어야 보인다.
