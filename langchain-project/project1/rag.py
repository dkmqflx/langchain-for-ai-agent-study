# ═══════════════════════════════════════════════════════════════════
# STEP 2 · 검색 + 생성 + 근거(Citation): 질문 → 관련 청크 검색 → LLM이 그 문맥으로만 답변.
# 핵심 원칙  (1) 모델은 검색된 CONTEXT만 근거로 답한다(환각 억제)
#           (2) 답과 함께 '어느 문서 몇 쪽'(source·page)을 근거로 돌려준다
# 이 answer()가 이후 서빙(STEP 3)·평가(STEP 5)·하이브리드(STEP 6)의 공통 진입점이 된다.
#
# STEP 1(ingest.py)이 이미 chroma_db에 벡터를 저장해 뒀으므로, 여기서는 '검색'만 한다.
#   ingest.py = 쓰기(Load→Split→Embed→Store) / rag.py = 읽기(Retrieve→Generate→Cite)
# ═══════════════════════════════════════════════════════════════════
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model          # 문자열 하나로 챗 모델을 만드는 헬퍼
from langchain_core.prompts import ChatPromptTemplate      # system/human 메시지 템플릿

load_dotenv()                                              # .env의 OPENAI_API_KEY 로드 → 임베딩·LLM 호출에 사용

# ── 검색기(retriever) 준비 ─────────────────────────────────────────
# 질문을 벡터로 바꿔 chroma_db에서 유사한 청크를 찾으려면, 그 벡터가 저장 때와
# '같은 규칙'으로 만들어져야 한다. 그래서 STEP 1과 반드시 같은 임베딩 모델을 쓴다.
emb = OpenAIEmbeddings(model="text-embedding-3-small")     # STEP 1과 '같은' 임베딩 모델이어야 검색이 맞는다
# 이미 인덱싱된 벡터DB를 연다(새로 저장하지 않고 읽기 전용). collection·경로는 STEP 1과 동일하게.
vs = Chroma(collection_name="p1_docs", persist_directory="chroma_db", embedding_function=emb)
retriever = vs.as_retriever(search_kwargs={"k": 4})        # 질문당 가장 유사한 청크 4개를 가져오는 검색기
llm = init_chat_model("gpt-5-mini")                        # 답변을 생성할 LLM

# 프롬프트: system=규칙(문서 밖 내용은 지어내지 말 것 = 환각 억제), human=질문 + 검색된 문맥.
# system에 규칙을 못 박아야 "모델이 아는 척 지어내는" 것을 막고, 근거를 강제할 수 있다.
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 제공된 CONTEXT만 근거로 답한다. "
     "CONTEXT에 근거가 없으면 '문서에서 찾을 수 없습니다'라고 답하고 추측하지 마라. "
     "답변 끝에 사용한 근거 번호를 [1][2]처럼 표기하라."),
    ("human", "질문: {question}\n\nCONTEXT:\n{context}"),   # {question}·{context}는 invoke 때 채워짐
])


def format_context(docs):
    """검색된 청크들을 '[번호] (source, page) + 본문' 형태의 한 문자열로 합친다.

    번호를 매겨두면 LLM이 답변 끝에 [1][2]로 어떤 근거를 썼는지 가리킬 수 있다.
    이때 각 청크의 metadata(source·page)는 STEP 1의 PyMuPDFLoader가 자동으로 넣어 둔 값이다.
    """
    return "\n\n".join(
        f"[{i + 1}] (source={d.metadata.get('source')}, page={d.metadata.get('page')})\n{d.page_content}"
        for i, d in enumerate(docs)                        # i=0,1,2... 각 청크에 번호 부여
    )


def answer(question: str):
    """질문 하나를 받아 검색 → 생성 → 근거까지 끝낸 결과 dict를 돌려준다."""
    docs = retriever.invoke(question)                      # (1) 검색: 질문과 유사한 청크 4개
    msg = PROMPT.invoke({"question": question, "context": format_context(docs)})  # (2) 프롬프트에 질문·문맥 주입
    resp = llm.invoke(msg)                                 # (3) 생성: LLM이 문맥 기반으로 답변
    return {
        "answer": resp.content,                            # 최종 답변 텍스트
        # 근거 목록: 답이 어느 파일 몇 쪽에서 왔는지 (신뢰성의 핵심 — 특히 금융 약관 도메인에서 필수)
        "citations": [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs],
        "contexts": [d.page_content for d in docs],        # 검색된 원문들 → STEP 5 RAGAS 평가에서 사용
    }


if __name__ == "__main__":
    import json

    # 검증 게이트를 한 번에 확인: (A) 문서에 있는 질문 → 답변+citations,
    #                          (B) 문서에 없는 질문 → '문서에서 찾을 수 없습니다'(환각 억제).
    demos = [
        "예금 계좌의 이자는 어떻게 지급되나요?",     # (A) 금융 약관 문서에 있는 내용 → 근거와 함께 답해야 함
        "비트코인의 오늘 시세는 얼마인가요?",         # (B) 약관 문서에 없는 내용 → 지어내지 말고 못 찾았다고 해야 함
    ]
    for q in demos:
        print(f"\n{'=' * 60}\n질문: {q}\n{'=' * 60}")
        # 한글이 깨지지 않게 ensure_ascii=False로 보기 좋게 출력
        print(json.dumps(answer(q), ensure_ascii=False, indent=2))
