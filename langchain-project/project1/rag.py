# ═══════════════════════════════════════════════════════════════════
# STEP 2 · 검색 + 생성 + 근거(Citation): 질문 → 관련 청크 검색 → LLM이 그 문맥으로만 답변.
# 핵심 원칙  (1) 모델은 검색된 CONTEXT만 근거로 답한다(환각 억제)
#           (2) 답과 함께 '어느 문서 몇 쪽'(source·page)을 근거로 돌려준다
# 이 answer()가 이후 하이브리드(STEP 3)·서빙(STEP 5)·평가(STEP 7)의 공통 진입점이 된다.
#
# STEP 1(ingest.py)은 '쓰기'였다면 여기는 '읽기'다 — 이미 저장된 벡터DB를 열어 검색만 한다.
# ═══════════════════════════════════════════════════════════════════
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model          # 문자열 하나로 챗 모델을 만드는 헬퍼
from langchain_core.prompts import ChatPromptTemplate      # system/human 메시지 템플릿
from langchain_community.retrievers import BM25Retriever   # 키워드(단어 일치) 기반 검색기
# langchain_classic = '구식'이 아니라 '이사 간 옛 짐'. LangChain 1.0에서 langchain 패키지를
# 에이전트 중심으로 다시 지으면서, 0.x의 나머지(chains·retrievers 등)를 이 패키지로 옮겼다.
# EnsembleRetriever는 v1 네이티브 대안이 없다(core·community 어디에도 없음). deprecated도 아니다.
from langchain_classic.retrievers import EnsembleRetriever # 검색기 여러 개를 합쳐 순위를 섞어 주는 도구
from ingest import build_chunks                            # STEP 1의 '청크 만들기' 함수를 그대로 재사용

load_dotenv()                                              # .env의 OPENAI_API_KEY 로드
# 검색은 '질문도 같은 방식으로 벡터화해서' 청크 벡터와 비교하는 것 → STEP 1에서 저장할 때 쓴
# 임베딩 모델과 반드시 같아야 한다. 다르면 좌표계가 달라져 엉뚱한 청크가 걸린다.
emb = OpenAIEmbeddings(model="text-embedding-3-small")
# 이미 인덱싱된 벡터DB를 연다(새로 저장하지 않고 읽기 전용). collection·경로는 STEP 1과 동일하게
vs = Chroma(collection_name="p1_docs", persist_directory="chroma_db", embedding_function=emb)
llm = init_chat_model("gpt-5-mini")                        # 답변을 생성할 LLM

# 프롬프트: system=규칙(문서 밖 내용은 지어내지 말 것 = 환각 억제), human=질문 + 검색된 문맥
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 제공된 CONTEXT만 근거로 답한다. "
     "CONTEXT에 근거가 없으면 '문서에서 찾을 수 없습니다'라고 답하고 추측하지 마라. "
     "답변 끝에 사용한 근거 번호를 [1][2]처럼 표기하라."),
    ("human", "질문: {question}\n\nCONTEXT:\n{context}"),   # {question}·{context}는 invoke 때 채워짐
])

def format_context(docs):
    """검색된 청크들을 '[번호] (source, page) + 본문' 형태의 한 문자열로 합친다.

    LLM에게는 Document 객체가 아니라 '하나의 긴 문자열'로 문맥을 넘겨야 한다.
    이때 번호를 매겨두면 LLM이 답변 끝에 [1][2]로 어떤 근거를 썼는지 가리킬 수 있다.
    """
    return "\n\n".join(
        f"[{i+1}] (source={d.metadata.get('source')}, page={d.metadata.get('page')})\n{d.page_content}"
        for i, d in enumerate(docs)                        # i=0,1,2... 각 청크에 번호 부여
    )

# ═══════════════════════════════════════════════════════════════════
# STEP 3 · Hybrid 검색: 키워드 검색(BM25) + 의미 검색(임베딩)을 앙상블해 검색 누락을 줄인다.
# 임베딩은 '의미'에 강하나 정확한 용어·수치·희귀 단어(조항번호·상품명 등)에 약하고,
# BM25는 '정확한 단어 일치'에 강하다 → 둘을 합쳐 서로의 빈틈을 메운다.
#
# 동시에 검색기를 'mode 팩토리'로 재구성한다: baseline | hybrid  (STEP 4에서 rerank 추가)
# → STEP 7에서 같은 골든셋으로 여러 구성을 나란히 채점해 비교하기 위한 뼈대.
# ═══════════════════════════════════════════════════════════════════
# BM25는 벡터가 아니라 '원문에 그 단어가 몇 번 나오나'를 세는 방식이라 텍스트 청크가 그대로 필요하다.
# 그런데 Chroma에는 벡터만 들어 있으므로, PDF를 다시 읽어 청크를 만들어 둔다.
#
# 이 줄이 함수 '바깥'에 있는 게 중요하다 → 파일을 불러올 때 딱 한 번만 실행된다.
# answer() 안에 넣으면 질문할 때마다 PDF 전체를 다시 읽게 되어 몹시 느려진다.
chunks = build_chunks()

def _build_retriever(mode: str):
    """mode 이름(문자열)을 받아 그에 해당하는 검색기 객체를 만들어 돌려준다.

    검색 전략을 문자열 하나로 갈아끼울 수 있게 만드는 것이 이 리팩터링의 핵심이다.
    """
    dense = vs.as_retriever(search_kwargs={"k": 4})        # 의미(임베딩) 검색 — STEP 2에서 쓰던 그 검색기

    if mode == "baseline":
        return dense                                       # STEP 2 그대로: 의미 검색만 쓴다

    bm25 = BM25Retriever.from_documents(chunks)            # 청크들로 키워드 색인을 만든다
    bm25.k = 4                                             # BM25가 돌려줄 청크 수 (의미 검색과 똑같이 4개)

    if mode == "hybrid":
        # EnsembleRetriever = 두 검색기에게 각자 top-4를 뽑게 한 뒤,
        #                     weights 비율로 점수를 섞어 하나의 최종 순위를 만든다.
        # [0.4, 0.6] → 앞의 bm25에 40%, 뒤의 dense에 60% 비중.
        #   의미 검색이 평소 더 안정적이라 비중을 더 주고,
        #   BM25는 의미 검색이 놓치는 순간(조항번호·고유용어)을 메우는 보험 역할을 맡는다.
        return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])

    # 오타 등으로 없는 이름이 들어오면 조용히 넘어가지 말고 즉시 에러를 내 알린다
    raise ValueError(f"unknown mode: {mode}")

_retrievers = {}   # 이미 만들어 둔 검색기를 담는 보관함 — {mode 이름: 검색기 객체}

def get_retriever(mode: str = "hybrid"):
    """검색기를 돌려준다. 처음 요청받은 mode만 새로 만들고, 그다음부터는 보관함에서 꺼내 쓴다."""
    # BM25Retriever.from_documents()는 호출할 때마다 색인을 처음부터 새로 만들어 느리다.
    # 질문이 올 때마다 반복하지 않도록, 한 번 만든 것을 보관해 두고 재사용한다.
    if mode not in _retrievers:                            # 보관함에 없으면 = 이 mode는 처음 쓰는 것
        _retrievers[mode] = _build_retriever(mode)         # 그때만 만들어 보관한다
    return _retrievers[mode]

# 참고: 이름 앞의 _ 는 파이썬에서 '바깥에서 직접 쓰지 마세요'라는 관례적 표시다.
#      바깥(main.py·evaluate.py)에 열어 둔 문은 get_retriever() 하나뿐이다.

def answer(question: str, mode: str = "hybrid"):           # mode 인자 추가 — 기본은 hybrid
    """질문 하나를 받아 검색 → 생성 → 근거까지 끝낸 결과 dict를 돌려준다."""
    docs = get_retriever(mode).invoke(question)            # (1) 검색: mode에 해당하는 검색기로
    msg = PROMPT.invoke({"question": question, "context": format_context(docs)})  # (2) 프롬프트에 질문·문맥 주입
    resp = llm.invoke(msg)                                 # (3) 생성: LLM이 문맥 기반으로 답변
    return {
        "answer": resp.content,                            # 최종 답변 텍스트
        # 근거 목록: 답이 어느 파일 몇 쪽에서 왔는지 (신뢰성의 핵심)
        "citations": [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs],
        "contexts": [d.page_content for d in docs],   # 검색된 원문들 → STEP 7 RAGAS 평가에서 사용
    }

# 검증 게이트용 기본 질문: 신용카드표준약관 제40·41조에 실제로 있는 내용
DEFAULT_QUESTION = "신용카드 회원의 카드 분실·도난 시 책임은 어떻게 되나요?"

if __name__ == "__main__":
    import json
    # 인자를 주면 그 질문으로 실행 → 검증 게이트의 '문서에 없는 질문'도 파일 수정 없이 확인 가능
    #   uv run python rag.py "문서에 없는 질문"
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    # 결과를 보기 좋게 출력(한글 안 깨지게 ensure_ascii=False)
    print(json.dumps(answer(question), ensure_ascii=False, indent=2))
