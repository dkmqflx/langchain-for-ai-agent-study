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

load_dotenv()                                              # .env의 OPENAI_API_KEY 로드
# 검색은 '질문도 같은 방식으로 벡터화해서' 청크 벡터와 비교하는 것 → STEP 1에서 저장할 때 쓴
# 임베딩 모델과 반드시 같아야 한다. 다르면 좌표계가 달라져 엉뚱한 청크가 걸린다.
emb = OpenAIEmbeddings(model="text-embedding-3-small")
# 이미 인덱싱된 벡터DB를 연다(새로 저장하지 않고 읽기 전용). collection·경로는 STEP 1과 동일하게
vs = Chroma(collection_name="p1_docs", persist_directory="chroma_db", embedding_function=emb)
retriever = vs.as_retriever(search_kwargs={"k": 4})        # 질문당 가장 유사한 청크 4개를 가져오는 검색기
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

def answer(question: str):
    """질문 하나를 받아 검색 → 생성 → 근거까지 끝낸 결과 dict를 돌려준다."""
    docs = retriever.invoke(question)                      # (1) 검색: 질문과 유사한 청크 4개
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
