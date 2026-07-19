# ═══════════════════════════════════════════════════════════════════
# STEP 1 · 인제스천(indexing): 원본 PDF를 '검색 가능한 벡터'로 바꿔 저장한다.
# 파이프라인  Load(읽기) → Split(쪼개기) → Embed(벡터화) → Store(저장)
# 이 스크립트를 한 번 실행해 두면, 이후 rag.py는 저장된 벡터DB에서 '검색'만 하면 된다.
#
# 아래 import가 각각 파이프라인의 어느 단계를 맡는지:
#   PyMuPDFLoader                        → Load  : PDF를 페이지별 Document로 읽기
#   RecursiveCharacterTextSplitter       → Split : 긴 텍스트를 청크로 분할
#   OpenAIEmbeddings                     → Embed : 텍스트를 의미 벡터로 변환
#   CacheBackedEmbeddings/LocalFileStore → Embed : 임베딩 캐시(재계산·재과금 방지)
#   Chroma                               → Store : 벡터 저장 + 유사도 검색(벡터DB)
# ═══════════════════════════════════════════════════════════════════
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import RateLimitError
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_chroma import Chroma

load_dotenv()                    # .env를 읽어 환경변수로 등록 → OpenAI 임베딩 호출에 쓸 API 키가 준비됨
DOCS_DIR = Path("data")          # 원본 PDF들이 들어 있는 폴더
PERSIST_DIR = "chroma_db"        # 벡터DB를 저장할 폴더(디스크에 남아 다음 실행에서 그대로 재사용)
COLLECTION = "p1_docs"           # 벡터DB 안에서 이 문서 묶음을 가리키는 이름표(테이블 이름 같은 것)


def build_chunks():
    """data/의 모든 PDF를 읽어 '청크(검색 단위 조각)' 리스트로 반환한다.  (Load → Split)

    청크 = 검색의 기본 단위가 되는 문서 조각. 검색은 문서 전체가 아니라 이 조각
    단위로 이뤄지므로, '어떻게 쪼개는가'가 검색 품질을 크게 좌우한다.
    """
    docs = []
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):   # data/의 모든 .pdf를 이름순으로 하나씩 순회
        # PyMuPDFLoader(...).load() → PDF 1개를 '페이지별 Document' 리스트로 변환.
        # 각 Document.metadata에 source(파일 경로)·page(쪽 번호)가 자동으로 담긴다
        # → 이 값이 나중에 답변의 '근거(citation: 어느 문서 몇 쪽)'로 그대로 쓰인다.
        docs.extend(PyMuPDFLoader(str(pdf)).load())
    # chunk_size=1000  : 청크 하나의 최대 길이(글자). 크면 관련 없는 내용까지 섞여 노이즈↑, 작으면 문맥이 끊김
    # chunk_overlap=150: 이웃한 청크를 150자 겹치게 → 경계에서 문장이 잘려 뜻이 사라지는 것을 완화
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)        # 페이지 Document들을 청크 단위 Document로 다시 쪼개 반환


def get_embeddings():
    """청크를 벡터로 바꿀 임베딩기를 만든다. 단, 캐시를 씌워 같은 청크의 중복 계산을 막는다.  (Embed)"""
    base = OpenAIEmbeddings(model="text-embedding-3-small")   # 실제 벡터를 만드는 OpenAI 임베딩 모델(가볍고 저렴한 소형)
    store = LocalFileStore("./cache/embeddings")             # 임베딩 결과(벡터)를 저장해 둘 로컬 캐시 폴더
    # 같은 청크는 다시 임베딩하지 않고 캐시에서 꺼내 씀 → 스크립트를 재실행해도 비용·시간 절약
    # namespace=base.model : 모델이 다르면 벡터도 다르므로 캐시를 '모델 이름'별로 분리(캐시 섞임 방지)
    return CacheBackedEmbeddings.from_bytes_store(base, store, namespace=base.model)


# 신규 결제 계정은 임베딩 '분당 토큰(TPM)' 한도가 낮다(예: 40,000/분).
# 164청크(~11.6만 토큰)를 한 번에 보내면 429(rate_limit_exceeded)가 난다.
# → 작은 배치로 나눠 한도 아래로 보내고, 배치 사이에 잠깐 쉬어 분당 한도를 지킨다.
BATCH_SIZE = 40                  # 배치당 청크 수(대략 3만 토큰 이하 → 40k TPM 한도 아래로 유지)
COOLDOWN_SEC = 60                # 다음 배치 전 대기(초) → 분당 토큰 한도 창을 리셋


def add_in_batches(vs, chunks):
    """청크를 작은 배치로 나눠 임베딩·저장한다. 429(TPM 초과)가 나면 잠시 쉬고 재시도한다."""
    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        while True:                                  # 이 배치가 들어갈 때까지(성공 or 재시도)
            try:
                vs.add_documents(batch)              # 임베딩 → Chroma에 저장(디스크에 자동 반영)
                break
            except RateLimitError:                   # 분당 토큰 한도 초과 → 창이 리셋될 때까지 대기 후 재시도
                print(f"    TPM 한도 도달 — {COOLDOWN_SEC}s 대기 후 재시도")
                time.sleep(COOLDOWN_SEC)
        done = min(start + BATCH_SIZE, total)
        print(f"  임베딩·저장: {done}/{total}")
        if done < total:                             # 마지막 배치가 아니면 다음 배치 전 쿨다운
            time.sleep(COOLDOWN_SEC)


def main():
    chunks = build_chunks()             # (1) PDF들을 읽어 청크로 분할
    print(f"청크 수: {len(chunks)}")     # 몇 조각으로 나뉘었는지 확인(0이면 PDF가 비었거나 스캔본=텍스트 없음 의심)
    # (2) 저장할 Chroma 컬렉션을 연다. persist_directory를 주면 디스크에 영구 저장되어 재사용 가능
    vs = Chroma(
        collection_name=COLLECTION,     # 저장할 컬렉션 이름 → 검색할 때 같은 이름으로 접근
        embedding_function=get_embeddings(),  # 각 청크를 벡터로 바꿀 임베딩기(위에서 만든 캐시 버전)
        persist_directory=PERSIST_DIR,  # 저장 위치(폴더). 지정하면 메모리가 아닌 디스크에 남음
    )
    # (2-1) 멱등성 보장: Chroma의 add는 기존 문서를 지우지 않고 '덧붙인다'.
    #       그래서 이 스크립트를 N번 실행하면 같은 청크가 N배로 중복 적재된다(검색 품질·평가 왜곡).
    #       저장 전에 컬렉션을 비워, 몇 번을 실행하든 결과가 '항상 최신 청크 1벌'이 되게 한다.
    vs.reset_collection()               # 같은 이름의 컬렉션을 드롭 후 빈 상태로 재생성(중복 방지)
    add_in_batches(vs, chunks)          # (3) TPM 한도를 지키며 배치로 나눠 임베딩·저장
    print(f"저장 완료 -> {PERSIST_DIR} (collection={COLLECTION})")


if __name__ == "__main__":   # 이 파일을 직접 실행할 때만 main() 호출(다른 파일에서 import하면 실행 안 됨)
    main()
