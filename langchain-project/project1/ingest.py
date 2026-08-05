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
from pathlib import Path
from dotenv import load_dotenv
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
    # key_encoder="sha256" : 캐시 파일명(=청크 텍스트의 지문)을 만드는 해시. 기본값 SHA-1은 충돌 내성이 없어 경고가 뜬다
    return CacheBackedEmbeddings.from_bytes_store(
        base, store, namespace=base.model, key_encoder="sha256"
    )

def main():
    chunks = build_chunks()             # (1) PDF들을 읽어 청크로 분할
    print(f"청크 수: {len(chunks)}")     # 몇 조각으로 나뉘었는지 확인(0이면 PDF가 비었거나 스캔본=텍스트 없음 의심)
    # (2) 재실행 대비: 기존 컬렉션을 통째로 비운다(처음 실행이면 빈 것을 지우고 지나간다).
    #     아래 from_documents는 '덮어쓰기'가 아니라 '추가'라서, 비우지 않고 다시 돌리면 같은 청크가 중복으로 쌓인다.
    #     전부 다시 저장해도 임베딩은 cache/에서 꺼내 쓰므로 API를 다시 부르지 않는다(= 사실상 공짜).
    Chroma(collection_name=COLLECTION, persist_directory=PERSIST_DIR).delete_collection()
    # (3) 각 청크를 임베딩해 Chroma에 저장. persist_directory를 주면 디스크에 영구 저장되어 재사용 가능
    Chroma.from_documents(
        documents=chunks,               # 저장할 청크(Document) 목록
        embedding=get_embeddings(),     # 각 청크를 벡터로 바꿀 임베딩기(위에서 만든 캐시 버전)
        collection_name=COLLECTION,     # 저장할 컬렉션 이름 → 검색할 때 같은 이름으로 접근
        persist_directory=PERSIST_DIR,  # 저장 위치(폴더). 지정하면 메모리가 아닌 디스크에 남음
    )
    print(f"저장 완료 -> {PERSIST_DIR} (collection={COLLECTION})")

if __name__ == "__main__":   # 이 파일을 직접 실행할 때만 main() 호출(다른 파일에서 import하면 실행 안 됨)
    main()
