from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("./pdf-file.pdf")


docs = loader.load()


from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)

recursive_docs = text_splitter.split_documents(docs)

# text splitter 때문에 3개가 아닌 여러개의 Document 객체가 생성됨
# print(len(recursive_docs))

# print(recursive_docs[0].page_content)


from langchain_openai import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings


from dotenv import load_dotenv


load_dotenv()


underlying_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


store = LocalFileStore('./cache') 

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace="gemini-embedding-001"
)

from langchain_chroma import Chroma


CHROMA_PATH = "./chroma_db"

# 04-05.rag.py 와 다른 점은 데이터 불어오기 및 연결을 한다는 것
# 특징: documents= 인자가 없습니다. 
# 즉, 새로운 문서를 추가하는 게 아니라 기존에 저장된 데이터를 검색하기 위한 용도로 주로 사용합니다
db = Chroma(
    collection_name="rag_collection",
    embedding_function=cached_embedder,
    #  "텍스트를 숫자로 바꿀 때(임베딩), 이미 계산한 적이 있는 내용은 다시 계산하지 않고 저장해둔 값을 꺼내 쓰겠다"는 의미입니다.

    persist_directory=CHROMA_PATH,
    # 1. 정보가 저장된 위치 (읽기)
    # 이미 CHROMA_PATH 경로에 벡터 데이터(임베딩된 문서들)가 저장되어 있다면, 새로 데이터를 생성하지 않고 그 폴더에 있는 기존 데이터를 불러와서 사용하겠다는 뜻입니다.
    
    # 2. 정보가 저장될 위치 (쓰기)
    # 만약 해당 경로에 데이터가 없다면, 현재 처리 중인 문서들을 임베딩하여 그 폴더 안에 파일 형태로 영구히 저장(Persist)하겠다는 뜻입니다.

    collection_metadata={"hnsw:space": "cosine"}  # 옵션: "cosine", "l2" (유클리드), "ip" (내적)

    # hnsw: ChromaDB가 내부적으로 사용하는 고속 검색 알고리즘(Hierarchical Navigable Small World)의 약자입니다.
    # space: 검색 공간에서 거리를 측정하는 방식을 뜻합니다.
    # 즉, 이 설정은 "데이터를 찾을 때 코사인 유사도 알고리즘을 사용하여 가장 의미가 가까운 문장을 찾아라"라는 지시입니다.
)

query = "Tesla 투자 비중이 얼마나 되나요?"
results = db.similarity_search(query)
# # 벡터스토어에 있는 문서와 유사도를 계산하고, 유사도가 높은 순서대로 검색 결과를 반환한다

# print(len(results))

# print(f"검색된 문서 내용:\n{results[0].page_content}")
# 가장 유사한 문서의 내용을 반환한다
# 이 내용을 LLM에게 전달해서 답변을 생성할 수 있다
# 이것이 RAG 


# 검색기 생성
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.4}
    # score_threshold: 유사도 점수가 0.4 이상인 문서만 검색하겠다는 의미
    # 따라서 유사도 점수를 높이면 검색 결과가 없을 수도 있다.
)


results = retriever.invoke(query)

print(len(results))

print(results[0].page_content)

# 검색기 생성
retriever2 = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3,}

)


results2 = retriever2.invoke(query)

print(len(results2))

print(results2[0].page_content)