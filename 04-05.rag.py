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
# cache 폴더를 보면 파일이 생성된 것 확인할 수 있다
# 해당 파일은 임베딩 결과를 저장하는 파일로, recursive_docs의 page_content를 임베딩 한 결과를 저장하는 파일이다
# 파일 이름은 임베딩 모델의 이름과 page_content의 해시값으로 구성된다
# 띠리사 page_content와 똑같은 입력이 들어오면, 해싱을 한 다음 
# 동일한 파일이 있는지 찾고, 만약 같은게 있다면 임베딩 모델에 입력이 전달되는 것이 아니라
# 찾은 폴더의 파일을 읽어서 임베딩 결과를 반환하는 것이다

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace="gemini-embedding-001"
)

from langchain_chroma import Chroma

# 1. DB 경로 설정
CHROMA_PATH = "./chroma_db"


# 2. Store 생성
db = Chroma.from_documents(
    documents=recursive_docs,
    embedding=cached_embedder,
    persist_directory=CHROMA_PATH,
    collection_name="rag_collection"
)

query = "Tesla 투자 비중이 얼마나 되나요?"
results = db.similarity_search(query, k=1)
# # 벡터스토어에 있는 문서와 유사도를 계산하고, 유사도가 높은 순서대로 검색 결과를 반환한다

print(len(results))

print(f"검색된 문서 내용:\n{results[0].page_content}")
# 가장 유사한 문서의 내용을 반환한다
# 이 내용을 LLM에게 전달해서 답변을 생성할 수 있다
# 이것이 RAG 