from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


model = init_chat_model("google_genai:gemini-2.5-flash-lite")

response = model.invoke("ARKK 펀드의 Tesla 투자 비중이 어떻게 돼?")

print(response)
# ARKK 펀드의 Tesla 투자 비중은 계속 변동하기 때문에 정확한 수치를 실시간으로 제공하기는 어렵습니다 ...

from langchain.tools import tool
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.storage import LocalFileStore

CHROMA_PATH = "./chroma_db"


underlying_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
store = LocalFileStore('./cache') 



cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace="gemini-embedding-001"
)

db = Chroma(
    collection_name="rag_collection",
    embedding_function=cached_embedder,

    persist_directory=CHROMA_PATH,


    collection_metadata={"hnsw:space": "cosine"}  # 옵션: "cosine", "l2" (유클리드), "ip" (내적)
)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.4}
    # score_threshold: 유사도 점수가 0.4 이상인 문서만 검색하겠다는 의미
    # 따라서 유사도 점수를 높이면 검색 결과가 없을 수도 있다.

)
# 검색을 위한 tool 생성
@tool
def search_portfolio(query: str):
    """
    ARKK ETF의 포트폴리오 정보를 검색할 때 사용합니다.
    특정 기업의 보유 비중, 주식 수, 가치 등을 찾을 때 이 도구를 호출하세요.
    """
    documents = retriever.invoke(query)
    return "\n".join([document.page_content for document in documents])

    # 검색기 생성
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.4}
    # score_threshold: 유사도 점수가 0.4 이상인 문서만 검색하겠다는 의미
    # 따라서 유사도 점수를 높이면 검색 결과가 없을 수도 있다.
)

tools = [search_portfolio]


from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=tools,
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "ARKK 펀드의 Tesla 투자 비중이 어떻게 돼?"}]},
)

print(response['messages'][-1].content)