from langchain_community.retrievers import BM25Retriever

# Hybrid Search

# BM25 알고리즘이란?
# BM25(Best Matching 25)는 전통적인 키워드 기반 검색 알고리즘입니다.
# 단어의 빈도(TF)와 문서 빈도의 역수(IDF)를 조합하여 점수를 계산하며, ElasticSearch나 Lucene 등에서 기본 검색 알고리즘으로 널리 사용됩니다.
# 벡터 유사도 검색(Semantic Search)과 달리 정확한 키워드 매칭에 강점이 있습니다.


# LangChain의 BM25Retriever를 사용하여 텍스트 데이터에서 관련성 높은 문서를 검색하는 간단한 예제

# 텍스트 리스트 준비
retriever = BM25Retriever.from_texts(["foo", "bar", "world", "hello", "foo bar"])
# BM25Retriever.from_texts(): 단순 문자열 리스트를 검색 가능한 데이터셋으로 변환합니다.

# 'foo'와 가장 연관성 높은 문서 검색
result = retriever.invoke("foo")
# retriever.invoke("foo"): "foo"라는 쿼리와 가장 유사한(BM25 알고리즘 기준) 문서를 찾아 반환합니다.

print(result)


from langchain_core.documents import Document

# LangChain의 표준 데이터 단위인 Document 객체를 사용하여 검색기를 만듭니다.
retriever = BM25Retriever.from_documents([
    Document(page_content="사과는"),
    Document(page_content="사과합니다."),
    Document(page_content="사과 팝니다."),
])

result2 = retriever.invoke("사과드립니다.")

print(result2)


from kiwipiepy import Kiwi

# 1. Kiwi 형태소 분석기 준비
kiwi = Kiwi()

# 전처리 함수 정의 (명사, 동사, 형용사 등 의미 있는 품사만 추출하거나 전체를 추출)
def korean_tokenizer(text):
    # kiwi.tokenize()는 토큰 정보를 반환하므로, 거기서 단어 형태(form)만 추출
    return [token.form for token in kiwi.tokenize(text)]


print(korean_tokenizer("사과드립니다."))


# 2. 문서 데이터
docs = [
    Document(page_content="RAG 시스템을 구축할 때는 검색기의 성능이 매우 중요합니다."),
    Document(page_content="한국어는 교착어이므로 형태소 분석이 필수적입니다."),
    Document(page_content="BM25는 키워드 기반의 희소 검색(Sparse Retrieval) 알고리즘입니다."),
    Document(page_content="벡터 검색은 문맥을 잘 파악하지만 고유명사 검색에 약할 수 있습니다."),
]

# 3. BM25 검색기 생성 (preprocess_func에 한국어 토크나이저 주입)
retriever2 = BM25Retriever.from_documents(
    docs,
    preprocess_func=korean_tokenizer
)

result3 = retriever2.invoke("형태소 분석은 왜 해야하나요?")
print(f"검색 결과: {result3[0].page_content}")