# 테스트 텍스트
sample_text = """인공지능(AI) 기술은 지난 몇 년간 급격하게 발전했습니다. 특히 OpenAI의 GPT 시리즈와 같은 대규모 언어 모델(LLM)은 자연어 처리 분야에서 혁신을 일으켰습니다.

하지만 LLM에게는 명확한 한계가 존재합니다.
첫째, 학습 데이터가 특정 시점에 고정되어 있어 최신 정보를 알지 못합니다.
둘째, 모델이 한 번에 처리할 수 있는 입력 토큰의 수(Context Window)에 제한이 있습니다.
이러한 문제를 해결하기 위해 등장한 것이 바로 RAG(검색 증강 생성) 기술입니다.

RAG 시스템의 핵심 프로세스는 다음과 같습니다:
1. 데이터 로드 (Loading)
2. 텍스트 분할 (Splitting)
3. 임베딩 및 저장 (Embedding & Storage)
4. 검색 및 답변 생성 (Retrieval & Generation)

텍스트 분할은 매우 중요합니다. RecursiveCharacterTextSplitter는문단이너무길경우줄바꿈으로자르고줄바꿈도없으면공백으로자르고공백도없으면결국글자단위로자르게됩니다.이문장은공백이거의없어서글자단위분할테스트에적합합니다."""


from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
# document의 특징에 따라 최적의 chunk_size와 chunk_overlap이 다르다
# 다만 너무 chunk 사이즈가 큰 것보다는 작은게 낫다 
# 의미가 보존되면서 chunk가 유지되기 때문
# chunk가 너무 크면, llm 모델이 가져왔을 때 할루시네이션 문제가 발생할 수 있다 
# chunk_size는 보통 300~500 정도로 설정한다
# chunk_overlap은 보통 100 정도로 설정한다


recursive_texts = text_splitter.split_text(sample_text)


# 결과 출력
print(f"총 청크 개수: {len(recursive_texts)}\n")
for i, doc in enumerate(recursive_texts):
    print(f"--- Chunk {i+1} ({len(doc)}자) ---")
    print(doc)
    print()


# Overlap이 발생하는 조건
# overlap은 보통 하나의 연속된 긴 텍스트를 강제로 자를 때 그 절단면 전후의 문맥을 유지하기 위해 발생합니다.
# 하지만 현재 제공된 텍스트는 줄바꿈(\n)으로 이미 잘게 쪼개져 있는 상태라, 
# 스플리터가 "자연스러운 경계(줄바꿈)"를 선택하면서 overlap이 끼어들 틈이 없는 것입니다.


from langchain_text_splitters import CharacterTextSplitter

# 그냥 글자수대로 자르는 스플리터
char_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)

char_docs = char_splitter.split_text(sample_text)

# 결과 출력
print(f"총 청크 개수: {len(char_docs)}\n")
for i, doc in enumerate(char_docs):
    print(f"--- Chunk {i+1} ({len(doc)}자) ---")
    print(doc)
    print()