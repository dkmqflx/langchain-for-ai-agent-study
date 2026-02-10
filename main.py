
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


# 영화 정보를 위한 JSON Schema 선언
movie_json_schema = {
  "title": "Movie",
  "type": "object",
  "properties": {
    "title": {"type": "string", "description": "영화 제목"},
    "director": {"type": "string", "description": "감독"},
    "year": {"type": "integer", "description": "개봉 연도"},
    "genre": {"type": "string", "description": "장르"}
  },
  "required": ["title", "director", "year", "genre"]
}

# .env 파일에서 환경 변수 로드
load_dotenv()

# API Key 가져오기
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    # 보안을 위해 키의 일부만 출력
    masked_key = f"{api_key[:8]}...{api_key[-4:]}"
    print(f"API Key가 성공적으로 로드되었습니다: {masked_key}")
else:
    print("API Key를 찾을 수 없습니다. .env 파일을 확인해주세요.")

model = init_chat_model("gemini-2.0-flash",
  model_provider="google_genai",
  api_key=os.getenv("GEMINI_API_KEY"))


# JSON Schema 기반 구조적 출력으로 모델 래핑
model = model.with_structured_output(movie_json_schema)

response = model.invoke("Explain about the movie Truman Show")
print(response) # {'title': 'The Truman Show', 'director': 'Peter Weir', 'year': 1998, 'genre': 'Comedy-drama'}
