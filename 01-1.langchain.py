
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 영화 정보를 위한 Pydantic 모델 선언
from pydantic import BaseModel, Field

class Movie(BaseModel):
  """영화 정보"""
  title: str = Field(..., description="영화 제목")
  director: str = Field(..., description="감독")
  year: int = Field(..., description="개봉 연도")
  genre: str = Field(..., description="장르")

# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-2.5-flash-lite")


# Stream
for chunk in model.stream('Explain about the movie The Truman Show, Reply it briefly'):
  print(chunk)
  print(chunk.text, end='')



# Batch
inputs = [
  'Explain about the movie The Truman Show',
  'Explain about the movie The Truman Show, Reply it briefly'
]

for chunk in model.batch(inputs):
  for response in chunk:
    print(response)


# Structured Output# Movie 구조로 출력하도록 모델 래핑
mode_with_structured_output = model.with_structured_output(Movie)
response = mode_with_structured_output.invoke("Explain about the movie Truman Show")
print(response)
# title='The Truman Show' director='Peter Weir' year=1998 genre='Psychological comedy-drama'



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

model_with_json_schema = model.with_structured_output(movie_json_schema)
response = model_with_json_schema.invoke("Explain about the movie Truman Show")
print(response)
# title='The Truman Show' director='Peter Weir' year=1998 genre='Psychological comedy-drama'


