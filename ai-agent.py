
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent

# langchain 메시지 객체 임포트
from langchain.messages import HumanMessage, SystemMessage

# .env 파일에서 환경 변수 로드
load_dotenv()

# API Key 가져오기
api_key = os.getenv("GEMINI_API_KEY")
aladin_api_key = os.getenv("ALADIN_API_KEY")

if api_key:
    # 보안을 위해 키의 일부만 출력
    masked_key = f"{api_key[:8]}...{api_key[-4:]}"
    print(f"API Key가 성공적으로 로드되었습니다: {masked_key}")
else:
    print("API Key를 찾을 수 없습니다. .env 파일을 확인해주세요.")

model = init_chat_model("gemini-2.0-flash",
  model_provider="google_genai",
  api_key=os.getenv("GEMINI_API_KEY"))



import requests
import os
from typing import List, Dict, Any

@tool
def get_aladin_bestsellers(count: int = 10) -> List[Dict[str, Any]]:
    """
    알라딘 API를 사용하여 현재 가장 인기 있는 베스트셀러 도서 목록을 가져옵니다.
    
    Args:
        count (int): 가져올 도서의 수 (기본값 10, 최대 50).
        
    Returns:
        List[Dict]: 도서 제목, 저자, 가격, 링크 등을 포함한 딕셔너리 리스트.
    """
    # 2025-04-07 지침: 보안을 위해 TTBKey는 환경변수 사용 권장
  
    url = "http://www.aladin.co.kr/ttb/api/ItemList.aspx"
    
    # 알라딘 API 명세서 근거 파라미터 구성
    params = {
        "ttbkey": aladin_api_key,
        "QueryType": "Bestseller",  # 리스트 종류: 베스트셀러
        "MaxResults": min(count, 50), # 한 페이지 최대 50개 제한
        "start": 1,
        "SearchTarget": "Book",
        "output": "js",            # JSON 방식 출력
        "Version": "20131101"      # 최신 버전
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status() 
        #  요청 결과가 4xx 또는 5xx 에러(실패)일 때 예외(requests.exceptions.HTTPError)를 발생시킵니다.
        # 즉, 요청이 성공(200 OK 등)이 아니면 코드 실행을 중단하고 에러 처리를 하도록 만듭니다.
        # 이를 통해 API 호출 실패를 쉽게 감지하고, try-except로 예외를 처리할 수 있습니다.
        data = response.json()
        
        # 문서 응답 구조: 'item' 키 안에 도서 리스트가 담겨 있음
        items = data.get("item", [])
        
        # 에이전트가 이해하기 쉽도록 핵심 필드만 필터링 (Token 절약 및 가독성)
        return [
            {
                "title": item.get("title"),
                "author": item.get("author"),
                "publisher": item.get("publisher"),
                "price": item.get("priceSales"),
                "link": item.get("link")
            }
            for item in items
        ]
    except Exception as e:
        return [{"error": f"API 호출 실패: {str(e)}"}]



agent = create_agent(
    model=model,
    tools=[get_aladin_bestsellers],)


result = agent.invoke({"messages": [
  {"role": "user", "content": "현재 알라딘에서 가장 인기 있는 베스트셀러 책 5권을 알려줘."},
]})

print("Agent의 응답:", result) 



# ttbsayend04090057001