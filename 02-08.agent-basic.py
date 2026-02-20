
import os
from typing import Any, List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
import requests
from langchain.agents.structured_output import ToolStrategy

from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import LLMToolEmulator
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import PIIMiddleware 

# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델 선언
model = init_chat_model("google_genai:gemini-3-flash-preview")


# Define tools for testing
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 72°F"


@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    populations = {
        "paris": "2.2 million",
        "london": "8.9 million",
        "tokyo": "14 million",
        "seoul": "9.7 million"
    }
    return populations.get(city.lower(), "Population data not available")


# Define a structured response format using Pydantic
class CityInfoResponse(BaseModel):
    city_name: str
    weather: str
    population: str
    summary: str


# Create agent with structured output
agent = create_agent(
    model=model,
    tools=[get_weather, get_population],
    middleware=[
        LLMToolEmulator(model=model),  # LLM Tool Emulator - 실제 도구 호출 대신 LLM으로 시뮬레이션
    ],
    response_format=ToolStrategy(CityInfoResponse)  # Structured output format
)


# Test structured output
print("\n" + "=" * 80)
print("Structured Output 테스트")
print("=" * 80 + "\n")

query = "Tell me about Paris - what's the weather and population?"
print(f"질문: {query}\n")

response = agent.invoke({"messages": [{"role": "user", "content": query}]})
print(f"응답:\n{response}") 
print(response['messages'][-1].content)