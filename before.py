import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from dataclasses import dataclass
from langchain.agents.middleware import wrap_model_call, before_agent


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


# https://docs.langchain.com/oss/python/langchain/middleware/custom#agent-jumps
from langchain.agents.middleware import after_model, hook_config, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any


@before_agent(can_jump_to=["end"])
def before_agent_hook(state: AgentState, runtime: Runtime):
    print("before_agent_hook", state)
    human_message = state["messages"][-1]
    print("human_message", human_message)
    if "BLOCKED" in human_message.content:
        return {
            "messages": [AIMessage("I cannot respond to that request.")],
            "jump_to": "end"
        }
    return state
    

agent = create_agent(
    model=model,
    tools=[],
    middleware=[before_agent_hook]
)

result = agent.invoke({"messages": [{"role": "user", "content": "Hello, how are you? BLOCKED"}]})
print(result)

# AI Model에게 민감 정보가 노출되는 것을 사전에 방지하기 위해서 사용됩니다.
# 모델 호출 없이 사용자의 요청을 검사하고, 민감 정보가 포함되어 있으면 즉시 종료합니다.






# @before_agent는 전체 작업의 '시작' 버튼과 같고, @before_model은 작업 도중 AI에게 질문을 던질 때마다 켜지는 '센서'와 같습니다.

# 1. @before_agent: 전체 프로세스의 입구

# 실행 횟수: 단 한 번만 실행됩니다.

# 동작 시점: 사용자가 질문을 던지고 에이전트가 가동되는 가장 첫 단계에서 동작합니다.

# 비유: 프론트엔드에서 페이지가 처음 렌더링될 때 딱 한 번 실행되는 useEffect(() => {}, [])와 유사합니다.

# 주요 용도: 초기 설정, 전체 실행 로그 시작, 또는 에이전트가 가동되기 전 필수적인 환경 변수 체크.


# 2. @before_model: AI의 '생각(Reasoning)' 직전

# 실행 횟수: Model(LLM)이 호출될 때마다 반복 실행됩니다.

# 동작 시점: 이미지의 '4칙 연산 Agent 실행 결과' 예시를 보면, AI가 곱셈이 필요하다고 판단할 때(2번), 덧셈이 필요하다고 판단할 때(4번), 그리고 내부 추론을 시도할 때(6번) 등 AI 모델 엔진이 돌아가기 직전마다 매번 호출됩니다.

# 비유: React 컴포넌트가 리렌더링될 때마다 실행되는 일반 로직이나, API 요청 직전에 매번 실행되는 Axios Request Interceptor와 같습니다.

# 주요 용도: AI에게 보내는 프롬프트 최종 검사, 매 루프(Step)마다 변하는 상태(State) 로깅, 혹은 토큰 사용량 실시간 모니터링.




# 1. @after_model: AI의 답변 직후 (반복 실행)

# 에이전트 내부의 루프에서 AI 모델(LLM)이 추론을 마치고 메시지를 내뱉은 직후에 호출됩니다.

# 실행 시점: 이미지의 단계 중 2번(곱셈 판단), 4번(덧셈 판단), 6번(내부 추론)이 각각 끝날 때마다 매번 호출됩니다.

# 주요 역할:

# 응답 가공: AI가 내뱉은 답변 형식이 잘못되었다면 이를 수정하거나 파싱합니다.

# 실시간 토큰 집계: 이번 루프에서 모델이 사용한 토큰 양을 기록합니다.

# 상태 업데이트: 모델의 답변을 바탕으로 State의 특정 필드를 즉시 갱신합니다.

# 비유: 프론트엔드에서 API 요청 후 응답을 받았을 때 거치는 Axios Response Interceptor와 같습니다.


# 2. @after_agent: 전체 프로세스 종료 (1회 실행)

# 에이전트가 최종 답변을 생성하고 사용자에게 돌려주기 직전에 단 한 번 호출됩니다.

# 실행 시점: 이미지의 7번 단계(최종 답변 출력)가 완료된 후, 전체 프로세스가 마무리될 때 실행됩니다.

# 최종 결과 검증: 사용자에게 나가기 전 답변의 말투나 금칙어 포함 여부를 최종 확인합니다.

# 전체 통계 저장: @before_agent에서 시작한 전체 실행 시간이나 총 누적 비용을 DB에 저장합니다.

# 리소스 정리: 이번 실행 중에 사용했던 임시 메모리나 컨텍스트를 정리합니다.

# 비유: 함수 실행이 완전히 끝나고 값을 반환하기 직전의 finally 구문이나, 컴포넌트가 언마운트되기 직전의 정리 로직과 비슷합니다.



