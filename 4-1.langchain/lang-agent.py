from langchain import OpenAI

llm = OpenAI(
    openai_api_key="OPENAI_API_KEY",
    temperature=0,
    model_name="text-davinci-003"
)

from langchain.chains import LLMMathChain
from langchain.agents import Tool

llm_math = LLMMathChain(llm=llm)

# initialize the math tool
math_tool = Tool(
    name='Calculator',
    func=llm_math.run,
    description='Useful for when you need to answer questions about math.'
)
# when giving tools to LLM, we must pass as list of tools
tools = [math_tool]

# ReAct 프레임웍을 사용해 도구 설명 만을 기반으로 사용할 도구를 결정하게 된다. 
from langchain.agents import initialize_agent

zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3
)

# 이제 다음과 같이 제대로 에이전트가 작업을 수행하는 지 테스트해보자. 
zero_shot_agent("what is (4.5*2.1)^2.2?")
zero_shot_agent("if Mary has four apples and Giorgio brings two and a half apple boxes (apple box contains eight apples), how many apples do we have?")

# 제대로 실행되겠지만, 다음과 같은 질문을 하면 실패할 것이다. 
zero_shot_agent("what is the capital of Norway?")

# 우리는 단 하나의 계산기 도구만 있으므로, 랭체인에 이미 만들어 놓은, 언어를 해석해 논리적으로 추론하는 도구를 추가해 넣는다. 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# initialize the LLM tool
llm_tool = Tool(
    name='Language Model',
    func=llm_chain.run,
    description='use this tool for general purpose queries and logic'
)

tools.append(llm_tool)

# reinitialize the agent
zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3
)

# 이런 방식으로 에이전트를 확장해 나갈 수 있다.

# 사용자가 도구를 직접 만들 수도 있다. 다음은 원 둘래를 계산하는 도구 정의를 보여준다. 
from langchain.tools import BaseTool
from math import pi
from typing import Union

class CircumferenceTool(BaseTool):
      name = "Circumference calculator"
      description = "use this tool when you need to calculate a circumference using the radius of a circle"

    def _run(self, radius: Union[int, float]):
        return float(radius)*2.0*pi

    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")

# 다음과 같이 삼각형의 빗변을 계산하기 위해 여러 파라메터를 입력받은 도구도 만들 수 있다. 
from typing import Optional
from math import sqrt, cos, sin

desc = (
    "use this tool when you need to calculate the length of a hypotenuse"
    "given one or two sides of a triangle and/or an angle (in degrees). "
    "To use the tool, you must provide at least two of the following parameters "
    "['adjacent_side', 'opposite_side', 'angle']."
)

class PythagorasTool(BaseTool):
    name = "Hypotenuse calculator"
    description = desc
    
    def _run(
        self,
        adjacent_side: Optional[Union[int, float]] = None,
        opposite_side: Optional[Union[int, float]] = None,
        angle: Optional[Union[int, float]] = None
    ):
        # check for the values we have been given
        if adjacent_side and opposite_side:
            return sqrt(float(adjacent_side)**2 + float(opposite_side)**2)
        elif adjacent_side and angle:
            return adjacent_side / cos(float(angle))
        elif opposite_side and angle:
            return opposite_side / sin(float(angle))
        else:
            return "Could not calculate the hypotenuse of the triangle. Need two or more of `adjacent_side`, `opposite_side`, or `angle`."
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

tools = [PythagorasTool()]
