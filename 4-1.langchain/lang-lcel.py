from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser

# LCEL 예시
chain = ChatPromptTemplate() | ChatOpenAI() | CustomOutputParser()

이와 더불어, 목적에 맞는 다양한 프롬프트 템플릿, 구조화된 출력을 제공한다.
from langchain.output_parsers.json import SimpleJsonOutputParser

json_prompt = PromptTemplate.from_template(
    "Return a JSON object with `birthdate` and `birthplace` key that answers the following question: {question}"
)
json_parser = SimpleJsonOutputParser() # JSON 파서

# 프롬프트, 모델, 파서 체인 생성
json_chain = json_prompt | model | json_parser  # 유닉스 파이프라인 개념 차용함.

result_list = list(json_chain.stream({"question": "When and where was Elon Musk born?"}))
print(result_list)

from langchain_core.runnables import RunnableLambda

def add_five(x):
    return x + 5

def multiply_by_two(x):
    return x * 2

# wrap the functions with RunnableLambda
add_five = RunnableLambda(add_five)
multiply_by_two = RunnableLambda(multiply_by_two)

chain = add_five | multiply_by_two
chain.invoke(3)