from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import os
import gradio as gr

os.environ["OPENAI_API_KEY"] = "<input your openai key"  # API 키 설정

llm = ChatOpenAI(temperature=1.0, model='gpt-4o-mini')  

# LLM 응답 처리
def response(message, history, additional_input_info):
    history_langchain_format = []
    history_langchain_format.append(SystemMessage(content= additional_input_info))
    for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm(history_langchain_format)
    return gpt_response.content

# 인터페이스 생성
gr.ChatInterface(
    fn=response,   # LLM 응답처리 콜백함수 설정
    textbox=gr.Textbox(placeholder="Talk", container=False, scale=7),
    chatbot=gr.Chatbot(height=1000),
    title="ChatBot",
    description="I'm a chatbot that can chat with you. I'm lovely chatbot.",
    theme="soft",
    examples=[["Hi"], ["I'm good"], ["What's your name?"]],
    retry_btn="resend",
    undo_btn="delete❌",
    clear_btn="delete all💫",
    additional_inputs=[
        gr.Textbox("", label="Input System Prompt", placeholder="I'm chatbot.")
    ]
).launch()