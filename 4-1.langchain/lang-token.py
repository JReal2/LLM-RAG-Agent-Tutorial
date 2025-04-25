from langchain.callbacks import get_openai_callback
def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')
    return result

count_tokens(
    conversation_buf, 
    "My interest here is to explore the potential of integrating Large Language Models with external knowledge"
)

conversation = ConversationChain(
    llm=llm,
    memory=ConversationSummaryMemory(llm=llm)
)