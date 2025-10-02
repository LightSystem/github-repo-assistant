from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1", temperature=0.5)
llm_mini = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)
llm_mini_json = llm_mini.with_structured_output(method="json_mode")
