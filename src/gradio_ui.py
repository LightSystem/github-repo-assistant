import asyncio
from textwrap import dedent

import gradio as gr
from langchain_openai import ChatOpenAI

from support.vector_store import get_pg_engine, get_vector_store

pg_engine = get_pg_engine()
try:
    vector_store = get_vector_store(pg_engine)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
    llm_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    llm_mini_json = llm_mini.with_structured_output(method="json_mode")
    system_prompt = dedent("""\
    Instructions:
    Answer the following user query based on the provided context.
    If the answer is not in the context, check if it is in the Chat History, and answer based on that.
    If you still can't find an answer, say "That is out of my scope".
    Use the provided metadata to give sources of the context you used on your answer, if a source is available.""").strip()


    def inference_function(user_query, history):
        sub_query_prompt = dedent("""\
        
        """).strip()
        relevant_documents = vector_store.similarity_search_with_score(user_query, k=2)
        print(f"relevant_documents score: {[score for _, score in relevant_documents]}")
        relevant_documents = [document for document, score in relevant_documents if score < 0.9]
        print(f"relevant_documents after score filtering: {relevant_documents}")
        relevant_context = "\n\n".join([relevant_document.page_content for relevant_document in relevant_documents])
        relevant_metadata = [relevant_document.metadata for relevant_document in relevant_documents]
        final_prompt = dedent(f"""\
        User Query:
        {user_query}
        
        Context:
        {relevant_context}

        Metadata:
        {relevant_metadata}""").strip()
        llm_messages = [("system", system_prompt)]
        for history_message in history:
            llm_messages.append((history_message["role"], history_message["content"]))
        llm_messages.append(("user", final_prompt))
        print(f"messages: {llm_messages}")
        ai_msg = llm.invoke(llm_messages)
        return ai_msg.content


    gr.ChatInterface(inference_function, type="messages").launch()
finally:
    asyncio.run(pg_engine.close())
