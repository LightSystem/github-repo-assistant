import asyncio
from textwrap import dedent

import gradio as gr

from support.cli import parse_inference_args
from support.llm import llm_mini_json, llm
from support.vector_store import get_pg_engine, get_vector_store

args = parse_inference_args()
pg_engine = get_pg_engine()
try:
    vector_store = get_vector_store(pg_engine, args.table)
    system_prompt = dedent("""\
    Instructions:
    Answer the following User Query based solely on the provided Context.
    If the answer is not in the Context, check if it is in the Chat History, and answer based on that.
    If you still can't find an answer, say "That is out of my scope".
    Use the provided Metadata to give sources of the Context used on the answer, if a source is available.""").strip()


    def inference_function(user_query, history):
        sub_query_prompt = dedent(f"""\
        Given the following User Query, check if there are multiple questions being asked.
        If there are return each question separately in a JSON array.
        Otherwise return the JSON array with a single element which is the User Query verbatim.
        The JSON object should have a key named "questions" and a value which is a JSON array of questions.
        User Query:
        {user_query}""").strip()
        sub_queries = llm_mini_json.invoke(sub_query_prompt)["questions"]
        print(f"sub_queries: {sub_queries}")
        multi_query_prompt = dedent(f"""\
        Given the following list of questions, transform each into a set of 2 questions.
        Each question should on a different part of the original question, representing different points of view.
        Return the transformed questions in a JSON array.
        The JSON object should have a key named "questions" and a value which is a JSON array of questions.
        Questions:
        {sub_queries}""").strip()
        multi_queries = llm_mini_json.invoke(multi_query_prompt)["questions"]
        print(f"multi_queries: {multi_queries}")
        retrieved_docs = [doc for multi_query in multi_queries for doc in
                          vector_store.similarity_search_with_score(multi_query, k=1)]
        print(f"retrieved_docs score: {[score for _, score in retrieved_docs]}")
        relevant_retrieved_docs = []
        seen_ids = set()
        for document, score in retrieved_docs:
            document_id = document.id
            if score < 0.7 and document_id not in seen_ids:
                relevant_retrieved_docs.append(document)
                seen_ids.add(document_id)
        print(f"relevant_retrieved_docs: {relevant_retrieved_docs}")
        relevant_context = "\n\n".join([relevant_doc.page_content for relevant_doc in relevant_retrieved_docs])
        relevant_metadata = [relevant_doc.metadata for relevant_doc in relevant_retrieved_docs]
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
        return llm.invoke(llm_messages).content


    gr.ChatInterface(inference_function, type="messages").launch()
finally:
    asyncio.run(pg_engine.close())
