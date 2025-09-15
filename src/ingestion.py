import asyncio
import random

from langchain_community.document_loaders import GithubFileLoader
from langchain_text_splitters import MarkdownTextSplitter

from support.vector_store import get_vector_store, get_pg_engine, table_name


def make_file_filter():
    document_limit = 10
    count = 0

    def file_filter(file_path):
        nonlocal count
        is_md_file = file_path.endswith(".md")
        if is_md_file:
            count += 1
        return file_path.endswith(".md") and count <= document_limit

    return file_filter


documents = GithubFileLoader(repo="odoo/odoo", branch="18.0", file_filter=make_file_filter()).load()
for doc in documents:
    # Convert `api.github.com` URLs to `github.com` URLs
    doc.metadata["source"] = doc.metadata["source"].replace("api.", "")
nr_of_docs = len(documents)
print("Number of Documents: ", nr_of_docs)
print("Example Document: ", documents[random.randint(0, nr_of_docs - 1)])
chunks = MarkdownTextSplitter().split_documents(documents)
print("Example Chunk: ", chunks[random.randint(0, len(chunks) - 1)])
pg_engine = get_pg_engine()
try:
    pg_engine.drop_table(table_name)
    pg_engine.init_vectorstore_table(table_name=table_name, vector_size=1536)
    get_vector_store(pg_engine).add_documents(chunks)
finally:
    asyncio.run(pg_engine.close())
