import asyncio

from langchain_community.document_loaders import GithubFileLoader
from langchain_text_splitters import MarkdownTextSplitter

from support.cli import parse_args
from support.vector_store import get_vector_store, get_pg_engine, table_name

args = parse_args()
pg_engine = get_pg_engine()
# odoo benchmark: 5=6m44s; 10=6m37s
concurrency = 10
sem = asyncio.Semaphore(concurrency)
text_splitter = MarkdownTextSplitter()


async def ingest():
    try:
        pg_engine.drop_table(table_name)
        pg_engine.init_vectorstore_table(table_name=table_name, vector_size=1536)
        vector_store = get_vector_store(pg_engine)
        documents = GithubFileLoader(repo=args.repo, branch=args.branch,
                                     file_filter=lambda file_path: file_path.endswith(".md")).alazy_load()

        async def process_doc(doc):
            async with sem:
                doc.metadata["source"] = doc.metadata["source"].replace("api.", "")
                chunks = text_splitter.split_documents([doc])
                await vector_store.aadd_documents(chunks)

        tasks = []
        async for d in documents: tasks.append(asyncio.create_task(process_doc(d)))
        await asyncio.gather(*tasks)
    finally:
        await pg_engine.close()


asyncio.run(ingest())
