from anyio import CapacityLimiter, run, create_task_group
from langchain_community.document_loaders import GithubFileLoader
from langchain_text_splitters import MarkdownTextSplitter

from support.cli import parse_ingestion_args
from support.vector_store import get_vector_store, get_pg_engine

args = parse_ingestion_args()
table_name = args.table
pg_engine = get_pg_engine()
concurrency = 5
limiter = CapacityLimiter(concurrency)
text_splitter = MarkdownTextSplitter()


async def ingest():
    try:
        pg_engine.drop_table(table_name)
        pg_engine.init_vectorstore_table(table_name=table_name, vector_size=1536)
        vector_store = get_vector_store(pg_engine, table_name)
        documents = GithubFileLoader(repo=args.repo, branch=args.branch,
                                     file_filter=lambda file_path: file_path.endswith(".md")).alazy_load()

        async def process_doc(doc):
            async with limiter:
                # Convert `api.github.com` URLs to `github.com` URLs
                doc.metadata["source"] = doc.metadata["source"].replace("api.", "")
                chunks = text_splitter.split_documents([doc])
                await vector_store.aadd_documents(chunks)

        async with create_task_group() as tg:
            async for d in documents:
                tg.start_soon(process_doc, d)
    finally:
        await pg_engine.close()


run(ingest)
