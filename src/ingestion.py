from collections.abc import AsyncIterator
import datetime
from anyio import CapacityLimiter, run, create_task_group
from langchain_community.document_loaders import GithubFileLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from support.cli import parse_ingestion_args
from support.vector_store import get_vector_store, get_pg_engine

args = parse_ingestion_args()
table_name = args.table
repo = args.repo
branch = args.branch
pg_engine = get_pg_engine()
concurrency = 5
limiter = CapacityLimiter(concurrency)
md_text_splitter = RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN)
py_text_splitter = RecursiveCharacterTextSplitter.from_language(Language.PYTHON)


async def ingest():
    try:
        pg_engine.drop_table(table_name)
        pg_engine.init_vectorstore_table(table_name=table_name, vector_size=1536)
        vector_store = get_vector_store(pg_engine, table_name)

        def load_files(extension):
            return GithubFileLoader(
                repo=repo, branch=branch, file_filter=lambda file_path: file_path.endswith(extension)
            ).alazy_load()

        md_documents = load_files(".md")
        mdx_documents = load_files(".mdx")
        py_documents = load_files(".py")

        async def process_doc(doc, text_splitter: RecursiveCharacterTextSplitter):
            async with limiter:
                print(f"Processing {doc.metadata} at {datetime.datetime.now()}")
                # Convert `api.github.com` URLs to `github.com` URLs
                doc.metadata["source"] = doc.metadata["source"].replace("api.", "")
                chunks = text_splitter.split_documents([doc])
                await vector_store.aadd_documents(chunks)

        async with create_task_group() as tg:
            async def process_iterator(documents: AsyncIterator[Document],
                                       text_splitter: RecursiveCharacterTextSplitter):
                async for d in documents:
                    tg.start_soon(process_doc, d, text_splitter)

            tg.start_soon(process_iterator, md_documents, md_text_splitter)
            tg.start_soon(process_iterator, mdx_documents, md_text_splitter)
            tg.start_soon(process_iterator, py_documents, py_text_splitter)
    finally:
        await pg_engine.close()


run(ingest)
