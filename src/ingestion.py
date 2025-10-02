from collections.abc import AsyncIterator
import datetime
from anyio import CapacityLimiter, run, create_task_group
from langchain_community.document_loaders import GithubFileLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from textwrap import dedent
from support.llm import llm_mini
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
txt_text_splitter = RecursiveCharacterTextSplitter()


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
        txt_documents = load_files(".txt")

        async def process_doc(doc: Document, text_splitter: RecursiveCharacterTextSplitter):
            async with limiter:
                metadata = doc.metadata
                print(f"Processing {metadata} at {datetime.datetime.now()}")
                # Convert `api.github.com` URLs to `github.com` URLs
                metadata["source"] = metadata["source"].replace("api.", "")
                chunks = text_splitter.split_documents([doc])
                await vector_store.aadd_documents(chunks)
                summary_prompt = dedent(f"""\
                Given the following File Content for the file `{metadata["path"]}`, summarize it in a text of 100 words or less.
                Focus on it's intention and purpose.
                
                File Content:
                {doc.page_content}""")
                summary = await llm_mini.ainvoke(summary_prompt)
                await vector_store.aadd_documents(
                    [Document(page_content=summary.content, metadata=metadata | {"summary": True})])

        async with create_task_group() as tg:
            async def process_iterator(documents: AsyncIterator[Document],
                                       text_splitter: RecursiveCharacterTextSplitter):
                async for d in documents:
                    tg.start_soon(process_doc, d, text_splitter)

            tg.start_soon(process_iterator, md_documents, md_text_splitter)
            tg.start_soon(process_iterator, mdx_documents, md_text_splitter)
            tg.start_soon(process_iterator, py_documents, py_text_splitter)
            tg.start_soon(process_iterator, txt_documents, txt_text_splitter)
    finally:
        await pg_engine.close()


run(ingest)
