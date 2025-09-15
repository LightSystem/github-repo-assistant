from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine
from langchain_postgres import PGVectorStore


def get_pg_engine():
    return PGEngine.from_connection_string(
        "postgresql+psycopg://postgres:postgres@localhost:5432/github_repo_assistant")


table_name = "openai_embedding_small"


def get_vector_store(pg_engine: PGEngine) -> PGVectorStore:
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return PGVectorStore.create_sync(engine=pg_engine, table_name=table_name, embedding_service=embedding_model)
