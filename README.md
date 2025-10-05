# GitHub Repo Assistant

A RAG-powered chatbot that ingests GitHub repositories and lets you ask questions about the code and docs. For every
answer, it cites the original source file(s) the information came from.

The workflow is simple:

- Ingest a GitHub repo into a vector database (PostgreSQL + pgvector).
- Run an interactive chatbot (Gradio UI) that retrieves semantically relevant chunks and answers based strictly on those
  sources.

## Table of Contents
- [Overview](#github-repo-assistant)
- [Prerequisites](#prerequisites)
- [Running the scripts](#running-the-scripts)
  - [Ingest a GitHub repository](#1-ingest-a-github-repository)
  - [Run the RAG-enabled chatbot](#2-run-the-rag-enabled-chatbot)
- [How it works (high-level)](#how-it-works-high-level)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Environment variables
    - `OPENAI_API_KEY`: Your OpenAI API key (used for embeddings and chat completion).
    - `GITHUB_PERSONAL_ACCESS_TOKEN`: Token used by the GitHub loader to fetch repository files. See how to generate one
      here: https://python.langchain.com/docs/integrations/document_loaders/github/#setup-access-token

- Python
    - Recommended: Python 3.12 or newer.
    - Create and activate a virtual environment, then install dependencies from `requirements.txt`:
        - macOS/Linux
            - `python3 -m venv .venv`
            - `source .venv/bin/activate`
            - `pip install -r requirements.txt`
        - Windows (PowerShell)
            - `py -3 -m venv .venv`
            - `.venv\Scripts\Activate.ps1`
            - `pip install -r requirements.txt`

- PostgreSQL with pgvector
    - Easiest way is via Docker. Start a local PostgreSQL with `pgvector`:
        - `docker run --name github-repo-assistant -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg17-trixie`
    - For the next steps you'll need PostgreSQL client tools installed.
    - Create a database:
        - `createdb -h localhost -U postgres github_repo_assistant`
    - Connect and enable the `vector` extension:
        - `psql -h localhost -U postgres -d github_repo_assistant`
        - Inside psql: `create extension vector;`
    - Connection URL
        - You can override the database connection for the vector store by setting `DATABASE_URL` (e.g.,
          `postgresql+psycopg://postgres:postgres@localhost:5432/github_repo_assistant`).
        - If you followed the example commands above, you can omit `DATABASE_URL` because a matching default is used by
          the code.

## Running the scripts

Both entry points live in `src/`. You can run them directly as files `python src/<file>.py`.

Important: ensure the required environment variables are set in your shell before running the commands.

### 1) Ingest a GitHub repository

Script: `src/ingestion.py`

This script downloads files from a GitHub repository, chunks them, embeds the chunks with OpenAI embeddings, and writes
them to a PostgreSQL + pgvector table. It will recreate the specified table each time it runs.

Required arguments:

- `--table`: Name of the pgvector table to store embeddings (e.g., `my_repo_index`).
- `--repo`: GitHub repository in the form `owner/repo` (e.g., `openai/openai-cookbook`).
- `--branch`: Git branch to read from (e.g., `main`).

Example:

```
export OPENAI_API_KEY=...            # required
export GITHUB_PERSONAL_ACCESS_TOKEN=...  # required by the GitHub loader
# export DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/github_repo_assistant  # optional

python src/ingestion.py --table my_repo_index --repo owner/repo --branch main
```

Notes:

- The loader fetches specific file types: `.md`, `.mdx`, `.py`, `.txt`.
- Each document is split (language-aware where appropriate) before storage.
- A brief file summary is also generated and stored as an additional document with metadata `{"summary": True}`.

### 2) Run the RAG-enabled chatbot

Script: `src/inference.py`

This script starts a Gradio ChatInterface that answers questions based only on the ingested repository context, citing
the sources from which the answer was derived.

Required arguments:

- `--table`: Name of the pgvector table previously populated by the ingestion step.

Example:

```
export OPENAI_API_KEY=...
# export DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/github_repo_assistant  # optional

python src/inference.py --table my_repo_index
```

Behavior:

- The app opens a local Gradio UI (look for the printed local URL in the console, typically http://127.0.0.1:7860/).
- Answers are generated from retrieved, high-relevance chunks. If no relevant context is found, it will respond: "That
  is out of my scope".

## How it works (high-level)

- Embeddings: Uses OpenAI `text-embedding-3-small` to embed chunks.
- Vector store: Uses `langchain-postgres` `PGVectorStore` with a `PGEngine` built from `DATABASE_URL` (or default to
  `postgresql+psycopg://postgres:postgres@localhost:5432/github_repo_assistant`).
- Retrieval flow (inference):
    - Decomposes the user query into sub-questions and multi-queries to improve recall.
    - Performs similarity search for each reformulated query and filters by score.
    - Builds a final prompt comprised of the user query, retrieved context, and metadata, then sends it to the chat
      model.

## Troubleshooting

- Connection refused to PostgreSQL:
    - Ensure Docker container is running and port 5432 is available on localhost.
    - Ensure `create extension vector;` was executed in the target database.
- No results found / poor answers:
    - Verify you ingested the correct repo/branch and table name.
    - Re-run ingestion to avoid stale data.
- Authentication issues:
    - Verify `OPENAI_API_KEY` and `GITHUB_PERSONAL_ACCESS_TOKEN` are set in your environment.
