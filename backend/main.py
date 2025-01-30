from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse

import logging

app = FastAPI()

from .config import (
    CohereAsyncClients,
    CohereClients,
    WeaviateAsyncClient,
    WeaviateClient,
)
from .models import Question, Answer
from .upload import aupload_documents, upload_documents
from .query import aquery_rag, astream_rag, query_rag, stream_rag


@app.get("/")
async def read_root():
    logging.debug("GET request received at root...")
    return {"Hello": "World"}


@app.post("/astream")
async def astream(
    question: Question,
    cohere_async_clients: CohereAsyncClients,
    weaviate_async_client: WeaviateAsyncClient,
) -> StreamingResponse:
    logging.debug(f"POST request received at /astream...")

    # Stream the respones
    return StreamingResponse(
        astream_rag(
            question.question,
            question.rerank,
            cohere_async_clients,
            weaviate_async_client,
        ),
        media_type="text/event-stream",
    )


@app.post("/aquery")
async def aquery(
    question: Question,
    cohere_async_clients: CohereAsyncClients,
    weaviate_async_client: WeaviateAsyncClient,
) -> Answer:
    logging.debug(f"POST request received at /aquery...")

    # Return the full response
    response = await aquery_rag(
        question.question, question.rerank, cohere_async_clients, weaviate_async_client
    )
    return response


@app.post("/stream")
def stream(
    question: Question, cohere_clients: CohereClients, weaviate_client: WeaviateClient
) -> StreamingResponse:
    logging.debug(f"POST request received at /stream...")

    # Stream the respones
    return StreamingResponse(
        stream_rag(question.question, question.rerank, cohere_clients, weaviate_client),
        media_type="text/event-stream",
    )


@app.post("/query")
def query(
    question: Question, cohere_clients: CohereClients, weaviate_client: WeaviateClient
) -> Answer:
    logging.debug(f"POST request received at /query...")

    # Return the full response
    return query_rag(
        question.question, question.rerank, cohere_clients, weaviate_client
    )


@app.post("/auploadfiles")
async def auploadfiles(
    files: list[UploadFile],
    cohere_async_clients: CohereAsyncClients,
    weaviate_async_client: WeaviateAsyncClient,
):
    logging.debug("POST request received at /auploadfiles...")

    await aupload_documents(files, cohere_async_clients, weaviate_async_client)


@app.post("/uploadfiles")
def uploadfiles(
    files: list[UploadFile],
    cohere_clients: CohereClients,
    weaviate_client: WeaviateClient,
):
    logging.debug("POST request received at /uploadfiles...")

    upload_documents(files, cohere_clients, weaviate_client)
