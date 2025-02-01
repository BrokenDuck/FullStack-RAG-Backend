from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .config import cohere_async_clients, weaviate_async_client, settings
from .models import Question, Answer
from .upload import upload_documents
from .query import query_rag, stream_rag


@asynccontextmanager
async def lifespan(_: FastAPI):
    await weaviate_async_client.connect()
    yield
    await weaviate_async_client.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.allow_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    logging.debug("GET request received at root...")
    return {"Hello": "World"}


@app.post("/stream")
async def stream(question: Question) -> StreamingResponse:
    logging.debug(f"POST request received at /stream...")
    
    if not await weaviate_async_client.is_ready():
        raise HTTPException(status_code=503, detail="Weaviate is not ready.")

    return StreamingResponse(
        stream_rag(
            question.question,
            question.rerank,
            cohere_async_clients,
            weaviate_async_client,
        ),
        media_type="text/event-stream",
    )


@app.post("/query")
async def query(question: Question) -> Answer:
    logging.debug(f"POST request received at /query...")
    
    if not await weaviate_async_client.is_ready():
        raise HTTPException(status_code=503, detail="Weaviate is not ready.")

    # Return the full response
    response = await query_rag(
        question.question, question.rerank, cohere_async_clients, weaviate_async_client
    )
    return response


@app.post("/uploadfiles")
async def uploadfiles(files: list[UploadFile]):
    logging.debug("POST request received at /uploadfiles...")
    
    if not await weaviate_async_client.is_ready():
        raise HTTPException(status_code=503, detail="Weaviate is not ready.")

    errored_files = await upload_documents(files, cohere_async_clients, weaviate_async_client)
    
    if len(errored_files) != 0:
        return HTTPException(status_code=500, detail=f"File Upload failed for files: {", ".join(errored_files)}")
