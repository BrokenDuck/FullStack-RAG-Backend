import pymupdf4llm
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import (
    ExperimentalMarkdownSyntaxTextSplitter,
    RecursiveCharacterTextSplitter,
)
import cohere
import weaviate
import weaviate.classes as wvc
from asyncer import asyncify
import asyncio
import logging

from fastapi import UploadFile

from .config import CHUNK_SIZE, CHUNK_OVERLAP


async def process_file(
    file: UploadFile,
    markdown_text_splitter: ExperimentalMarkdownSyntaxTextSplitter,
    text_splitter: RecursiveCharacterTextSplitter,
    cohere_async_clients: dict[str, cohere.AsyncClientV2],
    weaviate_async_client: weaviate.WeaviateAsyncClient,
):
    logging.info('Extracting markdown...')
    # Parse pdf, also extracting tables
    md_text = await asyncify(pymupdf4llm.to_markdown)(
        pymupdf.open(stream=file.file.read(), filetype="pdf")
    )

    # We chunk the document
    md_header_splits = markdown_text_splitter.split_text(md_text)
    splits = text_splitter.split_documents(md_header_splits)
    for split in splits:
        split.metadata["title"] = ": ".join(
            split.metadata[f"Header {i}"]
            for i in range(1, 7)
            if f"Header {i}" in split.metadata
        )
        split.metadata["filename"] = file.filename

    collection = weaviate_async_client.collections.get("Documents")

    async def upload_splits(splits: list[Document]):
        # Create the embeddings
        # We use the english model, in case the document set also contains other languages the multilingual model should be used
        logging.info("Getting embeddings...")
        response = await cohere_async_clients["embed_english_async_client"].embed(
            texts=[split.page_content for split in splits],
            model="embed-english-v3.0",
            input_type="search_document",
            embedding_types=["float"],
        )

        logging.info("Uploading embeddings...")
        # Upload documents to the database
        document_objs = list()
        for i, emb in enumerate(response.embeddings.float):
            document_objs.append(
                wvc.data.DataObject(
                    properties={
                        "filename": splits[i].metadata["filename"],
                        "title": splits[i].metadata["title"],
                        "chunk_content": splits[i].page_content,
                    },
                    vector=emb,
                )
            )
        response = await collection.data.insert_many(document_objs)
        return response.has_errors

    batch_size = 96
    tasks = [
        upload_splits(splits[i : i + batch_size])
        for i in range(0, len(splits), batch_size)
    ]
    responses = await asyncio.gather(*tasks)
    return any(list(responses))


async def upload_documents(
    files: list[UploadFile],
    cohere_async_clients: dict[str, cohere.AsyncClientV2],
    weaviate_async_client: weaviate.WeaviateAsyncClient,
):

    # This splitter splits markdown based on header content, this allows for semantic parsing
    # We use the experimental version because is retains whitespaces better for tables extracted by pymupdf4llm
    markdown_text_splitter = ExperimentalMarkdownSyntaxTextSplitter()

    # Extra splitter in case the header chunks are too large for the cohere embedder
    # We choose the default sensible settings for the english language (they fit cohere embeddings)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    tasks = [
        process_file(
            file,
            markdown_text_splitter,
            text_splitter,
            cohere_async_clients,
            weaviate_async_client,
        )
        for file in files
    ]
    responses = await asyncio.gather(*tasks)
    filenames = [files[i].filename for i in range(len(files)) if responses[i] is True]
    return filenames