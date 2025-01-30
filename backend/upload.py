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

import asyncio

from fastapi import UploadFile

from .config import CHUNK_SIZE, CHUNK_OVERLAP


async def aprocess_file(
    file: UploadFile,
    markdown_text_splitter: ExperimentalMarkdownSyntaxTextSplitter,
    text_splitter: RecursiveCharacterTextSplitter,
    cohere_async_clients: dict[str, cohere.AsyncClientV2],
    weaviate_async_client: weaviate.WeaviateAsyncClient,
):
    # Parse pdf, also extracting tables
    md_text = pymupdf4llm.to_markdown(
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
        response = await cohere_async_clients["embed_english_async_client"].embed(
            texts=[split.page_content for split in splits],
            model="embed-english-v3.0",
            input_type="search_document",
            embedding_types=["float"],
        )
        print("Embedding async to cohere!")

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


async def aupload_documents(
    files: list[UploadFile],
    cohere_async_clients: dict[str, cohere.AsyncClientV2],
    weaviate_async_client: weaviate.WeaviateAsyncClient,
):

    # This splitter splits markdown based on header content, this allows for semantic parsing
    # We use the experimental version because is retains whitespaces better for tables extracted by pymupdf4llm
    markdown_text_splitter = ExperimentalMarkdownSyntaxTextSplitter(
        strip_headers=False,
    )

    # Extra splitter in case the header chunks are too large for the cohere embedder
    # We choose the default sensible settings for the english language (they fit cohere embeddings)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    # Context to automatically open and close the connection to the database
    await weaviate_async_client.connect()

    tasks = [
        aprocess_file(
            file,
            markdown_text_splitter,
            text_splitter,
            cohere_async_clients,
            weaviate_async_client,
        )
        for file in files
    ]
    responses = await asyncio.gather(*tasks)
    print("Uploading files!")
    filenames = [files[i].filename for i in range(len(files)) if responses[i] is True]
    await weaviate_async_client.close()
    return filenames


def process_file(
    file: UploadFile,
    weaviate_collection: weaviate.collections.Collection,
    markdown_text_splitter: ExperimentalMarkdownSyntaxTextSplitter,
    text_splitter: RecursiveCharacterTextSplitter,
    cohere_clients: dict[str, cohere.ClientV2],
):
    # Parse pdf, also extracting tables
    md_text = pymupdf4llm.to_markdown(
        pymupdf.Document(stream=file.file.read(), filetype="pdf")
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

    # Create the embeddings
    # We use the english model, in case the document set also contains other languages the multilingual model should be used
    batch_size = 96
    has_errors = False
    for i in range(0, len(splits), batch_size):
        batch_splits = splits[i : i + batch_size]
        response = cohere_clients["embed_english_client"].embed(
            texts=[split.page_content for split in batch_splits],
            model="embed-english-v3.0",
            input_type="search_document",
            embedding_types=["float"],
        )
        document_objs = list()
        for j, emb in enumerate(response.embeddings.float):
            document_objs.append(
                wvc.data.DataObject(
                    properties={
                        "filename": splits[i + j].metadata["filename"],
                        "title": splits[i + j].metadata["title"],
                        "chunk_content": splits[i + j].page_content,
                    },
                    vector=emb,
                )
            )
        response = weaviate_collection.data.insert_many(document_objs)
        has_errors = has_errors or response.has_errors
    return has_errors


def upload_documents(
    files: list[UploadFile],
    cohere_clients: dict[str, cohere.ClientV2],
    weaviate_client: weaviate.WeaviateClient,
):

    # This splitter splits markdown based on header content, this allows for semantic parsing
    # We use the experimental version because is retains whitespaces better for tables extracted by pymupdf4llm
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_text_splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    # Extra splitter in case the header chunks are too large for the cohere embedder
    # We choose the default sensible settings for the english language (they fit cohere embeddings)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    # Context to automatically open and close the connection to the database
    filename_fails = []
    with weaviate_client:
        collection = weaviate_client.collections.get("Documents")
        for file in files:
            has_error = process_file(
                file, collection, markdown_text_splitter, text_splitter, cohere_clients
            )
            if has_error:
                filename_fails.append(file.filename)
    return filename_fails
