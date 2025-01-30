from .prompts import query_generation_instruction
from .tools import query_generation_tool

import cohere
from cohere import UserChatMessageV2, SystemChatMessageV2, Document, CitationOptions
import weaviate

import json
import asyncio
import httpx

from typing import AsyncGenerator, Generator
from .models import AnswerChunk, Answer


async def aget_documents_rerank(
    question: str,
    cohere_async_clients: dict[str, cohere.AsyncClientV2],
    weaviate_async_client: weaviate.WeaviateAsyncClient,
) -> list[Document]:
    res = await cohere_async_clients["command_r_async_client"].chat(
        model="command-r-08-2024",
        messages=[
            SystemChatMessageV2(content=query_generation_instruction),
            UserChatMessageV2(content=question),
        ],
        tools=[query_generation_tool],
    )

    search_queries = list()
    if res.message.tool_calls:
        for tc in res.message.tool_calls:
            queries = json.loads(tc.function.arguments)["queries"]
            search_queries.extend(queries)

    search_queries_embeddings_response = await cohere_async_clients[
        "embed_english_async_client"
    ].embed(
        texts=search_queries,
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"],
    )

    with weaviate_async_client:
        await weaviate_async_client.is_ready()
        collection = weaviate_async_client.collections.get(name="Documents")
        tasks = [
            collection.query.hybrid(search_queries[i], vector=embedding, limit=5)
            for i, embedding in enumerate(
                search_queries_embeddings_response.embeddings.float
            )
        ]
        chunks_responses = await asyncio.gather(*tasks)

    chunks = [
        {
            "title": object.properties["title"],
            "chunk_content": object.properties["chunk_content"],
        }
        for chunks_response in chunks_responses
        for object in chunks_response.objects
    ]

    rerank_response = await cohere_async_clients["rerank_english_async_client"].rerank(
        model="rerank-v3.5",
        query=question,
        documents=map(lambda x: x["chunk_content"], chunks),
        top_n=3,
    )

    documents = [
        Document(
            id=str(i),
            data={
                "text": chunks[result.index]["chunk_content"],
                "title": chunks[result.index]["title"],
            },
        )
        for i, result in enumerate(rerank_response.results)
    ]
    return documents


async def aget_documents(
    question: str,
    cohere_async_clients: dict[str, cohere.AsyncClientV2],
    weaviate_async_client: weaviate.WeaviateAsyncClient,
) -> list[Document]:

    question_embedding_response = await cohere_async_clients[
        "embed_english_async_client"
    ].embed(
        texts=[question],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"],
    )

    async with weaviate_async_client:
        await weaviate_async_client.is_ready()
        collection = weaviate_async_client.collections.get(name="Documents")
        query_response = await collection.query.hybrid(
            question,
            vector=question_embedding_response.embeddings.float[0],
            limit=5,
        )

    documents = [
        Document(
            id=str(i),
            data={
                "text": object.properties["chunk_content"],
                "title": object.properties["title"],
            },
        )
        for i, object in enumerate(query_response.objects)
    ]

    return documents


async def astream_rag(
    question: str,
    rerank: bool,
    cohere_async_clients: dict[str, cohere.AsyncClientV2],
    weaviate_async_client: weaviate.WeaviateAsyncClient,
) -> AsyncGenerator[AnswerChunk, None]:
    if rerank is True:
        documents = await aget_documents_rerank(
            question, cohere_async_clients, weaviate_async_client
        )
    else:
        documents = await aget_documents(
            question, cohere_async_clients, weaviate_async_client
        )

    response = cohere_async_clients["command_r_async_client"].chat_stream(
        model="command-r-08-2024",
        messages=[UserChatMessageV2(content=question)],
        documents=documents,
        citation_options=CitationOptions(mode="FAST"),
    )

    try:
        async for chunk in response:
            if chunk:
                if chunk.type == "content-delta":
                    yield json.dumps(
                        {
                            "type": "response-chunk",
                            "text": chunk.delta.message.content.text,
                        }
                    )
                elif chunk.type == "citation-start":
                    yield json.dumps(
                        {
                            "type": "citation",
                            "title": chunk.delta.message.citations.sources[0].document[
                                "title"
                            ],
                            "text": chunk.delta.message.citations.sources[0].document[
                                "text"
                            ],
                        }
                    )
    except httpx.ReadError:
        pass


async def aquery_rag(
    question: str,
    rerank: bool,
    cohere_async_clients: dict[str, cohere.AsyncClientV2],
    weaviate_async_client: weaviate.WeaviateAsyncClient,
) -> Answer:

    if rerank is True:
        documents = await aget_documents_rerank(
            question, cohere_async_clients, weaviate_async_client
        )
    else:
        documents = await aget_documents(
            question, cohere_async_clients, weaviate_async_client
        )

    response = await cohere_async_clients["command_r_async_client"].chat(
        model="command-r-08-2024",
        messages=[UserChatMessageV2(content=question)],
        documents=documents,
    )

    return {
        "answer": response.message.content[0].text,
        "citations": [
            {
                "title": citation.sources[0].document["title"],
                "text": citation.sources[0].document["text"],
            }
            for citation in response.message.citations
        ],
    }


def get_documents_rerank(
    question: str,
    cohere_clients: dict[str, cohere.ClientV2],
    weaviate_client: weaviate.WeaviateClient,
) -> list[Document]:

    res = cohere_clients["command_r_client"].chat(
        model="command-r-08-2024",
        messages=[
            SystemChatMessageV2(content=query_generation_instruction),
            UserChatMessageV2(content=question),
        ],
        tools=[query_generation_tool],
    )

    search_queries = list()
    if res.message.tool_calls:
        for tc in res.message.tool_calls:
            queries = json.loads(tc.function.arguments)["queries"]
            search_queries.extend(queries)

    search_queries_embeddings_response = cohere_clients["embed_english_client"].embed(
        texts=search_queries,
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"],
    )

    with weaviate_client:
        collection = weaviate_client.collections.get(name="Documents")
        chunks = list()
        for embedding in search_queries_embeddings_response.embeddings.float:
            query_response = collection.query.hybrid(
                question, vector=embedding, limit=5
            )
            for object in query_response.objects:
                chunks.append(
                    {
                        "title": object.properties["title"],
                        "chunk_content": object.properties["chunk_content"],
                    }
                )

    rerank_response = cohere_clients["rerank_english_client"].rerank(
        model="rerank-v3.5",
        query=question,
        documents=map(lambda x: x["chunk_content"], chunks),
        top_n=3,
    )

    documents = [
        Document(
            id=str(i),
            data={
                "text": chunks[result.index]["chunk_content"],
                "title": chunks[result.index]["title"],
            },
        )
        for i, result in enumerate(rerank_response.results)
    ]
    return documents


def get_documents(
    question: str,
    cohere_clients: dict[str, cohere.ClientV2],
    weaviate_client: weaviate.WeaviateClient,
) -> list[Document]:
    question_embedding_response = cohere_clients["embed_english_client"].embed(
        texts=[question],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"],
    )

    with weaviate_client:
        collection = weaviate_client.collections.get(name="Documents")
        query_response = collection.query.hybrid(
            question,
            vector=question_embedding_response.embeddings.float[0],
            limit=5,
        )

    documents = [
        Document(
            id=str(i),
            data={
                "text": object.properties["chunk_content"],
                "title": object.properties["title"],
            },
        )
        for i, object in enumerate(query_response.objects)
    ]

    return documents


def stream_rag(
    question: str,
    rerank: bool,
    cohere_clients: dict[str, cohere.ClientV2],
    weaviate_client: weaviate.WeaviateClient,
) -> Generator[AnswerChunk, None, None]:
    if rerank is True:
        documents = get_documents_rerank(question, cohere_clients, weaviate_client)
    else:
        documents = get_documents(question, cohere_clients, weaviate_client)

    response = cohere_clients["command_r_plus_client"].chat_stream(
        model="command-r-08-2024",
        messages=[UserChatMessageV2(content=question)],
        documents=documents,
        citation_options=CitationOptions(mode="ACCURATE"),
    )

    try:
        for chunk in response:
            if chunk:
                if chunk.type == "content-delta":
                    yield json.dumps(
                        {
                            "type": "response-chunk",
                            "text": chunk.delta.message.content.text,
                        }
                    )
                elif chunk.type == "citation-start":
                    yield json.dumps(
                        {
                            "type": "citation",
                            "title": chunk.delta.message.citations.sources[0].document[
                                "title"
                            ],
                            "text": chunk.delta.message.citations.sources[0].document[
                                "text"
                            ],
                        }
                    )
    except httpx.RemoteProtocolError:
        pass


def query_rag(
    question: str,
    rerank: bool,
    cohere_clients: dict[str, cohere.ClientV2],
    weaviate_client: weaviate.WeaviateClient,
) -> Answer:

    if rerank is True:
        documents = get_documents_rerank(question, cohere_clients, weaviate_client)
    else:
        documents = get_documents(question, cohere_clients, weaviate_client)

    response = cohere_clients["command_r_client"].chat(
        model="command-r-08-2024",
        messages=[UserChatMessageV2(content=question)],
        documents=documents,
    )

    return {
        "answer": response.message.content[0].text,
        "citations": [
            {
                "title": citation.sources[0].document["title"],
                "text": citation.sources[0].document["text"],
            }
            for citation in response.message.citations
        ],
    }
