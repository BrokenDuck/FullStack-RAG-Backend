import cohere
import weaviate
import weaviate.classes as wvc

from functools import lru_cache

from fastapi import Depends

from pydantic_settings import BaseSettings, SettingsConfigDict

from typing import Annotated

FILE_CHUNK_SIZE = 1024 * 1024

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


class Settings(BaseSettings):
    weaviate_url: str
    weaviate_api_key: str
    command_r_url: str
    command_r_api_key: str
    command_r_plus_url: str
    command_r_plus_api_key: str
    embed_english_url: str
    embed_english_api_key: str
    rerank_english_url: str
    rerank_english_api_key: str

    model_config = SettingsConfigDict(env_file="../.env")


@lru_cache
def get_settings():
    return Settings()


async def get_cohere_async_clients():
    settings = get_settings()
    return {
        "command_r_async_client": cohere.AsyncClientV2(
            api_key=settings.command_r_api_key, base_url=settings.command_r_url
        ),
        "command_r_plus_async_client": cohere.AsyncClientV2(
            api_key=settings.command_r_plus_api_key,
            base_url=settings.command_r_plus_url,
        ),
        "embed_english_async_client": cohere.AsyncClientV2(
            api_key=settings.embed_english_api_key, base_url=settings.embed_english_url
        ),
        "rerank_english_async_client": cohere.AsyncClientV2(
            api_key=settings.rerank_english_api_key,
            base_url=settings.rerank_english_url,
        ),
    }


CohereAsyncClients = Annotated[
    dict[str, cohere.AsyncClientV2], Depends(get_cohere_async_clients)
]


def get_cohere_clients():
    settings = get_settings()
    return {
        "command_r_client": cohere.ClientV2(
            api_key=settings.command_r_api_key, base_url=settings.command_r_url
        ),
        "command_r_plus_client": cohere.ClientV2(
            api_key=settings.command_r_plus_api_key,
            base_url=settings.command_r_plus_url,
        ),
        "embed_english_client": cohere.ClientV2(
            api_key=settings.embed_english_api_key, base_url=settings.embed_english_url
        ),
        "rerank_english_client": cohere.ClientV2(
            api_key=settings.rerank_english_api_key,
            base_url=settings.rerank_english_url,
        ),
    }


CohereClients = Annotated[dict[str, cohere.ClientV2], Depends(get_cohere_clients)]


async def get_weaviate_async_client():
    settings = get_settings()
    return weaviate.use_async_with_local(
        auth_credentials=wvc.init.Auth.api_key(settings.weaviate_api_key)
    )


WeaviateAsyncClient = Annotated[
    weaviate.WeaviateAsyncClient, Depends(get_weaviate_async_client)
]


def get_weaviate_client():
    settings = get_settings()
    return weaviate.connect_to_local(
        auth_credentials=wvc.init.Auth.api_key(settings.weaviate_api_key)
    )


WeaviateClient = Annotated[weaviate.WeaviateClient, Depends(get_weaviate_client)]
