import cohere
import weaviate
import weaviate.classes as wvc

from pydantic_settings import BaseSettings, SettingsConfigDict

FILE_CHUNK_SIZE = 1024 * 1024

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


class Settings(BaseSettings):
    weaviate_host: str
    weaviate_api_key: str
    command_r_url: str
    command_r_api_key: str
    command_r_plus_url: str = ""
    command_r_plus_api_key: str = ""
    embed_english_url: str = ""
    embed_english_api_key: str = ""
    rerank_english_url: str = ""
    rerank_english_api_key: str = ""
    allow_origins: str = "*"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()

cohere_async_clients = {
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

weaviate_async_client = weaviate.use_async_with_local(
    host=settings.weaviate_host,
    auth_credentials=wvc.init.Auth.api_key(settings.weaviate_api_key),
)
