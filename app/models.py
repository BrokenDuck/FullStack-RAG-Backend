from pydantic import BaseModel, Field
from typing import Literal, Optional


class Question(BaseModel):
    question: str = Field(
        max_length=500, description="The question asked to the RAG pipeline."
    )
    rerank: bool = Field(
        default=False, description="Use more advanced reranking models"
    )


class Answer(BaseModel):
    answer: str = Field(description="The response of the RAG pipeline.")
    citations: list[dict[str, str]] = Field(
        default_factory=list,
        description="A list of citations with a title and content.",
    )


class AnswerChunk(BaseModel):
    type: Literal["response-chunk", "citation"] = Field(
        description="The type of the chunk, either a chunk of the response or a citation."
    )
    title: Optional[str] = Field(
        description="The title of the citation if the type of the chunk is 'citation'"
    )
    text: str = Field(
        description="The chunk content. A part of the answer if chunk type is 'response-chunk'. The citation content if chunk type is 'citation'"
    )
