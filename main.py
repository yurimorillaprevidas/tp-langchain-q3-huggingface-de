from contextlib import asynccontextmanager
import re

from fastapi import FastAPI, Request
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, ConfigDict, Field
from transformers import pipeline


MODEL = "Helsinki-NLP/opus-mt-en-de"
SOURCE_TOKENS = 250


@asynccontextmanager
async def lifespan(app: FastAPI):
    translator = pipeline("translation", model=MODEL, device=-1)
    app.state.tokenizer = translator.tokenizer
    app.state.translator = HuggingFacePipeline(pipeline=translator, model_id=MODEL, batch_size=4)
    yield


app = FastAPI(title="Tradução do inglês para o alemão com LangChain", lifespan=lifespan)


class TranslationInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=20000)


class TranslationOutput(BaseModel):
    translation: str


def split_text(text, tokenizer):
    if len(tokenizer.tokenize(text)) <= SOURCE_TOKENS:
        return [text]

    # Corta perto do meio, de preferência entre palavras.
    middle = len(text) // 2
    boundary = text.rfind(" ", middle // 2, middle)
    cut = boundary if boundary > 0 else middle
    return split_text(text[:cut].strip(), tokenizer) + split_text(text[cut:].strip(), tokenizer)


@app.post("/translate", response_model=TranslationOutput)
def translate(body: TranslationInput, request: Request):
    state = request.app.state
    paragraphs = re.split(r"\r?\n(?:[ \t]*\r?\n)+", body.text)
    translations = []
    for paragraph in paragraphs:
        chunks = split_text(paragraph.strip(), state.tokenizer)
        results = state.translator.batch(chunks)
        translations.append(" ".join(results))
    return {"translation": "\n\n".join(translations)}
