"""Provider-neutral OpenAI-compatible server for CogAgent-9B."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = os.getenv("MODEL_PATH", "zai-org/cogagent-9b-20241220")
SERVED_MODEL_NAME = os.getenv("SERVED_MODEL_NAME", "cogagent-9b-20241220")
API_KEY = os.getenv("COGAGENT_API_KEY", "")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DTYPE_NAME = os.getenv("COGAGENT_DTYPE", "bfloat16").lower()

tokenizer: Any = None
model: Any = None
generation_lock = asyncio.Lock()


class ChatCompletionRequest(BaseModel):
    model: str = SERVED_MODEL_NAME
    messages: list[dict[str, Any]]
    temperature: float = 0.0
    top_p: float = Field(default=1.0, gt=0, le=1)
    max_tokens: int = Field(default=512, ge=1, le=4096)


def _dtype() -> torch.dtype:
    choices = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if DTYPE_NAME not in choices:
        raise ValueError(f"Unsupported COGAGENT_DTYPE: {DTYPE_NAME}")
    return choices[DTYPE_NAME]


def _authorize(authorization: str | None = Header(default=None)) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def _decode_data_image(url: str) -> Image.Image:
    if not url.startswith("data:image/") or "," not in url:
        raise HTTPException(status_code=400, detail="A base64 data image URL is required")
    try:
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image payload") from exc


def _extract_query_and_image(messages: list[dict[str, Any]]) -> tuple[str, Image.Image]:
    text_parts: list[str] = []
    image: Image.Image | None = None
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else image_url
                image = _decode_data_image(url)
    if image is None:
        raise HTTPException(status_code=400, detail="A screenshot is required")
    query = "\n".join(part for part in text_parts if part).strip()
    if not query:
        raise HTTPException(status_code=400, detail="A text prompt is required")
    return query, image


def _generate(query: str, image: Image.Image, request: ChatCompletionRequest) -> str:
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "image": image, "content": query}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    input_length = inputs["input_ids"].shape[1]
    kwargs: dict[str, Any] = {
        "max_new_tokens": request.max_tokens,
        "do_sample": request.temperature > 0,
    }
    if request.temperature > 0:
        kwargs.update(temperature=request.temperature, top_p=request.top_p)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **kwargs)
    return tokenizer.decode(output_ids[0, input_length:], skip_special_tokens=True).strip()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=_dtype(),
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    yield
    tokenizer = None
    model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="CogAgent-9B OpenAI-compatible server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if model is not None else "loading",
        "model": SERVED_MODEL_NAME,
        "dtype": DTYPE_NAME,
    }


@app.get("/v1/models", dependencies=[Depends(_authorize)])
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "zai-org"}],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(_authorize)])
async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    if request.model not in {SERVED_MODEL_NAME, MODEL_PATH}:
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")
    query, image = _extract_query_and_image(request.messages)
    created = int(time.time())
    async with generation_lock:
        content = await asyncio.to_thread(_generate, query, image, request)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": SERVED_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def main() -> None:
    uvicorn.run(app, host=HOST, port=PORT, workers=1)


if __name__ == "__main__":
    main()
