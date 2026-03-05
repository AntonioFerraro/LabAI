import os
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

## you can change model. On hugging face go to Models and then you have the ID. For example:
## Nanbeige/Nanbeige4.1-3B

## Careful about how big the model is, as HF free resources are limited

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

_pipe = None
_pipe_lock = threading.Lock()

class Request(BaseModel):
    prompt: str
    temperature: float = 0.0
    max_tokens: int = 50 ## you can pass the parameter in the request


@app.get("/")
def health():
    return {"status": "running", "model_loaded": _pipe is not None}


def get_pipe():
    global _pipe
    if _pipe is None:
        with _pipe_lock:
            if _pipe is None:
                _pipe = pipeline(
                    "text-generation",
                    model=MODEL_ID,
                    device=-1
                )
    return _pipe


@app.post("/generate") ## this is the endpoint that you call in the notebook
def generate(req: Request):
    try:
        pipe = get_pipe()

        do_sample = req.temperature > 0

        out = pipe(
            req.prompt,
            max_new_tokens=int(req.max_tokens),
            temperature=float(req.temperature),
            do_sample=do_sample,
            return_full_text=False
        )

        return {"response": out[0]["generated_text"].strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))