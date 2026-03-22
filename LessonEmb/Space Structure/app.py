import torch
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

MODEL_NAME = os.getenv("MODEL_NAME", "jhu-clsp/mmBERT-base")

app = FastAPI(title="ModernBERT Embedding API", version="1.0.0")

print("Loading model:", MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


class EmbedRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/embed")
def embed(req: EmbedRequest):
    text = (req.text or "").strip()

    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    with torch.no_grad():
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        outputs = model(**inputs)

        mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

        emb = embeddings[0].tolist()

    return {
        "model": MODEL_NAME,
        "dim": len(emb),
        "preview_first_8": [round(x, 4) for x in emb[:8]],
        "embedding": emb,
    }