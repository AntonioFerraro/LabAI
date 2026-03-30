from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
import os

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

app = FastAPI(title="Qwen Mini Extractor", version="3.1.0")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
model.eval()

SYSTEM_PROMPT = """
You extract structured candidate or job information from text.
Return only valid JSON.
No markdown.
No explanations.
Do not invent information.
If a field is missing, use empty string or empty list.
All list fields must contain strings only.
"""

def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\s*Text:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def build_user_prompt(text: str, document_type: str) -> str:
    return f"""
Document type: {document_type}

Return ONLY this JSON schema:

{{
  "job_title": "",
  "skills": [],
  "experiences": [],
  "location": "",
  "summary": ""
}}

Rules:
- job_title = current role or most relevant target role
- if job_title is missing, use the most recent experience title
- experiences = past experience titles only, as strings, ordered from most recent to oldest when possible
- skills = concise list of professional skills
- location = main location if present
- summary = very short summary, max 25 words
- no nested objects
- no extra keys
- no text before or after JSON
- do not use null
- if unknown, use "" or []

Text:
{text}
"""

def extract_json_block(text: str) -> dict:
    text = text.strip()

    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return json.loads(fence_match.group(1))

    fence_match_generic = re.search(r"```\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match_generic:
        return json.loads(fence_match_generic.group(1))

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])

    raise ValueError("No balanced JSON object found")

def to_string_list(value) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, str):
                s = v.strip()
                if s:
                    out.append(s)
            elif v is not None:
                s = str(v).strip()
                if s:
                    out.append(s)
        return list(dict.fromkeys(out))

    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []

    s = str(value).strip()
    return [s] if s else []

def clean_scalar(value) -> str:
    if value is None:
        return ""

    s = str(value).strip()

    invalid_values = {
        "n/a",
        "na",
        "none",
        "null",
        "unknown",
        "not specified",
        "not provided",
        "-"
    }

    if s.lower() in invalid_values:
        return ""

    return s

def normalize_profile(profile: dict) -> dict:
    if not isinstance(profile, dict):
        profile = {}

    job_title = clean_scalar(profile.get("job_title", ""))
    skills = to_string_list(profile.get("skills", []))
    experiences = to_string_list(profile.get("experiences", []))
    location = clean_scalar(profile.get("location", ""))
    summary = clean_scalar(profile.get("summary", ""))

    if not job_title and experiences:
        job_title = experiences[0].strip()

    return {
        "job_title": job_title,
        "skills": skills,
        "experiences": experiences,
        "location": location,
        "summary": summary,
    }

class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1)
    document_type: str = "generic"

class ExtractResponse(BaseModel):
    profile: dict
    model: str
    raw_output: str | None = None

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/extract_profile", response_model=ExtractResponse)
def extract_profile(payload: ExtractRequest):
    text = normalize_text(payload.text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text, payload.document_type)}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    try:
        raw_profile = extract_json_block(generated)
        profile = normalize_profile(raw_profile)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": str(e),
                "raw_output": generated
            }
        )

    return {
        "profile": profile,
        "model": MODEL_NAME,
        "raw_output": generated
    }