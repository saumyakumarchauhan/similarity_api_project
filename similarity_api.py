import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import httpx
import os
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["OPTIONS", "POST"],
    allow_headers=["*"],
)

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
if not AIPIPE_TOKEN:
    raise RuntimeError("Please set the AIPIPE_TOKEN environment variable")

AIPIPE_BASE_URL = "https://aipipe.org/openai/v1"
EMBEDDING_MODEL = "text-embedding-3-small"

class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

class SimilarityResponse(BaseModel):
    matches: List[str]

async def get_embedding(text: str) -> List[float]:
    url = f"{AIPIPE_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json"
    }
    json_payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=json_payload)
        response.raise_for_status()
        data = response.json()
        embedding = data["data"][0]["embedding"]
        return embedding

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

@app.post("/similarity", response_model=SimilarityResponse)
async def similarity(request: SimilarityRequest):
    try:
        docs_embeddings = await asyncio.gather(*[get_embedding(doc) for doc in request.docs])
        query_embedding = await get_embedding(request.query)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=500, detail=f"Embedding service error: {str(e)}")

    query_vec = np.array(query_embedding)
    sims = []
    for i, doc_emb in enumerate(docs_embeddings):
        doc_vec = np.array(doc_emb)
        sim_score = cosine_similarity(query_vec, doc_vec)
        sims.append((sim_score, request.docs[i]))

    sims.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in sims[:3]]

    return {"matches": top_docs}
