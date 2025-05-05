from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class SignatureData(BaseModel):
    student_id: str
    path: list  # assume this is SignaturePad's .toData() array

def flatten_signature(path):
    # Flatten SignaturePad strokes for naive comparison
    points = []
    for stroke in path:
        points.extend(stroke['points'])
    return np.array([[p['x'], p['y']] for p in points])

def compute_similarity(path1, path2):
    p1 = flatten_signature(path1)
    p2 = flatten_signature(path2)
    if len(p1) == 0 or len(p2) == 0:
        return 0
    min_len = min(len(p1), len(p2))
    return np.mean(np.linalg.norm(p1[:min_len] - p2[:min_len], axis=1))

@app.post("/verify")
async def verify_signature(sig: SignatureData):
    # TODO: load reference from Supabase
    reference_path = ...  # replace with actual fetch
    similarity = compute_similarity(sig.path, reference_path)
    return { "score": similarity, "match": similarity < 15 }  # tune threshold