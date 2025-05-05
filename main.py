from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import math
import httpx
import os
from typing import List, Dict
import numpy as np



app = FastAPI()

# Load from environment or hardcode for now
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

#set threhold

THRESHOLD = 0.15

# Pydantic model to validate input

class SignatureInput(BaseModel):
    student_id: str
    signature_path: List[List[Dict[str, float]]]  # List of strokes (list of points with x and y)

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.post("/verify")
async def verify_signature(payload: SignatureInput):
    # Fetch reference signature from Supabase
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/signature_reference",
            headers=headers,
            params={
                "student_id": f"eq.{payload.student_id}",
                "select": "*"  # This is critical
            })

    if res.status_code != 200 or not res.json():
        raise HTTPException(status_code=404, detail="Reference signature not found")

    reference = res.json()[0]
    ref_path = json.loads(reference['signature_path'])
    submitted_path = payload.signature_path

    # Run comparison
    score = compare_paths(submitted_path, ref_path)
    match = score < THRESHOLD  # you can tune this threshold later

    return {"match": match, "score": round(score, 4)}


def compare_paths(path1, path2, num_points=100):
    def flatten_and_resample(path):
        points = [pt for stroke in path for pt in stroke]
        if len(points) < 2:
            return np.zeros((num_points, 2))  # avoid div by zero
        xs = [pt['x'] for pt in points]
        ys = [pt['y'] for pt in points]
        # Normalize to [0,1] box to handle scale variance
        xs = np.array(xs)
        ys = np.array(ys)
        xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-6)
        ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-6)
        # Resample to fixed number of points
        coords = np.stack([xs, ys], axis=1)
        if len(coords) < num_points:
        # Repeat or pad last point
            coords = np.pad(coords, ((0, num_points - len(coords)), (0, 0)), mode='edge')
        indices = np.linspace(0, len(coords) - 1, num_points).astype(int)
        resampled = coords[indices]
        return resampled
    pts1 = flatten_and_resample(path1)
    pts2 = flatten_and_resample(path2)
    distances = np.linalg.norm(pts1 - pts2, axis=1)
    return float(np.mean(distances))