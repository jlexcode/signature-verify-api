from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import httpx
import os
from typing import List
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

app = FastAPI()

# CORS middleware (for dev; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Env vars
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# SSIM threshold
THRESHOLD = 0.85  # You can tune this

# === Models ===
class Point(BaseModel):
    x: float
    y: float
    time: float
    pressure: float

class Stroke(BaseModel):
    penColor: str
    dotSize: float
    minWidth: float
    maxWidth: float
    velocityFilterWeight: float
    compositeOperation: str
    points: List[Point]

class SignatureInput(BaseModel):
    student_id: str
    signature_path: List[Stroke]

# === Routes ===
@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.post("/verify")
async def verify_signature(payload: SignatureInput):
    # Fetch reference from Supabase
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/signature_reference",
            headers=headers,
            params={"student_id": f"eq.{payload.student_id}", "select": "*"}
        )

    if res.status_code != 200 or not res.json():
        raise HTTPException(status_code=404, detail="Reference signature not found")

    reference = res.json()[0]
    ref_path = [Stroke(**s) for s in json.loads(reference['signature_path'])]
    submitted_path = payload.signature_path

    score = compare_signatures_ssim(submitted_path, ref_path)
    match = score >= THRESHOLD

    return {"match": match, "score": round(score, 4)}

# === SSIM Comparison Logic ===
def draw_signature_to_image(strokes: List[Stroke], size=256) -> Image.Image:
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)

    all_points = [pt for stroke in strokes for pt in stroke.points]
    if not all_points:
        return img

    xs = [p.x for p in all_points]
    ys = [p.y for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x + 1e-6
    height = max_y - min_y + 1e-6

    def normalize(x, y):
        return (
            int(((x - min_x) / width) * (size - 1)),
            int(((y - min_y) / height) * (size - 1))
        )

    for stroke in strokes:
        if len(stroke.points) < 2:
            continue
        for i in range(len(stroke.points) - 1):
            p1 = normalize(stroke.points[i].x, stroke.points[i].y)
            p2 = normalize(stroke.points[i+1].x, stroke.points[i+1].y)
            draw.line([p1, p2], fill=0, width=2)

    return img

def compare_signatures_ssim(path1: List[Stroke], path2: List[Stroke]) -> float:
    img1 = draw_signature_to_image(path1)
    img2 = draw_signature_to_image(path2)
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    return ssim(arr1, arr2)