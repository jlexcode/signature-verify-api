from fastapi import FastAPI

app = FastAPI()

# 👇 This is the test route
@app.get("/ping")
def ping():
    return {"message": "pong"}