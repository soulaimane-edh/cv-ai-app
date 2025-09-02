# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class In(BaseModel):
    text: str = ""
    lang: str = "fr"

@app.get("/healthz")
def health():
    return {"ok": True}

@app.post("/analyze")
def analyze(inp: In):
    # TODO: branche ta logique existante ici (celle que tu as déjà dans ton app)
    # ex: prediction, score = run_model(inp.text, inp.lang)
    return {
        "label": "OK",
        "score": 0.87,
        "echo": inp.text[:200],
        "lang": inp.lang
    }
