import os, re, httpx, numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
PIXABAY_KEY = os.environ.get("PIXABAY_KEY", "")
USE_EMBEDDINGS = os.environ.get("USE_EMBEDDINGS", "1") == "1"

# 1) RULES: abstract lyric words -> searchable visuals (grow this over time)
RULES = {
    "heartbreak": "rain window", "cry": "rain window", "crying": "rain window",
    "tears": "rain drops", "love": "sunset couple", "kiss": "couple silhouette",
    "free": "open road", "freedom": "birds flying sky", "run": "road horizon",
    "alone": "empty street night", "lonely": "empty street night",
    "night": "city night lights", "dance": "party lights", "fire": "flames dark",
    "sea": "ocean waves", "ocean": "ocean waves", "sky": "clouds sky",
    "home": "cozy house window", "dream": "starry sky", "dreams": "starry sky",
    "star": "starry sky", "stars": "starry sky", "rain": "rain window",
    "sun": "sunrise", "morning": "sunrise mountains", "road": "open road",
    "river": "river flowing", "moon": "moon night", "summer": "summer beach",
    "snow": "snowy landscape", "flower": "flowers field", "flowers": "flowers field",
    "city": "city night lights", "mountain": "mountain sunrise", "pray": "candle dark",
}

# 2) EMBEDDING FALLBACK concepts (add ~200 over time)
CONCEPTS = [
    "ocean waves", "city night lights", "forest path", "rain window",
    "mountain sunrise", "empty street night", "sunset beach", "starry sky",
    "open road", "flowers field", "snowy landscape", "candle dark",
    "party lights", "birds flying sky", "river flowing", "cozy house window",
]

_model = None
_vecs = None

def get_model():
    global _model, _vecs
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        _vecs = np.array(list(_model.embed(CONCEPTS)))
        _vecs /= np.linalg.norm(_vecs, axis=1, keepdims=True)
    return _model, _vecs

def rule_match(line):
    for w in re.findall(r"[a-z']+", line.lower()):
        if w in RULES:
            return RULES[w]
    return None

def embed_match(line):
    model, vecs = get_model()
    v = np.array(list(model.embed([line])))[0]
    v /= np.linalg.norm(v)
    return CONCEPTS[int(np.argmax(vecs @ v))]

# 3) PIXABAY (images + videos) with in-memory cache
_cache = {}

async def pixabay(query, style, media):
    key = f"{media}|{query}|{style}"
    if key in _cache:
        return _cache[key]
    q = f"{query} {style}".strip()
    url = None
    async with httpx.AsyncClient(timeout=12) as c:
        if media == "video":
            r = await c.get("https://pixabay.com/api/videos/",
                            params={"key": PIXABAY_KEY, "q": q, "per_page": 5, "safesearch": "true"})
            hits = r.json().get("hits", []) if r.status_code == 200 else []
            if hits:
                v = hits[0]["videos"]
                url = (v.get("medium") or v.get("small") or v.get("tiny") or {}).get("url")
        else:
            r = await c.get("https://pixabay.com/api/",
                            params={"key": PIXABAY_KEY, "q": q, "image_type": "photo",
                                    "orientation": "horizontal", "safesearch": "true", "per_page": 5})
            hits = r.json().get("hits", []) if r.status_code == 200 else []
            if hits:
                url = hits[0]["largeImageURL"]
    _cache[key] = url
    return url

class Req(BaseModel):
    lines: list[str]
    style: str = ""
    media: str = "image"          # "image" or "video"
    default_query: str = "abstract light"

@app.post("/visuals")
async def visuals(req: Req):
    out, prev_q, prev_url = [], None, None
    for line in req.lines:
        if not line.strip():
            q = prev_q or req.default_query
        else:
            q = rule_match(line)
            if not q and USE_EMBEDDINGS:
                try: q = embed_match(line)
                except Exception: q = None
            q = q or prev_q or req.default_query
        if q == prev_q and prev_url:          # adjacent lines share visual -> no flicker
            url = prev_url
        else:
            url = await pixabay(q, req.style, req.media) or await pixabay(req.default_query, req.style, req.media)
        out.append({"line": line, "query": q, "url": url, "media": req.media})
        prev_q, prev_url = q, url
    return out

@app.get("/health")
def health():
    return {"ok": True, "pixabay_key_set": bool(PIXABAY_KEY)}

@app.get("/")
def index():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")
