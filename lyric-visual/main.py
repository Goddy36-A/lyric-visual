import os, re, hmac, random, httpx, numpy as np
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
PIXABAY_KEY = os.environ.get("PIXABAY_KEY", "")
APP_API_KEY = os.environ.get("APP_API_KEY", "")
USE_EMBEDDINGS = os.environ.get("USE_EMBEDDINGS", "1") == "1"

# 1) RULES: abstract lyric words -> searchable visuals (grow this over time)
# A value can be a string OR a list of phrases (a random one is used each time).
RULES = {
    "heartbreak": ["rain window", "broken heart", "lonely rain city", "wilting rose", "sad silhouette"],
    "cry": ["rain window", "rain drops glass", "sad silhouette", "tears"],
    "crying": ["rain window", "rain drops glass", "sad silhouette", "tears"],
    "tears": ["rain drops", "rain window", "water drops macro"],
    "love": ["sunset couple", "holding hands", "couple beach", "romantic lights", "red roses"],
    "kiss": ["couple silhouette", "couple sunset kiss", "romantic couple"],
    "free": ["open road", "birds flying sky", "open field sky", "hot air balloon"],
    "freedom": ["birds flying sky", "open road", "eagle flying", "mountain top freedom"],
    "run": ["road horizon", "runner sunrise", "running path forest"],
    "alone": ["empty street night", "lonely bench", "lone tree field", "foggy road"],
    "lonely": ["empty street night", "lonely bench", "lone tree field", "foggy road"],
    "night": ["city night lights", "night sky", "street lights night", "neon city"],
    "dance": ["party lights", "dancing silhouette", "club lights", "concert crowd"],
    "fire": ["flames dark", "campfire night", "bonfire sparks", "burning embers"],
    "sea": ["ocean waves", "sea sunset", "beach waves", "ocean horizon"],
    "ocean": ["ocean waves", "sea sunset", "beach waves", "ocean horizon"],
    "sky": ["clouds sky", "blue sky clouds", "sunset sky", "dramatic clouds"],
    "home": ["cozy house window", "house lights evening", "family home", "village house"],
    "dream": ["starry sky", "milky way", "dreamy clouds", "lantern sky"],
    "dreams": ["starry sky", "milky way", "dreamy clouds", "lantern sky"],
    "star": ["starry sky", "milky way", "stars night"],
    "stars": ["starry sky", "milky way", "stars night"],
    "rain": ["rain window", "rain drops", "rainy street", "umbrella rain"],
    "sun": ["sunrise", "sun rays", "golden sunlight", "sunset"],
    "morning": ["sunrise mountains", "morning mist", "sunrise field", "dawn sky"],
    "road": ["open road", "highway sunset", "country road", "road trip"],
    "river": ["river flowing", "waterfall", "river forest", "stream rocks"],
    "moon": ["moon night", "full moon", "moon clouds", "moonlight lake"],
    "summer": ["summer beach", "sunflower field", "summer sunset", "tropical beach"],
    "snow": ["snowy landscape", "snowfall", "winter forest", "snow mountain"],
    "flower": ["flowers field", "blooming flowers", "wildflowers", "rose closeup"],
    "flowers": ["flowers field", "blooming flowers", "wildflowers", "rose closeup"],
    "city": ["city night lights", "city skyline", "busy street", "city rooftop"],
    "mountain": ["mountain sunrise", "mountain peak", "mountain mist", "mountain lake"],
    "pray": ["candle dark", "praying hands", "church light", "candle flame"],
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
    """Collect every matching word's phrases, then pick one at random."""
    options = []
    for w in re.findall(r"[a-z']+", line.lower()):
        if w in RULES:
            val = RULES[w]
            options.extend(val if isinstance(val, list) else [val])
    return random.choice(options) if options else None


def embed_match(line):
    model, vecs = get_model()
    v = np.array(list(model.embed([line])))[0]
    v /= np.linalg.norm(v)
    return CONCEPTS[int(np.argmax(vecs @ v))]


# 3) PIXABAY (images + videos) with a cached POOL of results per query
_pool = {}  # key -> list of urls (up to 40)


async def pixabay_pool(query, style, media, refresh=False):
    key = f"{media}|{query}|{style}"
    if not refresh and _pool.get(key):
        return _pool[key]

    q = f"{query} {style}".strip()
    page = random.randint(1, 3) if refresh else 1
    urls = []
    try:
        async with httpx.AsyncClient(timeout=12) as c:
            if media == "video":
                r = await c.get(
                    "https://pixabay.com/api/videos/",
                    params={"key": PIXABAY_KEY, "q": q, "per_page": 40,
                            "page": page, "safesearch": "true"},
                )
                hits = r.json().get("hits", []) if r.status_code == 200 else []
                for h in hits:
                    v = h.get("videos", {})
                    u = (v.get("medium") or v.get("small") or v.get("tiny") or {}).get("url")
                    if u:
                        urls.append(u)
            else:
                r = await c.get(
                    "https://pixabay.com/api/",
                    params={"key": PIXABAY_KEY, "q": q, "image_type": "photo",
                            "orientation": "horizontal", "safesearch": "true",
                            "per_page": 40, "page": page},
                )
                hits = r.json().get("hits", []) if r.status_code == 200 else []
                urls = [h["largeImageURL"] for h in hits if h.get("largeImageURL")]
    except Exception:
        urls = []

    # If a random page was empty (e.g. page 3 doesn't exist), fall back to page 1 once
    if not urls and refresh and page != 1:
        return await pixabay_pool(query, style, media, refresh=False)

    if urls:  # never cache failures
        _pool[key] = urls
        return urls
    return _pool.get(key, [])


async def pick(query, style, media, used, refresh):
    pool = await pixabay_pool(query, style, media, refresh)
    if not pool:
        return None
    fresh = [u for u in pool if u not in used]
    url = random.choice(fresh or pool)
    used.add(url)
    return url


class Req(BaseModel):
    lines: list[str]
    style: str = ""
    media: str = "image"          # "image" or "video"
    default_query: str = "abstract light"
    refresh: bool = False         # set true when the user clicks "look for visuals"


def require_key(x_api_key: str = Header(default="")):
    if not APP_API_KEY:
        raise HTTPException(503, "APP_API_KEY is not configured on the server")
    if not hmac.compare_digest(x_api_key, APP_API_KEY):
        raise HTTPException(401, "Invalid or missing X-API-Key")


@app.post("/visuals")          # used by the built-in web page
async def visuals(req: Req):
    return await build(req)


@app.post("/v1/visuals", dependencies=[Depends(require_key)])   # for your other apps
async def v1_visuals(req: Req):
    if len(req.lines) > 200:
        raise HTTPException(400, "Max 200 lines per request")
    return await build(req)


@app.get("/v1/visual", dependencies=[Depends(require_key)])     # one line, simple GET
async def v1_visual(line: str, media: str = "image", style: str = "", refresh: bool = False):
    res = await build(Req(lines=[line], media=media, style=style, refresh=refresh))
    return res[0]


async def build(req: Req):
    out, prev_q, used = [], None, set()
    for line in req.lines:
        if not line.strip():
            q = prev_q or req.default_query
        else:
            q = rule_match(line)
            if not q and USE_EMBEDDINGS:
                try:
                    q = embed_match(line)
                except Exception:
                    q = None
            q = q or prev_q or req.default_query
        url = (await pick(q, req.style, req.media, used, req.refresh)
               or await pick(req.default_query, req.style, req.media, used, req.refresh))
        out.append({"line": line, "query": q, "url": url, "media": req.media})
        prev_q = q
    return out


@app.get("/health")
def health():
    return {"ok": True, "pixabay_key_set": bool(PIXABAY_KEY), "cached_queries": len(_pool)}


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")