# Lyric Visuals

Rule-based + MiniLM embedding fallback -> Pixabay images or videos. FastAPI backend serves the web app.

## Run locally
    pip install -r requirements.txt
    export PIXABAY_KEY=your_key
    uvicorn main:app --reload
Open http://localhost:8000

## Push to GitHub
    git init && git add . && git commit -m "Lyric visuals app"
    git branch -M main
    git remote add origin https://github.com/YOUR_USER/lyric-visuals.git
    git push -u origin main

## Deploy on Render
New Web Service -> connect repo (render.yaml is picked up). Set env var PIXABAY_KEY.
Set USE_EMBEDDINGS=0 if you hit the 512 MB memory limit (rules only).

## API for other apps
Set env var APP_API_KEY on Render (any long random string). Send it as header X-API-Key.

    curl -X POST https://YOUR-APP.onrender.com/v1/visuals \
      -H "X-API-Key: YOUR_KEY" -H "Content-Type: application/json" \
      -d '{"lines":["I cry in the rain"],"media":"video","style":"cinematic"}'

    curl "https://YOUR-APP.onrender.com/v1/visual?line=open%20road&media=image" -H "X-API-Key: YOUR_KEY"

Interactive docs: /docs
