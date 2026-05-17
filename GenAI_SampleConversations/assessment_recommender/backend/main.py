from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pathlib import Path

# load .env located at the assessment_recommender root (one level up from backend)
here = Path(__file__).resolve().parent.parent
load_dotenv(here / '.env')
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

import sys
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure backend folder is on path so local imports work when running with --app-dir
sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog_loader import load_catalog
from recommender import Recommender
try:
    from hf_generator import generate_assessment
    HF_GEN_AVAILABLE = True
except Exception:
    HF_GEN_AVAILABLE = False
try:
    from local_generator import generate_assessment_local
    LOCAL_GEN_AVAILABLE = True
except Exception:
    LOCAL_GEN_AVAILABLE = False

app = FastAPI(title='SHL Assessment Recommender')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    jd: str
    top: Optional[int] = 8
    use_embeddings: Optional[bool] = True


class GenerateRequest(BaseModel):
    jd: str
    model: Optional[str] = None
    max_tokens: Optional[int] = 512


@app.on_event('startup')
def startup_event():
    # load catalog once
    app.state.catalog = load_catalog()
    app.state.recommender = Recommender(app.state.catalog, use_embeddings=True)


# Serve frontend static files from the sibling `frontend` folder (if present)
frontend_dir = Path(__file__).resolve().parent.parent / 'frontend'
if frontend_dir.exists():
    # serve static assets at /static and expose index.html at /
    app.mount('/static', StaticFiles(directory=str(frontend_dir), html=True), name='static')

    @app.get('/')
    def _root():
        return FileResponse(str(frontend_dir / 'index.html'))


@app.post('/recommend')
def recommend(req: RecommendRequest):
    if not req.jd or not req.jd.strip():
        raise HTTPException(status_code=400, detail='JD text required')
    # optionally rebuild recommender with/without embeddings
    if req.use_embeddings and not app.state.recommender.use_embeddings:
        # try to enable embeddings
        app.state.recommender = Recommender(app.state.catalog, use_embeddings=True)
    results = app.state.recommender.recommend(req.jd, top_n=req.top or 8)
    return {'count': len(results), 'results': results}


@app.post('/generate')
def generate(req: GenerateRequest):
    if not req.jd or not req.jd.strip():
        raise HTTPException(status_code=400, detail='JD text required')
    # prefer local generator if available
    if LOCAL_GEN_AVAILABLE:
        try:
            out = generate_assessment_local(req.jd, model=req.model, max_tokens=req.max_tokens or 256)
            # if model returned raw text or invalid JSON, fallback to simple generator
            if isinstance(out, dict) and 'raw' in out:
                from local_generator import simple_assessment_from_jd
                fallback = simple_assessment_from_jd(req.jd)
                return {'generated': fallback, 'source': 'fallback'}
            return {'generated': out, 'source': 'local'}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Local generation failed: {e}')
    if HF_GEN_AVAILABLE:
        try:
            out = generate_assessment(req.jd, model=req.model, max_length=req.max_tokens or 512)
            return {'generated': out, 'source': 'hf'}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'HF generation failed: {e}')
    raise HTTPException(status_code=500, detail='No generator available on server')


if __name__ == '__main__':
    uvicorn.run('assessment_recommender.backend.main:app', host='127.0.0.1', port=8000, reload=False)
