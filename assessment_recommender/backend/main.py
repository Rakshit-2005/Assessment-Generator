from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pathlib import Path
import os

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

# Avoid loading large local models on constrained platforms (e.g., Render free tier).
DISABLE_LOCAL_GEN = os.getenv('DISABLE_LOCAL_GEN') == '1' or os.getenv('RENDER') is not None
DISABLE_EMBEDDINGS = os.getenv('DISABLE_EMBEDDINGS') == '1' or os.getenv('RENDER') is not None
try:
    from hf_generator import generate_assessment
    HF_GEN_AVAILABLE = True
except Exception:
    HF_GEN_AVAILABLE = False
if DISABLE_LOCAL_GEN:
    LOCAL_GEN_AVAILABLE = False
else:
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
    app.state.recommender = Recommender(app.state.catalog, use_embeddings=not DISABLE_EMBEDDINGS)
    # session storage for simple chat flows
    app.state.chat_sessions = {}


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
    if req.use_embeddings and not DISABLE_EMBEDDINGS and not app.state.recommender.use_embeddings:
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


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/chat')
def chat(req: ChatRequest):
    """Lightweight chat endpoint that asks for JD when missing, returns recommendations,
    and can generate an assessment when the user asks "generate". Stores minimal
    session state in-memory (not persistent)."""
    sid = req.session_id or None
    # ensure session store exists
    if not hasattr(app.state, 'chat_sessions'):
        app.state.chat_sessions = {}
    sessions = app.state.chat_sessions
    msg = (req.message or '').strip()
    low = msg.lower()

    # simple JD detection heuristics
    is_jd = False
    if 'jd:' in low or 'job description' in low or 'job:' in low or len(msg.split()) > 12:
        is_jd = True

    # create session id when needed
    import uuid
    if not sid:
        sid = str(uuid.uuid4())

    if is_jd:
        jd = msg
        sessions[sid] = {'jd': jd}
        # return recommendations (top 5)
        try:
            results = app.state.recommender.recommend(jd, top_n=5)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Recommendation failed: {e}')
        return {'session_id': sid, 'reply': 'I found recommendations for that JD. Reply "generate" to create an assessment or ask to refine constraints.', 'recommendations': results}

    # if user asked to generate an assessment
    if low.startswith('generate') or 'generate' in low or low.startswith('create'):
        ctx = sessions.get(sid) or {}
        jd = ctx.get('jd')
        if not jd:
            return {'session_id': sid, 'reply': 'I need the job description (JD) before generating an assessment. Please paste the JD.'}
        # prefer local generator if available
        try:
            if LOCAL_GEN_AVAILABLE:
                out = generate_assessment_local(jd, model=None, max_tokens=256)
                return {'session_id': sid, 'reply': 'Generated assessment (local).', 'generated': out}
            if HF_GEN_AVAILABLE:
                out = generate_assessment(jd, model=None, max_length=512)
                return {'session_id': sid, 'reply': 'Generated assessment (hf).', 'generated': out}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Generation failed: {e}')
        return {'session_id': sid, 'reply': 'No generator available.'}

    # default: ask clarifying question
    return {'session_id': sid, 'reply': 'Please provide the job description (JD) or say "generate" if you already provided a JD.'}


if __name__ == '__main__':
    uvicorn.run('assessment_recommender.backend.main:app', host='127.0.0.1', port=8000, reload=False)
