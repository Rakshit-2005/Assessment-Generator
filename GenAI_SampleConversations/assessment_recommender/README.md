# SHL Assessment Recommender

SHL Assessment Recommender

This project contains a minimal full-stack application that ingests the SHL product catalog and recommends assessments based on a job description (JD).

Components
- `backend/` - FastAPI app exposing `/recommend` to score and return matching assessments.
- `frontend/` - Minimal static HTML page that calls the backend and displays results.

Quickstart (python 3.10+)

1. Create and activate a virtualenv, then install dependencies:

```bash
python -m venv .venv
.
.venv\Scripts\activate   # Windows PowerShell
pip install -r GenAI_SampleConversations/assessment_recommender/requirements.txt
```

2. Run the backend API:

```bash
python -m assessment_recommender.backend.main
```

3. Open the frontend in a browser:

Open `GenAI_SampleConversations/assessment_recommender/frontend/index.html` and click Recommend (backend must be running at `http://127.0.0.1:8000`).

Notes & next steps
- The recommender uses sentence-transformers embeddings when available (model `all-MiniLM-L6-v2`). If embeddings are not installed or fail to load, it falls back to TF-IDF and a simple keyword heuristic.
- For production: add authentication, caching, better error handling, language/level filters, and supervised mappings from your example JDs (C1–C10).

Generation (LLM)
-----------------
- This project includes a `/generate` endpoint that returns a JSON assessment for a given JD. The backend prefers a local `transformers`-based generator when available and falls back to a deterministic senior-level template if the model doesn't produce valid JSON.
- To enable local generation a few packages are required: `transformers`, `torch`, and `accelerate`. These are added to `requirements.txt`.
- The local model can be overridden with the `LOCAL_GEN_MODEL` environment variable (defaults to `google/flan-t5-small`). Larger models may require significant disk space and RAM.

Environment / secrets
---------------------
- The repository includes `assessment_recommender/.env` but the `HF_API_TOKEN` value is intentionally blank to avoid committing secrets. Set your HF token in your shell when needed instead of placing it in the file.

	PowerShell:
	```powershell
	$env:HF_API_TOKEN="hf_your_token_here"
	```

	Bash:
	```bash
	export HF_API_TOKEN="hf_your_token_here"
	```

Running the server (recommended)
-------------------------------
Use `uvicorn` to run the FastAPI app from the repository root:

```powershell
& ".venv/Scripts/python.exe" -m uvicorn assessment_recommender.backend.main:app --host 127.0.0.1 --port 8000 --reload --app-dir GenAI_SampleConversations
```

Frontend
--------
- The frontend (`frontend/index.html`) includes a "Generate Assessment" button which calls `/generate` and displays the returned JSON. The response payload contains a `source` field indicating whether the result came from the local LLM or the fallback template.

Security note
-------------
- Never commit API keys to the repository. If you accidentally committed a key, rotate it immediately and remove it from history. I can help remove keys from git history if needed.

