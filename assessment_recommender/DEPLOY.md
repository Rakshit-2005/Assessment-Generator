Render and Heroku deployment guide

Overview
This document explains how to deploy the `assessment_recommender` service (FastAPI) to Render or Heroku, plus a simple GitHub Actions CI workflow that installs dependencies and checks the `/health` endpoint.

Common notes
- Do NOT commit secrets to the repository. Set `HF_API_TOKEN` as an environment/config var in the host dashboard.
- The repo provides `Dockerfile` and `Procfile` in this folder.
- Endpoints available: GET `/health`, POST `/chat`, POST `/recommend`, POST `/generate`.

Render (web service)
1. Push your repo to GitHub.
2. Create a new Web Service on Render and connect your GitHub repo.
3. Set the Root directory to `GenAI_SampleConversations/assessment_recommender` if you are deploying that subfolder.
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
6. Environment: Add `HF_API_TOKEN` with your Hugging Face key. Optionally set `PYTHONUNBUFFERED=1`.
7. Deploy. Render will show the public URL for your service.

Heroku (container or Git)
Option A — Git deploy (recommended for quick setup):
1. Install the Heroku CLI and log in: `heroku login`.
2. Create an app: `heroku create your-app-name`.
3. Set config var: `heroku config:set HF_API_TOKEN=your_hf_key`
4. Ensure `Procfile` exists in `assessment_recommender` (it does). If deploying the whole repo, put `Procfile` at repo root or set buildpack.
5. Push to Heroku:
   - If your app uses a subdirectory, consider using `heroku buildpacks:set` or a monorepo buildpack, or push from the `assessment_recommender` subfolder via `git subtree`.
6. Heroku will run the `web` command from the `Procfile`. The service will be available at `https://<your-app>.herokuapp.com`.

Option B — Container deploy:
1. Build locally: `docker build -t my-app .` (run from `assessment_recommender`).
2. Push to Heroku Container Registry and release (see Heroku docs). Set `HF_API_TOKEN` in Heroku config.

Cold-start considerations
- If you use local `transformers` model loading, the first request may incur model download/load time (cold-start). Consider using HF Inference API for faster startup.

Troubleshooting
- 500 errors on generation: verify `HF_API_TOKEN` is set or that the `transformers` dependencies are installed if using local generation.
- Model download failures: ensure sufficient memory and disk on the host for model artifacts.

CI — GitHub Actions (basic)
- See `.github/workflows/ci.yml` for a simple workflow that installs Python deps and checks the `/health` endpoint after launching the server.

Security reminder
- Remove any committed secrets and rotate the Hugging Face key if it was ever pushed to a public repo.
