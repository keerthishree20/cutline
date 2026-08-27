# Deploying Cutline

Two services. The API carries the model; the dashboard is static and reads from
it. Neither needs the 64 MB parquet — serving uses only `models/cutline.joblib`
and the generated files in `reports/`, all of which are committed.

## 1. API — Railway (or Render)

The repo has a `Dockerfile`, `railway.json` and `render.yaml`. Both platforms
detect the Dockerfile automatically.

**Railway**

1. <https://railway.app> → *New Project* → *Deploy from GitHub repo* →
   `keerthishree20/cutline`
2. It builds from `Dockerfile`. Health check is `/health`, already configured.
3. Once the frontend is deployed, set the variable:
   `ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app`
   Without it CORS blocks the dashboard, and the browser console will say so
   plainly.
4. Copy the generated URL.

**Render** is the same picture with `render.yaml`: *New* → *Blueprint* → pick
the repo, then set `ALLOWED_ORIGINS`.

> **Memory.** The image carries LightGBM, scikit-learn and SHAP, so it needs
> roughly 512 MB–1 GB of RAM. The SHAP explainer is built at startup, which
> costs a couple of seconds and some memory but keeps the first `/explain`
> fast. If a free tier OOMs, that is the reason — drop `/explain` before you
> drop the model.

## 2. Dashboard — Vercel

```bash
cd frontend
vercel login
vercel --prod
```

Set one environment variable in the Vercel project:

```
NEXT_PUBLIC_API_URL=https://<your-railway-app>.up.railway.app
```

It is baked in at build time, so **set it before deploying**, or redeploy after
changing it. See `frontend/.env.example`.

## 3. Verify, in this order

```bash
curl https://<api>/health          # {"ok": true, "trained_on": "ieee-cis"}
curl https://<api>/policy          # tau_block must NOT be 0.5
curl -s https://<api>/curve | head -c 200
```

`tau_block == 0.5` means the cost curve failed to load and the service quietly
reverted to the default threshold — which undoes the entire argument of the
project. `scripts/04_api_check.py` asserts against exactly this locally.

Then open the dashboard and press **Score 25 held-out transactions**. If the
queue renders but that button fails, it is CORS: `ALLOWED_ORIGINS` does not
match the frontend's real origin.

## Local, either way

```bash
docker build -t cutline-api . && docker run -p 8000:8000 cutline-api
cd frontend && npm run dev
```

## Regenerating what the API serves

The committed `reports/*.json` were generated from the trained bundle. If you
retrain, regenerate them or the dashboard will describe a model that no longer
exists:

```bash
.venv/bin/python scripts/02_model.py
.venv/bin/python scripts/03_cost_model.py
.venv/bin/python scripts/05_demo_queue.py
```
