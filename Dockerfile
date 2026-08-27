# The scoring service. Deliberately does NOT carry the 64MB parquet: serving
# reads the model bundle and two generated JSON files, nothing else.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp is LightGBM's OpenMP runtime — without it the import fails at
# startup with an error that says nothing useful about the cause.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# kaggle and matplotlib are build-time tools, not serving dependencies.
RUN grep -viE '^(kaggle|matplotlib)==' requirements.txt > runtime.txt \
 && pip install --no-cache-dir -r runtime.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/cutline.joblib ./models/cutline.joblib
COPY reports/cost_curve.json reports/demo_queue.json reports/sample_upload.csv ./reports/

EXPOSE 8000
# Hosts inject $PORT; default to 8000 so `docker run -p 8000:8000` also works.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
