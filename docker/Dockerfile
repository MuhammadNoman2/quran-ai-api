# CPU image. Torch-free by design: the inference path is CTranslate2 + ONNX
# Runtime, which is both faster on CPU and far smaller than a torch install.
#
#   docker build -f docker/Dockerfile -t quran-api .
#   docker run --rm -p 8000:8000 quran-api

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt


FROM python:3.12-slim AS runtime

# Audio decoding uses PyAV, which bundles its own FFmpeg, so no system codec
# packages are needed. curl is here only for the healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/models \
    QURAN_DATA_DIR=/app/data/quran \
    RECITATIONS_DIR=/data/recitations

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app ./app
COPY web ./web
COPY scripts ./scripts

# Build the Quran asset at image build time: it is deterministic, takes six
# seconds, and baking it means a container starts ready instead of spending that
# time on every boot.
RUN python scripts/prepare_quran_data.py

RUN useradd --create-home --uid 10001 quran \
 && mkdir -p /models /data/recitations \
 && chown -R quran:quran /models /data /app
USER quran

EXPOSE 8000

# Models download on first use into /models. Mount a volume there to avoid
# re-downloading roughly 220 MB every time a container is recreated.
VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fs http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
