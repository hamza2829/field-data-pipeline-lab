# mlops image. Build context: repository root.
#   docker build -f docker/mlops.Dockerfile -t field-data-mlops .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY simulator/ simulator/
COPY ingest/ ingest/
COPY dataops/ dataops/
COPY mlops/ mlops/
COPY pipeline.py .
RUN useradd --create-home --uid 10001 fieldlab && mkdir /data && chown fieldlab /data
USER fieldlab
VOLUME ["/data"]
CMD ["sh", "-c", "python pipeline.py version --work /data && python pipeline.py evaluate --work /data"]
