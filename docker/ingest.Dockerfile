# ingest image. Build context: repository root.
#   docker build -f docker/ingest.Dockerfile -t field-data-ingest .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY simulator/ simulator/
COPY ingest/ ingest/
COPY dataops/ dataops/
COPY pipeline.py .
RUN useradd --create-home --uid 10001 fieldlab && mkdir /data && chown fieldlab /data
USER fieldlab
VOLUME ["/data"]
CMD ["sh", "-c", "python pipeline.py ingest --work /data && python pipeline.py validate --work /data && python pipeline.py convert --work /data"]
