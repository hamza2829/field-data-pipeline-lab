# simulator image. Build context: repository root.
#   docker build -f docker/simulator.Dockerfile -t field-data-simulator .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir "numpy>=1.26"
COPY simulator/ simulator/
COPY pipeline.py .
RUN useradd --create-home --uid 10001 fieldlab && mkdir /data && chown fieldlab /data
USER fieldlab
VOLUME ["/data"]
CMD ["python", "pipeline.py", "generate", "--work", "/data"]
