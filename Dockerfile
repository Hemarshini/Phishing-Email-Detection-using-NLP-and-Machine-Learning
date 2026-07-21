# --- Phishing Email Detector: Production Container ---
# Base image: slim Python to keep the image small and fast to build/pull
FROM python:3.10-slim

# Prevents Python from writing .pyc files and buffers stdout (cleaner container logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (separate layer) so Docker caches this step
# and doesn't reinstall every dependency just because app code changed
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code
COPY . .

# Gradio's default port
EXPOSE 7860

# Basic healthcheck so orchestrators (Docker Compose, Kubernetes, etc.)
# can detect if the app has actually started serving, not just that the container is running
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" || exit 1

CMD ["python", "app.py"]
