FROM python:3.11-slim

# HF Spaces expects port 7860
EXPOSE 7860

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# HF Spaces runs as non-root
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

CMD ["python", "server.py"]
