# Dockerfile
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies (build tools are useful for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Streamlit config via env
ENV EKA_BOOTSTRAP=false
ENV EKA_DISABLE_RERANKER=true
ENV EKA_USE_RERANKER=false

# Default command: run the Streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
