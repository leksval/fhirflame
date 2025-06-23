# FhirFlame Medical AI Platform
# Professional containerization for Gradio UI and A2A API deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including PDF processing tools
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY static/ ./static/
COPY app.py .
COPY frontend_ui.py .
COPY database.py .
COPY fhirflame_logo.svg .
COPY fhirflame_logo_450x150.svg .
COPY index.html .

# Copy environment file if it exists
COPY .env* ./

# Create logs directory
RUN mkdir -p logs test_results

# Set Python path for proper imports
ENV PYTHONPATH=/app

# Expose ports for both Gradio UI (7860) and A2A API (8000)
EXPOSE 7860 8000

# Health check for both possible services
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860 || curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden in docker-compose)
CMD ["python", "app.py"]