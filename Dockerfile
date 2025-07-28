# Multi-stage build for optimized performance
FROM python:3.9-slim-bullseye as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
    PyPDF2==3.0.0 \
    spacy==3.7.4 \
    numpy==1.26.4 \
    scikit-learn==1.4.2 \
    rank_bm25==0.2.2 \
    sentence-transformers==2.4.0 \
    pdfminer.six==20221105

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Production stage
FROM python:3.9-slim-bullseye

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY run_analysis.py /app/

# Create documents directory
RUN mkdir -p /app/documents

# Set Python optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TOKENIZERS_PARALLELISM=false

# Default command
CMD ["python", "run_analysis.py"]