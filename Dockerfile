# DeepAgentForce Dockerfile - Multi-stage build for optimal image size
# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Configure Debian mirror for faster builds
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated Python environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Configure PyPI mirror for faster installation
RUN pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple/ && \
    pip config set global.trusted-host mirrors.ustc.edu.cn

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Stage 2: Runtime image
FROM python:3.12-slim

WORKDIR /app

# Configure Debian mirror
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    SQLITE_DB_PATH=data/deepagentforce.db \
    DEEPAGENTFORCE_DATA_DIR=/app/data

# Copy application code
# Note: Runtime data (SQLite, uploads, etc.) should be mounted via volumes
COPY . .

# Create data directories for persistence
RUN mkdir -p /app/data/uploads \
             /app/data/outputs \
             /app/data/rag_storage \
             /app/data/history \
             /app/data/skill \
             /app/data/sessions

# Expose API port
EXPOSE 8000

# Health check for orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start application
CMD ["python", "main.py"]
