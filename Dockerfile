# =============================================================================
# DeepAgentForce Dockerfile
# =============================================================================

# -----------------------------------------------------------------------------
# 阶段 1: 构建阶段
# -----------------------------------------------------------------------------
    FROM python:3.12-slim AS builder

    WORKDIR /app
    
    # 安装构建依赖
    RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
    
    # 创建虚拟环境
    RUN python -m venv /opt/venv
    ENV PATH="/opt/venv/bin:$PATH"
    
    # 设置 pip 镜像（阿里云加速）
    # https://mirrors.ustc.edu.cn/pypi/simple/
    RUN pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple/ && \
        pip config set global.trusted-host mirrors.ustc.edu.cn
    
    # 安装依赖
    COPY requirements.txt .
    RUN pip install --no-cache-dir --timeout=300 -r requirements.txt
    
    # -----------------------------------------------------------------------------
    # 阶段 2: 运行阶段
    # -----------------------------------------------------------------------------
    FROM python:3.12-slim
    
    WORKDIR /app
    
    # 安装运行时依赖
    RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
    
    # 从构建阶段复制虚拟环境
    COPY --from=builder /opt/venv /opt/venv
    ENV PATH="/opt/venv/bin:$PATH"
    
    # 复制应用代码（data/ 已通过 .dockerignore 排除）
    COPY . .
    
    # 运行时创建数据目录（不打入镜像，挂载用）
    RUN mkdir -p /app/data/uploads \
                 /app/data/outputs \
                 /app/data/rag_storage \
                 /app/data/history \
                 /app/data/skill \
                 /app/data/sessions
    
    ENV PYTHONUNBUFFERED=1
    ENV PYTHONDONTWRITEBYTECODE=1
    
    EXPOSE 8000
    
    HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
        CMD curl -f http://localhost:8000/ || exit 1
    
    CMD ["python", "main.py"]