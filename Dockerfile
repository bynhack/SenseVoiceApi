# 使用 Python 3.10 镜像（支持 CUDA）
FROM python:3.10-slim



# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（使用代理）
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p speaker outputs

# 暴露端口
EXPOSE 7007 27100

# 默认启动 WebSocket 服务
CMD ["python", "server_wss_v3.py", "--port", "27100"]

