# Docker 构建说明

本项目提供了统一的 Docker 配置，支持在构建时使用代理下载依赖。

## 文件说明

- `Dockerfile` - 统一 Dockerfile（同时支持 HTTP 和 WebSocket 服务）
- `docker-compose.yml` - Docker Compose 配置（包含 HTTP 和 WebSocket 两个服务）
- `.dockerignore` - Docker 构建忽略文件

## 代理配置

构建时默认使用代理：`http://192.168.1.32:7890`

如需修改代理地址，可以通过环境变量设置：

```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
export NO_PROXY=localhost,127.0.0.1
```

## 使用方法

### 1. 使用 docker-compose（推荐）

#### 启动所有服务（HTTP + WebSocket）

```bash
docker-compose up -d
```

#### 只启动 WebSocket 服务

```bash
docker-compose up -d sensevoice-wss
```

#### 只启动 HTTP API 服务

```bash
docker-compose up -d sensevoice-http
```

### 2. 使用 docker build

#### 构建镜像

```bash
docker build \
  --build-arg HTTP_PROXY=http://192.168.1.32:7890 \
  --build-arg HTTPS_PROXY=http://192.168.1.32:7890 \
  --build-arg NO_PROXY=localhost,127.0.0.1 \
  -t sensevoice:latest .
```

#### 运行容器

```bash
# HTTP API
docker run -d \
  -p 7007:7007 \
  -v ~/.cache/modelscope:/root/.cache/modelscope \
  -v $(pwd)/speaker:/app/speaker \
  -v $(pwd)/outputs:/app/outputs \
  --name sensevoice-http \
  sensevoice:latest \
  python server.py --port 7007

# WebSocket API
docker run -d \
  -p 27100:27100 \
  -v ~/.cache/modelscope:/root/.cache/modelscope \
  -v $(pwd)/speaker:/app/speaker \
  -v $(pwd)/outputs:/app/outputs \
  --name sensevoice-wss \
  sensevoice:latest \
  python server_wss_v3.py --port 27100
```

## 服务端口

- **HTTP API**: `http://localhost:7007`
- **WebSocket API**: `ws://localhost:27100/ws/transcribe`

## 卷挂载说明

- `~/.cache/modelscope` - 模型缓存目录，避免重复下载
- `./speaker` - 说话人音频输出目录
- `./outputs` - 模型微调输出目录

## GPU 支持

Docker Compose 配置已包含 GPU 支持。服务会自动检测并使用可用的 GPU 进行加速。

### 启用 GPU（需要 NVIDIA GPU）

确保已安装 NVIDIA Container Toolkit：

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

启动时服务会自动检测 GPU：

```bash
docker-compose up -d
```

查看日志确认 GPU 使用情况：

```bash
docker logs sensevoice-wss | grep "使用设备"
docker logs sensevoice-http | grep "使用设备"
```

### 不使用 GPU（纯 CPU 模式）

如果不使用 GPU，服务会自动回退到 CPU 模式，但性能会较慢。

## 注意事项

1. **代理仅用于构建时下载依赖**，运行时不使用代理
2. 首次启动会下载模型文件，需要较长时间
3. 确保挂载的目录有足够的磁盘空间
4. 建议至少 16GB 内存以确保正常运行
5. **使用 GPU 加速需要 NVIDIA GPU 和 NVIDIA Container Toolkit**

## 查看日志

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f sensevoice-wss
docker-compose logs -f sensevoice-http

# 或使用 docker logs
docker logs -f sensevoice-http
docker logs -f sensevoice-wss
```

## 停止服务

```bash
# 停止所有服务
docker-compose down

# 停止特定服务
docker-compose stop sensevoice-wss
docker-compose stop sensevoice-http

# 或使用 docker stop
docker stop sensevoice-http sensevoice-wss
docker rm sensevoice-http sensevoice-wss
```

