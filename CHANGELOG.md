# 更新日志

## 2025-10-26

### 新增功能
- ✅ 添加视频转写端点 `/transcribe_video`，支持下载视频并提取音频进行转写
- ✅ 抖音视频下载支持，自动添加防盗链 headers
- ✅ 详细的性能监控，统计每个步骤的耗时
- ✅ GPU 自动检测和使用

### 改进
- 🎯 HTTP 和 WebSocket 服务都支持 GPU 加速
- 🎯 优化日志输出，添加详细的性能指标
- 🎯 文档整理到 `docs/` 目录
- 🎯 Docker 配置简化，移除多余的服务版本

### 移除
- ❌ 删除 `server_wss.py`（旧版 WebSocket）
- ❌ 删除 `server_wss_v2.py`（旧版 WebSocket v2）
- ❌ 删除 `DOCKER.md`、`API_USAGE.md`、`DOUYIN_DOWNLOAD.md`（已移动到 docs/）

### 文件变更
- `server.py` - 添加 GPU 支持、视频转写、性能监控
- `server_wss_v3.py` - 添加 GPU 自动检测
- `requirements.txt` - 添加 httpx、python-multipart
- `Dockerfile` - 添加 ffmpeg 支持
- `docker-compose.yml` - 添加 GPU 支持、简化配置
- `README.md` - 更新项目结构说明

### 依赖变化
- 新增：`httpx`、`python-multipart`
- 移除：`yt-dlp`、`ffmpeg-python`

### 性能优化
- 添加详细的性能监控日志
- 统计下载、提取、处理、转写各步骤耗时
- 显示转写速度（实时倍数）

### GPU 支持
- 自动检测 CUDA 是否可用
- 优先使用 GPU，无 GPU 时回退到 CPU
- Docker 配置中添加 GPU 运行时支持

