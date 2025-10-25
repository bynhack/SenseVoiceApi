# 抖音视频下载说明

## 实现方式

代码使用简单的 HTTP 请求直接下载视频文件，然后使用 ffmpeg 提取音频。

## 抖音特殊处理

当检测到抖音链接时，会自动添加以下 headers：

```python
{
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.douyin.com/',
    'Accept': '*/*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Sec-Fetch-Dest': 'video',
    'Sec-Fetch-Mode': 'no-cors',
    'Sec-Fetch-Site': 'cross-site',
}
```

> 这些 headers 与 Node.js 项目中的实现保持一致，确保能够绕过抖音的防盗链。

## 使用示例

```bash
# 转写抖音视频
curl -X POST "http://localhost:7007/transcribe_video" \
  -F "video_url=https://example.com/video.mp4"

# 查看日志
docker logs -f sensevoice-http
```

## 支持格式

- 抖音视频直链 URL
- 其他视频平台的直链 URL
- 任意 HTTP/HTTPS 可访问的视频文件

## 工作流程

1. 接收视频 URL
2. 检测是否为抖音链接
3. 如果是抖音链接，添加特殊 headers
4. 使用 httpx 下载视频
5. 使用 ffmpeg 提取音频（16kHz, 单声道, PCM 16-bit）
6. 返回音频字节数据

## 注意事项

- 需要提供直链 URL（可以直接下载的视频文件链接）
- Docker 镜像已包含 ffmpeg
- 如果视频链接需要身份验证，可能需要提供 Cookie

