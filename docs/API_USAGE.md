# API 使用说明

## 端点列表

### 1. POST /transcribe - 音频转写

转写音频文件或音频 URL。

**请求参数：**
- `file` (可选): 上传的音频文件
- `url` (可选): 音频文件的 URL

**示例：**

```bash
# 上传音频文件
curl -X POST "http://localhost:7007/transcribe" \
  -F "file=@audio.wav"

# 使用音频 URL
curl -X POST "http://localhost:7007/transcribe" \
  -F "url=https://example.com/audio.mp3"
```

**响应格式：**
```json
{
  "code": 0,
  "msg": "success, transcription time: 2.34 seconds",
  "data": "转写结果文本"
}
```

### 2. POST /transcribe_video - 视频转写

转写视频中的音频，支持各种视频平台（包括抖音）。

**请求参数：**
- `video_url` (必需): 视频链接

**示例：**

```bash
# 转写抖音视频
curl -X POST "http://localhost:7007/transcribe_video" \
  -F "video_url=https://www.douyin.com/video/123456789"

# 转写其他平台视频
curl -X POST "http://localhost:7007/transcribe_video" \
  -F "video_url=https://www.youtube.com/watch?v=example"
```

**响应格式：**
```json
{
  "code": 0,
  "msg": "success, transcription time: 5.67 seconds",
  "data": "转写结果文本"
}
```

## 特殊功能

### 抖音链接支持

当检测到抖音链接时，系统会自动添加必要的 headers 来跳过防盗链：

- `douyin.com`
- `douyin.io`
- `iesdouyin.com`

### 自动音频提取

使用 `httpx` 和 `ffmpeg` 自动：
1. 使用 HTTP 请求下载视频
2. 使用 ffmpeg 提取音频
3. 转换为 WAV 格式（16kHz, 单声道, PCM 16-bit）
4. 进行转写

## 支持的文件格式

### 音频格式
- WAV
- MP3
- WebM
- OGG
- 其他 torchaudio 支持的格式

### 视频格式
- MP4
- WebM
- FLV
- 其他 yt-dlp 支持的格式

## 注意事项

1. 视频下载和转写需要较长时间，请耐心等待
2. 临时文件会在处理后自动清理
3. 确保服务器有足够的磁盘空间和内存
4. 抖音链接需要稳定的网络连接

## Python 客户端示例

```python
import requests

# 转写音频文件
files = {'file': open('audio.wav', 'rb')}
response = requests.post('http://localhost:7007/transcribe', files=files)
print(response.json())

# 转写视频
data = {'video_url': 'https://www.douyin.com/video/123456789'}
response = requests.post('http://localhost:7007/transcribe_video', data=data)
print(response.json())
```

## JavaScript 客户端示例

```javascript
// 转写音频文件
const formData = new FormData();
formData.append('file', audioFile);

fetch('http://localhost:7007/transcribe', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));

// 转写视频
const formData = new FormData();
formData.append('video_url', 'https://www.douyin.com/video/123456789');

fetch('http://localhost:7007/transcribe_video', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

