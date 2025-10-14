# SenseVoice 实时语音识别 WebSocket API

基于 FunASR SenseVoice 模型的实时语音识别服务，支持说话人识别、音频降噪、ASR 错误修正等高级功能。

## 功能特性

### 核心功能
- ✅ **实时语音识别**：基于 SenseVoice Small 模型，支持流式 WebSocket 连接
- ✅ **说话人识别**：自动识别和区分多个说话人（基于声纹特征）
- ✅ **多语言支持**：支持中文、英文、粤语、日语、韩语等多种语言
- ✅ **情感识别**：识别语音中的情感（开心、悲伤、愤怒、中性等）
- ✅ **事件检测**：检测背景音乐、掌声、笑声、哭泣、咳嗽等音频事件

### 高级功能
- 🎯 **音频降噪**：支持多种降噪算法（频谱减法、维纳滤波、带通滤波、能量阈值法）
- 🎯 **ASR 错误修正**：自动修正同音字、专有名词、重复字符等常见识别错误
- 🎯 **噪音过滤**：智能过滤语气词、单字噪音等无效识别结果
- 🎯 **VAD 语音活动检测**：精确检测语音段起止点
- 🎯 **微调模型支持**：支持加载微调后的模型权重

## 项目结构

```
sensevoice/
├── server_wss_v3.py          # WebSocket 区分说话人
├── server_wss_v2.py          # WebSocket 区分说话人 只识别当前说话人
├── server_wss.py             # WebSocket 
├── server.py                 # HTTP API 
├── model.py                  # SenseVoice 模型定义（支持微调）
├── client_wss.html           # WebSocket 客户端测试页面
├── requirements.txt          # Python 依赖包
├── realtime_shell.py         # 实时命令行测试工具
├── README_realtime_shell.md  # 实时测试工具说明
└── outputs/                  # 模型微调输出目录
    └── model.pt.avg10        # 微调后的平均权重文件
```

## 环境要求

- Python 3.8+
- CUDA 支持（推荐用于 GPU 加速）
- 16GB+ RAM（模型加载需要）



### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型（可选）

模型会在首次运行时自动从 ModelScope 下载并缓存到 `~/.cache/modelscope/hub/`。

如果需要手动下载：
- **ASR 模型**：`iic/SenseVoiceSmall`
- **VAD 模型**：`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
- **说话人识别模型**：`iic/speech_campplus_sv_zh-cn_16k-common`

## 使用方法

### 启动 WebSocket 服务器

```bash
python server_wss_v3.py --port 27100
```

服务器将在 `ws://0.0.0.0:27100/ws/transcribe` 上监听 WebSocket 连接。

### 配置参数

可以通过环境变量或修改代码中的 `Config` 类来调整参数：

```python
class Config(BaseSettings):
    sd_thr: float = 0.25                    # 说话人分离相似度阈值
    chunk_size_ms: int = 300                # 分片大小（毫秒）
    sample_rate: int = 16000                # 采样率（Hz）
    bit_depth: int = 16                     # 位深
    channels: int = 1                       # 声道数
    avg_logprob_thr: float = -0.25          # 平均对数概率阈值
    min_confidence: float = 0.6             # 最小置信度阈值
    enable_error_correction: bool = True    # 启用错误修正
    filter_noise_words: bool = True         # 过滤噪音误识别词
    min_text_length: int = 2                # 最小有效文本长度
    enable_audio_denoise: bool = True       # 启用音频降噪
    denoise_method: str = "spectral"        # 降噪方法
    noise_reduce_strength: float = 0.5      # 降噪强度(0.0-1.0)
```

### WebSocket 客户端使用

#### 1. 使用浏览器测试页面

打开 `client_wss.html` 文件，修改 WebSocket 地址：

```javascript
ws = new WebSocket(`ws://localhost:27100/ws/transcribe?lang=zh&client_id=test_client`);
```

#### 2. WebSocket 连接参数

- `lang`：语言代码（`auto`、`zh`、`en`、`yue`、`ja`、`ko`）
- `client_id`：客户端唯一标识（可选，用于日志追踪）
- `sv`：旧版说话人验证参数（已弃用，现在自动进行说话人识别）

#### 3. 音频格式要求

- **采样率**：16000 Hz
- **位深**：16-bit
- **声道**：单声道（Mono）
- **格式**：PCM（原始音频数据）

#### 4. 响应格式

服务器返回 JSON 格式的响应：

**ASR 识别结果（code=0）**：
```json
{
  "code": 0,
  "info": "{\"text\": \"<|zh|>你好世界\", \"timestamp\": [...]}",
  "data": "你好世界",
  "client": "test_client",
  "speaker": "speaker_1"
}
```

**新说话人检测（code=4）**：
```json
{
  "code": 4,
  "info": "new_speaker_detected",
  "data": "speaker_2",
  "client": "test_client",
  "speaker": "speaker_2"
}
```

### Python 客户端示例

```python
import asyncio
import websockets
import numpy as np

async def transcribe_audio():
    uri = "ws://localhost:27100/ws/transcribe?lang=zh"
    
    async with websockets.connect(uri) as websocket:
        # 读取音频文件或录制音频
        audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
        
        # 发送音频数据
        await websocket.send(audio_data.tobytes())
        
        # 接收识别结果
        response = await websocket.recv()
        print(f"识别结果: {response}")

asyncio.run(transcribe_audio())
```

## 高级功能说明

### 1. 音频降噪

支持四种降噪方法：

- **spectral**（频谱减法）：适用于稳态噪音，效果较好
- **wiener**（维纳滤波）：适用于各种噪音类型
- **bandpass**（带通滤波）：保留语音频率范围（300-3400Hz）
- **energy**（能量阈值法）：基于短时能量的简单降噪

配置示例：
```python
config.enable_audio_denoise = True
config.denoise_method = "spectral"
config.noise_reduce_strength = 0.5  # 0.0-1.0
```

### 2. ASR 错误修正

自动修正三种常见错误：

- **替换错误**：同音字、专有名词（如"小右" → "小佑"）
- **插入错误**：重复字符、多余语气词
- **删除错误**：缺失的标点符号

可在 `ASRErrorCorrector` 类中自定义修正规则：

```python
self.homophone_dict = {
    "小右": "小佑",
    "在见": "再见",
    # 添加更多规则...
}
```

### 3. 噪音过滤

智能过滤无效识别结果：

- 纯语气词（"嗯"、"啊"、"呃"等）
- 单个字符 + 低置信度
- 纯标点符号
- 过短文本（可配置最小长度）

### 4. 说话人识别

基于声纹特征的自动说话人识别：

- 自动注册新说话人（speaker_1, speaker_2, ...）
- 实时计算声纹相似度
- 可调节相似度阈值（`sd_thr`）

## 微调模型

项目支持加载微调后的 SenseVoice 模型权重：

```python
model_asr = AutoModel(
    model=asr_model_cache,
    trust_remote_code=True,
    remote_code="./model.py",
    device="cuda:0",
    initial_model="./outputs/model.pt.avg10"  # 微调权重路径
)
```

将微调后的权重文件放在 `outputs/` 目录下即可。

## 性能优化

### 1. GPU 加速

确保安装了 CUDA 版本的 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 批处理大小

调整 `batch_size_s` 参数以平衡速度和内存使用：

```python
result = model_asr.generate(
    input=audio,
    batch_size_s=60,  # 增大以提高吞吐量
)
```

### 3. VAD 参数调优

调整 VAD 模型参数以优化语音段检测：

```python
model_vad = AutoModel(
    model=vad_model_cache,
    max_end_silence_time=500,  # 最大静音时长（毫秒）
    # speech_noise_thres=0.6,  # 语音噪音阈值
)
```

## 常见问题

### 1. 模型下载失败

**问题**：首次运行时模型下载超时或失败。

**解决方案**：
- 手动从 ModelScope 下载模型到 `~/.cache/modelscope/hub/`
- 使用国内镜像或代理
- 检查网络连接

### 2. 内存不足

**问题**：加载模型时出现 OOM 错误。

**解决方案**：
- 使用更小的模型（如 SenseVoiceSmall）
- 减少 `batch_size_s` 参数
- 增加系统内存或使用 GPU

### 3. 识别结果不准确

**问题**：识别文本错误率高。

**解决方案**：
- 启用音频降噪（`enable_audio_denoise=True`）
- 启用错误修正（`enable_error_correction=True`）
- 调整置信度阈值（`min_confidence`）
- 使用微调后的模型

### 4. 说话人识别不准确

**问题**：同一说话人被识别为多人，或不同说话人被识别为同一人。

**解决方案**：
- 调整相似度阈值（`sd_thr`）：
  - 降低阈值（如 0.2）：更容易识别为新说话人
  - 提高阈值（如 0.3）：更倾向于归为已知说话人
- 确保音频质量良好
- 确保每个语音段有足够的音频数据

## 日志说明

项目使用 `loguru` 进行日志记录，日志级别：

- **DEBUG**：详细的调试信息（音频处理、降噪、修正等）
- **INFO**：重要事件（连接建立、说话人识别、识别结果等）
- **ERROR**：错误信息（异常、失败等）

日志格式：
```
2025-10-14 16:24:42 [INFO] server_wss_v3.py:938 - [client] 建立连接，client_id=test_client
```

## 技术栈

- **FastAPI**：Web 框架
- **FunASR**：阿里达摩院语音识别框架
- **SenseVoice**：多语言语音识别模型
- **ModelScope**：模型下载和管理
- **PyTorch**：深度学习框架
- **NumPy/SciPy**：音频信号处理
- **Loguru**：日志记录

## 许可证

本项目基于开源协议发布，具体许可证请参考项目根目录的 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过 Issue 联系。

---

**注意**：本项目依赖的 SenseVoice 模型由阿里巴巴达摩院开发，使用前请遵守相关模型的使用协议。
