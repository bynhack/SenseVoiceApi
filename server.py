from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic import BaseModel
from funasr import AutoModel
import asyncio
import numpy as np
import torch
import torchaudio
import io
import soundfile as sf
import argparse
import uvicorn
import time
import logging
import sys
import httpx
import os
import tempfile
import re
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


emo_dict = {
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
}

event_dict = {
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|Cry|>": "😭",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "🤧",
}

emoji_dict = {
	"<|nospeech|><|Event_UNK|>": "❓",
	"<|zh|>": "",
	"<|en|>": "",
	"<|yue|>": "",
	"<|ja|>": "",
	"<|ko|>": "",
	"<|nospeech|>": "",
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
	"<|Cry|>": "😭",
	"<|EMO_UNKNOWN|>": "",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "😷",
	"<|Sing|>": "",
	"<|Speech_Noise|>": "",
	"<|withitn|>": "",
	"<|woitn|>": "",
	"<|GBG|>": "",
	"<|Event_UNK|>": "",
}

lang_dict =  {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}

def format_str(s):
	for sptk in emoji_dict:
		s = s.replace(sptk, emoji_dict[sptk])
	return s


def format_str_v2(s):
	sptk_dict = {}
	for sptk in emoji_dict:
		sptk_dict[sptk] = s.count(sptk)
		s = s.replace(sptk, "")
	emo = "<|NEUTRAL|>"
	for e in emo_dict:
		if sptk_dict[e] > sptk_dict[emo]:
			emo = e
	for e in event_dict:
		if sptk_dict[e] > 0:
			s = event_dict[e] + s
	s = s + emo_dict[emo]

	for emoji in emo_set.union(event_set):
		s = s.replace(" " + emoji, emoji)
		s = s.replace(emoji + " ", emoji)
	return s.strip()

def format_str_v3(s):
	def get_emo(s):
		return s[-1] if s[-1] in emo_set else None
	def get_event(s):
		return s[0] if s[0] in event_set else None

	s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
	for lang in lang_dict:
		s = s.replace(lang, "<|lang|>")
	s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
	new_s = " " + s_list[0]
	cur_ent_event = get_event(new_s)
	for i in range(1, len(s_list)):
		if len(s_list[i]) == 0:
			continue
		if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
			s_list[i] = s_list[i][1:]
		#else:
		cur_ent_event = get_event(s_list[i])
		if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
			new_s = new_s[:-1]
		new_s += s_list[i].strip().lstrip()
	new_s = new_s.replace("The.", " ")
	return new_s.strip()


app = FastAPI()

# 设置跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，可以根据需要指定特定的域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有请求头
)

# ============ 模型加载配置 ============
# 优先使用本地缓存,避免网络问题

# ASR 模型
asr_model_cache = os.path.expanduser('~/.cache/modelscope/hub/iic/SenseVoiceSmall')
if not os.path.exists(asr_model_cache):
    asr_model_cache = "iic/SenseVoiceSmall"
    logger.warning(f"本地缓存不存在,将尝试从 ModelScope 下载: {asr_model_cache}")
else:
    logger.info(f"使用本地缓存的 ASR 模型: {asr_model_cache}")

# Initialize the model outside the endpoint to avoid reloading it for each request
# 检查是否有可用的 GPU
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"使用设备: {device}")

model = AutoModel(
    model=asr_model_cache,
#   vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
#   vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
    device=device,
)



def transcribe_with_timing(*args, **kwargs):
    start_time = time.time()
    logger.info(f"[ASR] 开始转写，使用设备: {device}")
    result = model.generate(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"[ASR] 转写完成，耗时: {elapsed_time:.2f} 秒")
    return result, elapsed_time


@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error("Exception occurred", exc_info=True)
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        message = exc.detail
        data = ""
    elif isinstance(exc, RequestValidationError):
        status_code = HTTP_422_UNPROCESSABLE_ENTITY
        message = "Validation error: " + str(exc.errors())
        data = ""
    else:
        status_code = 500
        message = "Internal server error: " + str(exc)
        data = ""
    
    return JSONResponse(
        status_code=status_code,
        content=TranscriptionResponse(
            code=status_code,
            msg=message,
            data=data
        ).model_dump()
    )

# Define the response model
class TranscriptionResponse(BaseModel):
    code: int
    msg: str
    data: str

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(None),
    url: str = Form(None),
):
    """
    转写音频文件或音频 URL
    支持本地上传文件或提供音频 URL
    """
    try:
        start_total = time.time()
        logger.info(f"[音频转写] 收到请求 - file: {file.filename if file else None}, url: {url}")
        
        # 统一从本地上传文件或 URL 下载获取字节与类型
        file_content: bytes = None
        content_type: str = ""
        filename_hint: str = ""

        download_start = time.time()
        if file is not None:
            # Read the file content and reset the file pointer
            file.file.seek(0)
            file_content = await file.read()
            content_type = file.content_type or ""
            filename_hint = file.filename or ""
            logger.info(f"[音频转写] 上传文件大小: {len(file_content) / 1024:.2f} KB")

        if (file_content is None or len(file_content) == 0) and url:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to download url, status={resp.status_code}")
                file_content = resp.content
                content_type = resp.headers.get("content-type", "")
                filename_hint = url.split("?")[0].split("#")[0]
                logger.info(f"[音频转写] 下载音频大小: {len(file_content) / 1024:.2f} KB")
        
        download_time = time.time() - download_start

        if file_content is None or len(file_content) == 0:
            raise HTTPException(status_code=400, detail="No audio provided. Provide file or url")

        # 根据 content-type 或文件名后缀判断解析方式
        audio_process_start = time.time()
        ct = (content_type or "").lower()
        lower_name = (filename_hint or "").lower()

        try:
            if ct.startswith('audio/wav') or lower_name.endswith('.wav'):
                # 使用 soundfile 读取，确保是 int16 路径
                bio = io.BytesIO(file_content)
                input_wav, sr = sf.read(bio, dtype=np.int16)
                bit_depth = sf.info(io.BytesIO(file_content)).subtype
                is16 = True if bit_depth == 'PCM_16' else False
            elif ct.startswith('audio/webm') or lower_name.endswith('.webm'):
                # 使用 torchaudio 读取 webm
                input_wav, sr = torchaudio.load(io.BytesIO(file_content))
                # torchaudio 返回 tensor，通常为 float32
                is16 = False
                input_wav = input_wav.squeeze().numpy()
            else:
                # 尝试通用解码（如 mp3/ogg 等），依赖系统后端
                input_wav, sr = torchaudio.load(io.BytesIO(file_content))
                is16 = False
                input_wav = input_wav.squeeze().numpy()
        except Exception:
            raise HTTPException(status_code=400, detail="Unsupported or invalid audio format")

        
        #filename = (file.filename if file.filename else "test") + "." + suffix
        #with open(filename, "wb") as f:
            #f.write(file_content)
            
        
        if len(input_wav.shape) > 1:
            input_wav = input_wav.mean(-1)

        if is16: # indicate the audio data is not float, so convert it to float32
            input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max

        if sr != 16000:
            logger.info(f"[音频转写] 采样率 {sr} Hz，重采样到 16000 Hz")
            resampler = torchaudio.transforms.Resample(sr, 16000)
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
        
        audio_process_time = time.time() - audio_process_start
        audio_duration = len(input_wav) / 16000
        logger.info(f"[性能监控] 音频处理耗时: {audio_process_time:.2f} 秒，音频时长: {audio_duration:.2f} 秒")
                
        # 执行转写
        transcribe_start = time.time()
        async def generate_text():
            return await asyncio.to_thread(transcribe_with_timing, 
                                           input=input_wav,
                                           cache={},
                                           language="auto",
                                           use_itn=True,
                                           batch_size=64)

        # Run the asynchronous function
        # Run the asynchronous function
        resp, elapsed_time = await generate_text()
        transcribe_time = time.time() - transcribe_start
        
        text = format_str_v3(resp[0]["text"])
        logger.info(f'[音频转写] 转写结果: {text}')
        
        # 总耗时统计
        total_time = time.time() - start_total
        logger.info(f"[性能监控] ════════════════════════════════════════")
        logger.info(f"[性能监控] 总计耗时: {total_time:.2f} 秒")
        logger.info(f"[性能监控]   - 下载: {download_time:.2f} 秒 ({download_time/total_time*100:.1f}%)")
        logger.info(f"[性能监控]   - 音频处理: {audio_process_time:.2f} 秒 ({audio_process_time/total_time*100:.1f}%)")
        logger.info(f"[性能监控]   - ASR 转写: {transcribe_time:.2f} 秒 ({transcribe_time/total_time*100:.1f}%)")
        logger.info(f"[性能监控] 转写速度: {audio_duration/transcribe_time:.2f}x 实时速度")
        logger.info(f"[性能监控] ════════════════════════════════════════")
        
        # Create the response
        response = TranscriptionResponse(
            code=0,
            msg=f"success, total: {total_time:.2f}s (download: {download_time:.2f}s, process: {audio_process_time:.2f}s, transcribe: {transcribe_time:.2f}s)",
            data=text
        )
    except Exception as e: 
        logger.error("Exception occurred", exc_info=True)
        response = TranscriptionResponse(
            code=1,
            msg=str(e),
            data=""
        )
    return JSONResponse(content=response.model_dump())


def is_douyin_url(url: str) -> bool:
    """判断是否为抖音链接"""
    douyin_patterns = [
        r'douyin\.com',
        r'douyin\.io',
        r'iesdouyin\.com',
        r'douyinvod\.com',  # 抖音 CDN
    ]
    return any(re.search(pattern, url, re.IGNORECASE) for pattern in douyin_patterns)


def is_direct_video_url(url: str) -> bool:
    """判断是否为视频直链（包含常见视频格式或抖音 CDN）"""
    video_extensions = ['.mp4', '.webm', '.flv', '.m3u8', '.mov', '.avi']
    douyin_cdn_patterns = [
        r'douyinvod\.com',
        r'/aweme/',
        r'/video/tos/',
    ]
    return (
        any(url.lower().endswith(ext) for ext in video_extensions) or
        any(re.search(pattern, url, re.IGNORECASE) for pattern in douyin_cdn_patterns)
    )


def get_douyin_headers() -> dict:
    """获取抖音下载所需的 headers（参考 Node.js 实现）"""
    return {
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


async def download_video_and_extract_audio(video_url: str) -> bytes:
    """
    下载视频并提取音频
    
    Args:
        video_url: 视频链接
        
    Returns:
        audio_bytes: 音频字节数据
    """
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 检查是否为视频直链
        if not is_direct_video_url(video_url):
            raise HTTPException(
                status_code=400,
                detail="请提供视频直链 URL（如：https://.../aweme/.../video.mp4），而不是抖音网页链接"
            )
        
        # 如果是抖音链接，使用特殊的 headers
        headers = None
        if is_douyin_url(video_url):
            headers = get_douyin_headers()
        
        # 下载视频
        download_http_start = time.time()
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=True,
            verify=True,
            http2=False,
        ) as client:
            try:
                response = await client.get(video_url, headers=headers)
            except httpx.HTTPStatusError as e:
                logger.error(f"[视频下载] HTTP 状态错误: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"下载视频失败，状态码: {e.response.status_code}"
                )
            
            if response.status_code != 200:
                logger.error(f"[视频下载] 状态码非 200: {response.status_code}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"下载视频失败，状态码: {response.status_code}"
                )
            
            video_bytes = response.content
        
        download_http_time = time.time() - download_http_start
        logger.info(f"[性能监控] HTTP 下载耗时: {download_http_time:.2f} 秒，视频大小: {len(video_bytes) / 1024 / 1024:.2f} MB")
        
        # 保存视频到临时文件
        video_ext = 'mp4'
        video_file = os.path.join(temp_dir, f'video.{video_ext}')
        with open(video_file, 'wb') as f:
            f.write(video_bytes)
        
        # 使用 ffmpeg 提取音频
        extract_start = time.time()
        audio_file = os.path.join(temp_dir, 'audio.wav')
        
        cmd = [
            'ffmpeg',
            '-i', video_file,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            audio_file
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            logger.error(f"[视频下载] ffmpeg 错误: {result.stderr}")
            raise HTTPException(
                status_code=400,
                detail=f"音频提取失败: {result.stderr}"
            )
        
        extract_time = time.time() - extract_start
        
        # 读取音频文件
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
        
        logger.info(f"[性能监控] ffmpeg 提取耗时: {extract_time:.2f} 秒，音频大小: {len(audio_bytes) / 1024 / 1024:.2f} MB")
        return audio_bytes
        
    except httpx.HTTPError as e:
        logger.error(f"[视频下载] HTTP 错误: {e}")
        raise HTTPException(status_code=400, detail=f"HTTP 请求失败: {str(e)}")
    except subprocess.TimeoutExpired:
        logger.error(f"[视频下载] ffmpeg 超时")
        raise HTTPException(status_code=400, detail="音频提取超时")
    except Exception as e:
        logger.error(f"[视频下载] 下载失败: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")
    finally:
        # 清理临时文件
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"[视频下载] 清理临时文件失败: {e}")


@app.post("/transcribe_video", response_model=TranscriptionResponse)
async def transcribe_video(
    video_url: str = Form(...),
):
    """
    转写视频中的音频
    接收视频链接，下载视频并提取音频进行转写
    支持抖音链接（自动添加防盗链 headers）
    """
    try:
        start_total = time.time()
        
        # 下载视频并提取音频
        download_start = time.time()
        audio_bytes = await download_video_and_extract_audio(video_url)
        download_time = time.time() - download_start
        
        if not audio_bytes or len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Failed to extract audio from video")
        
        # 使用 torchaudio 读取音频
        audio_process_start = time.time()
        try:
            bio = io.BytesIO(audio_bytes)
            input_wav, sr = torchaudio.load(bio)
            is16 = False
            input_wav = input_wav.squeeze().numpy()
        except Exception as e:
            logger.error(f"[视频转写] 音频解码失败: {e}")
            raise HTTPException(status_code=400, detail="Unsupported or invalid audio format")
        
        # 处理多声道音频
        if len(input_wav.shape) > 1:
            input_wav = input_wav.mean(-1)
        
        # 转换为 float32
        if input_wav.dtype != np.float32:
            input_wav = input_wav.astype(np.float32)
            if is16:
                input_wav = input_wav / np.iinfo(np.int16).max
        
        # 重采样到 16000 Hz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
        
        audio_process_time = time.time() - audio_process_start
        audio_duration = len(input_wav) / 16000  # 音频时长（秒）
        
        # 执行转写
        transcribe_start = time.time()
        async def generate_text():
            return await asyncio.to_thread(transcribe_with_timing, 
                                           input=input_wav,
                                           cache={},
                                           language="auto",
                                           use_itn=True,
                                           batch_size=64)
        
        resp, elapsed_time = await generate_text()
        transcribe_time = time.time() - transcribe_start
        
        text = format_str_v3(resp[0]["text"])
        
        # 总耗时统计
        total_time = time.time() - start_total
        logger.info(f"[性能监控] ════════════════════════════════════════")
        logger.info(f"[性能监控] 总计耗时: {total_time:.2f} 秒")
        logger.info(f"[性能监控]   - 下载+提取: {download_time:.2f} 秒 ({download_time/total_time*100:.1f}%)")
        logger.info(f"[性能监控]   - 音频处理: {audio_process_time:.2f} 秒 ({audio_process_time/total_time*100:.1f}%)")
        logger.info(f"[性能监控]   - ASR 转写: {transcribe_time:.2f} 秒 ({transcribe_time/total_time*100:.1f}%)")
        logger.info(f"[性能监控] 转写速度: {audio_duration/transcribe_time:.2f}x 实时速度")
        logger.info(f"[性能监控] ════════════════════════════════════════")
        
        # 创建响应
        response = TranscriptionResponse(
            code=0,
            msg=f"success, total time: {total_time:.2f}s (download: {download_time:.2f}s, process: {audio_process_time:.2f}s, transcribe: {transcribe_time:.2f}s)",
            data=text
        )
        
    except HTTPException:
        raise
    except Exception as e: 
        logger.error("[视频转写] 发生异常", exc_info=True)
        response = TranscriptionResponse(
            code=1,
            msg=str(e),
            data=""
        )
    
    return JSONResponse(content=response.model_dump())


if __name__ == "__main__":
    # 强制刷新输出缓冲区
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    logger.info("=" * 50)
    logger.info("SenseVoice API 服务启动中...")
    logger.info("=" * 50)
    
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=7008, help='Port number to run the FastAPI app on.')
    parser.add_argument('--certfile', type=str, default=None, help='SSL certificate file')
    parser.add_argument('--keyfile', type=str, default=None, help='SSL key file')
    args = parser.parse_args()
    
    logger.info(f"服务器将启动在端口: {args.port}")
    
    # 检查SSL证书文件是否存在
    import os
    use_ssl = False
    if args.certfile and args.keyfile:
        if os.path.exists(args.certfile) and os.path.exists(args.keyfile):
            use_ssl = True
            print(f"Using SSL with cert: {args.certfile}, key: {args.keyfile}")
        else:
            print("SSL certificate files not found, running without SSL")
    else:
        print("No SSL certificates specified, running without SSL")
    
    # 配置 uvicorn 日志
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["default"],
        },
    }
    
    if use_ssl:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=args.port, 
            ssl_certfile=args.certfile, 
            ssl_keyfile=args.keyfile,
            log_config=log_config
        )
    else:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=args.port,
            log_config=log_config
        )