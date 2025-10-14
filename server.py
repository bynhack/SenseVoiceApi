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
import httpx
import os

# Set up logging
logging.basicConfig(level=logging.ERROR)
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
model = AutoModel(model=asr_model_cache,
#                  vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
#                  vad_kwargs={"max_single_segment_time": 30000},
                  trust_remote_code=True,
                  )



def transcribe_with_timing(*args, **kwargs):
    start_time = time.time()
    result = model.generate(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Transcription execution time: {elapsed_time:.2f} seconds")
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
    try:
        # 统一从本地上传文件或 URL 下载获取字节与类型
        file_content: bytes = None
        content_type: str = ""
        filename_hint: str = ""

        if file is not None:
            # Read the file content and reset the file pointer
            file.file.seek(0)
            file_content = await file.read()
            content_type = file.content_type or ""
            filename_hint = file.filename or ""

        if (file_content is None or len(file_content) == 0) and url:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to download url, status={resp.status_code}")
                file_content = resp.content
                content_type = resp.headers.get("content-type", "")
                # 从 URL 推断后缀
                filename_hint = url.split("?")[0].split("#")[0]

        if file_content is None or len(file_content) == 0:
            raise HTTPException(status_code=400, detail="No audio provided. Provide file or url")

        print(f"[DEBUG] UploadFile Object is {file}, url={url}")
        # 根据 content-type 或文件名后缀判断解析方式
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
            print(f"[DEBUG] Audio data sample rate is {sr}")
            resampler = torchaudio.transforms.Resample(sr, 16000)
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
                
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
        print(f"[DEBUG] Transcribe raw response is {resp}")
        text = format_str_v3(resp[0]["text"])
        print(f'[DEBUG] res:{resp} text:{text}')
        
        # Create the response
        response = TranscriptionResponse(
            code=0,
            msg=f"success, transcription time: {elapsed_time:.2f} seconds",
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=7000, help='Port number to run the FastAPI app on.')
    parser.add_argument('--certfile', type=str, default=None, help='SSL certificate file')
    parser.add_argument('--keyfile', type=str, default=None, help='SSL key file')
    args = parser.parse_args()
    
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
    
    if use_ssl:
        uvicorn.run(app, host="0.0.0.0", port=args.port, ssl_certfile=args.certfile, ssl_keyfile=args.keyfile)
    else:
        uvicorn.run(app, host="0.0.0.0", port=args.port)