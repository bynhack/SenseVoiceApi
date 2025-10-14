from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from funasr import AutoModel
import numpy as np
import soundfile as sf
import argparse
import uvicorn
from urllib.parse import parse_qs
import os
import re
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from loguru import logger
import sys
import json
import traceback
import time

logger.remove()
log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level}] {file}:{line} - {message}"
logger.add(sys.stdout, format=log_format, level="DEBUG", filter=lambda record: record["level"].no < 40)
logger.add(sys.stderr, format=log_format, level="ERROR", filter=lambda record: record["level"].no >= 40)


class Config(BaseSettings):
    sv_thr: float = Field(0.3, description="说话人验证阈值")
    chunk_size_ms: int = Field(300, description="分片大小（毫秒）")
    sample_rate: int = Field(16000, description="采样率（Hz）")
    bit_depth: int = Field(16, description="位深")
    channels: int = Field(1, description="声道数")
    avg_logprob_thr: float = Field(-0.25, description="平均对数概率阈值")

config = Config()

emo_dict = {
	"<|HAPPY|>": "",
	"<|SAD|>": "",
	"<|ANGRY|>": "",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "",
	"<|DISGUSTED|>": "",
	"<|SURPRISED|>": "",
}

event_dict = {
	"<|BGM|>": "",
	"<|Speech|>": "",
	"<|Applause|>": "",
	"<|Laughter|>": "",
	"<|Cry|>": "",
	"<|Sneeze|>": "",
	"<|Breath|>": "",
	"<|Cough|>": "",
}

emoji_dict = {
	"<|nospeech|><|Event_UNK|>": "",
	"<|zh|>": "",
	"<|en|>": "",
	"<|yue|>": "",
	"<|ja|>": "",
	"<|ko|>": "",
	"<|nospeech|>": "",
	"<|HAPPY|>": "",
	"<|SAD|>": "",
	"<|ANGRY|>": "",
	"<|NEUTRAL|>": "",
	"<|BGM|>": "",
	"<|Speech|>": "",
	"<|Applause|>": "",
	"<|Laughter|>": "",
	"<|FEARFUL|>": "",
	"<|DISGUSTED|>": "",
	"<|SURPRISED|>": "",
	"<|Cry|>": "",
	"<|EMO_UNKNOWN|>": "",
	"<|Sneeze|>": "",
	"<|Breath|>": "",
	"<|Cough|>": "",
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

def contains_chinese_english_number(s: str) -> bool:
    # Check if the string contains any Chinese character, English letter, or Arabic number
    return bool(re.search(r'[\u4e00-\u9fffA-Za-z0-9]', s))


# ============ 模型加载配置 ============
# 优先使用本地缓存,避免网络问题

# 1. 说话人验证模型
sv_model_cache = os.path.expanduser('~/.cache/modelscope/hub/iic/speech_campplus_sv_zh-cn_16k-common')
if not os.path.exists(sv_model_cache):
    sv_model_cache = 'iic/speech_campplus_sv_zh-cn_16k-common'
    logger.warning(f"本地缓存不存在,将尝试从 ModelScope 下载: {sv_model_cache}")
else:
    logger.info(f"使用本地缓存的说话人验证模型: {sv_model_cache}")

sv_pipeline = pipeline(
    task='speaker-verification',
    model=sv_model_cache,
    model_revision='v1.0.0' if sv_model_cache.startswith('iic/') else None
)

# 2. ASR 模型(使用微调权重)
asr_model_cache = os.path.expanduser('~/.cache/modelscope/hub/iic/SenseVoiceSmall')
if not os.path.exists(asr_model_cache):
    asr_model_cache = "iic/SenseVoiceSmall"
    logger.warning(f"本地缓存不存在,将尝试从 ModelScope 下载: {asr_model_cache}")
else:
    logger.info(f"使用本地缓存的 ASR 模型: {asr_model_cache}")

# 先加载原始模型
model_asr = AutoModel(
    model=asr_model_cache,
    trust_remote_code=True,
    remote_code="./model.py",    
    device="cuda:0",
    disable_update=True,
    initial_model="./outputs/model.pt.avg10"  # 加载微调后的平均权重(推荐)
)

# 3. VAD 模型
vad_model_cache = os.path.expanduser('~/.cache/modelscope/hub/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch')
if not os.path.exists(vad_model_cache):
    vad_model_cache = "fsmn-vad"
    vad_model_revision = "v2.0.4"
    logger.warning(f"本地缓存不存在,将尝试从 ModelScope 下载: {vad_model_cache}")
else:
    vad_model_cache = vad_model_cache
    vad_model_revision = None
    logger.info(f"使用本地缓存的 VAD 模型: {vad_model_cache}")

model_vad = AutoModel(
    model=vad_model_cache,
    model_revision=vad_model_revision,
    disable_pbar=True,
    max_end_silence_time=500,
    # speech_noise_thres=0.6,
    disable_update=True,
)

reg_spks_files = [
    "speaker/record_out.wav"
]

def reg_spk_init(files):
    reg_spk = {}
    for f in files:
        data, sr = sf.read(f, dtype="float32")
        k, _ = os.path.splitext(os.path.basename(f))
        reg_spk[k] = {
            "data": data,
            "sr":   sr,
        }
    return reg_spk

reg_spks = reg_spk_init(reg_spks_files)

def speaker_verify(audio, sv_thr):
    hit = False
    for k, v in reg_spks.items():
        res_sv = sv_pipeline([audio, v["data"]], sv_thr)
        if res_sv["score"] >= sv_thr:
           hit = True
        logger.info(f"[speaker_verify] audio_len: {len(audio)}; sv_thr: {sv_thr}; hit: {hit}; {k}: {res_sv}")
    return hit, k


def speaker_verify_enrolled(audio, ref_audio, sv_thr):
    """
    使用当前语音段开头截取的参考音频(ref_audio)进行说话人验证。
    """
    if ref_audio is None or len(ref_audio) == 0:
        return False, ""
    res_sv = sv_pipeline([audio, ref_audio], sv_thr)
    hit = res_sv["score"] >= sv_thr
    logger.info(f"[speaker_verify_enrolled] audio_len: {len(audio)}; sv_thr: {sv_thr}; hit: {hit}; score: {res_sv}")
    return hit, "segment_enroll"


def asr(audio, lang, cache, use_itn=False):
    # with open('test.pcm', 'ab') as f:
    #     logger.debug(f'write {f.write(audio)} bytes to `test.pcm`')
    # result = asr_pipeline(audio, lang)
    start_time = time.time()
    result = model_asr.generate(
        input           = audio,
        cache           = cache,
        language        = lang.strip(),
        use_itn         = use_itn,
        batch_size_s    = 60,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug(f"asr elapsed: {elapsed_time * 1000:.2f} milliseconds")
    return result

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    info: str
    data: str

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    try:
        query_params = parse_qs(websocket.scope['query_string'].decode())
        sv = query_params.get('sv', ['false'])[0].lower() in ['true', '1', 't', 'y', 'yes']
        lang = query_params.get('lang', ['auto'])[0].lower()
        
        await websocket.accept()
        chunk_size = int(config.chunk_size_ms * config.sample_rate / 1000)
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)

        cache = {}
        cache_asr = {}
        # 当前语音段的参考声纹，满足最小长度后用于本段内部的说话人验证
        segment_ref = None
        segment_reference_established = False
        # 记录当前语音段的起始采样点，便于截取参考片段
        segment_start_idx = None
        # 标记是否已向前端发送“本段已登记”事件
        segment_ref_announced = False
        # 记录当前语音段的起始时间戳，用于统计从收到音频到识别完成的耗时
        segment_start_time = None
        # 参考片段的最小采样长度（此处为 0.5 秒）
        segment_ref_min_samples = int(0.5 * config.sample_rate)
        last_vad_beg = last_vad_end = -1
        offset = 0
        hit = False
        
        buffer = b""
        while True:
            data = await websocket.receive_bytes()
            # logger.info(f"received {len(data)} bytes")

            
            buffer += data
            if len(buffer) < 2:
                continue
                
            audio_buffer = np.append(
                audio_buffer, 
                np.frombuffer(buffer[:len(buffer) - (len(buffer) % 2)], dtype=np.int16).astype(np.float32) / 32767.0
            )
            
            # with open('buffer.pcm', 'ab') as f:
            #     logger.debug(f'write {f.write(buffer[:len(buffer) - (len(buffer) % 2)])} bytes to `buffer.pcm`')
                
            buffer = buffer[len(buffer) - (len(buffer) % 2):]
   
            while len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]
                audio_vad = np.append(audio_vad, chunk)
                
                # with open('chunk.pcm', 'ab') as f:
                #     logger.debug(f'write {f.write(chunk)} bytes to `chunk.pcm`')
                    
                if last_vad_beg > 1:
                    if sv:
                        # speaker verify
                        # If no hit is detected, continue accumulating audio data and check again until a hit is detected
                        # `hit` will reset after `asr`.
                        if not hit:
                            # 若已完成本语音段的参考声纹登记，则用登记的参考进行比对；
                            # 否则尝试从该语音段起始处截取最小参考长度进行登记。
                            if segment_start_idx is None:
                                start_ms = max(0.0, last_vad_beg - offset)
                                segment_start_idx = max(
                                    0,
                                    int(start_ms * config.sample_rate / 1000)
                                )
                                if segment_start_time is None:
                                    segment_start_time = time.time()
                                logger.info(
                                    f"[segment] 检测到语音段起点，起始采样点={segment_start_idx}，时间戳={segment_start_time:.3f}"
                                )

                            current_segment = audio_vad[segment_start_idx:]

                            if segment_ref is None and len(current_segment) >= segment_ref_min_samples:
                                segment_ref = current_segment[:segment_ref_min_samples].copy()
                                segment_reference_established = True
                                logger.info(
                                    f"[enroll] 建立语音段参考声纹，采样点数={len(segment_ref)}"
                                )
                                if not segment_ref_announced:
                                    try:
                                        await websocket.send_json(TranscriptionResponse(
                                            code=3,
                                            info="enrolled",
                                            data="segment_enroll"
                                        ).model_dump())
                                    except Exception as _:
                                        logger.debug("发送本段登记事件失败，继续执行")
                                    segment_ref_announced = True

                            if segment_ref is not None and len(current_segment) > len(segment_ref):
                                audio_for_verify = current_segment[-len(segment_ref):]
                                hit, speaker = speaker_verify_enrolled(
                                    audio_for_verify,
                                    segment_ref,
                                    config.sv_thr
                                )
                            else:
                                hit, speaker = False, ""
                            if hit:
                                response = TranscriptionResponse(
                                    code=2,
                                    info="detect speaker",
                                    data=speaker
                                )
                                await websocket.send_json(response.model_dump())
                    else:
                        response = TranscriptionResponse(
                            code=2,
                            info="detect speech",
                            data=''
                        )
                        await websocket.send_json(response.model_dump())

                res = model_vad.generate(input=chunk, cache=cache, is_final=False, chunk_size=config.chunk_size_ms)
                # logger.info(f"vad inference: {res}")
                if len(res[0]["value"]):
                    vad_segments = res[0]["value"]
                    for segment in vad_segments:
                        if segment[0] > -1: # speech begin
                            last_vad_beg = segment[0]
                            if segment_start_time is None:
                                segment_start_time = time.time()
                                start_sample = int(max(0.0, (segment[0] - offset)) * config.sample_rate / 1000)
                                logger.info(
                                    f"[segment] 检测到语音段起点，起始采样点={start_sample}，时间戳={segment_start_time:.3f}"
                                )
                        if segment[1] > -1: # speech end
                            last_vad_end = segment[1]
                        if last_vad_beg > -1 and last_vad_end > -1:
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            beg = int(last_vad_beg * config.sample_rate / 1000)
                            end = int(last_vad_end * config.sample_rate / 1000)
                            logger.info(f"[vad segment] audio_len: {end - beg}")
                            # 若开启了声纹且尚未命中，但这是登记段，则仍然执行 ASR；否则保持原策略
                            allow_asr = (not sv) or hit or segment_reference_established or segment_ref is None
                            result = asr(audio_vad[beg:end], lang.strip(), cache_asr, True) if allow_asr else None
                            logger.info(f"asr response: {result}")

                            segment_elapsed_ms = None
                            if segment_start_time is not None:
                                segment_elapsed_ms = (time.time() - segment_start_time) * 1000

                            audio_vad = audio_vad[end:]
                            last_vad_beg = last_vad_end = -1
                            hit = False
                            segment_ref = None
                            segment_reference_established = False
                            segment_start_idx = None
                            segment_ref_announced = False
                            logger.info("[segment] 语音段结束，已清空参考声纹缓存")

                            if segment_elapsed_ms is not None:
                                logger.info(f"[segment] 语音段总耗时：{segment_elapsed_ms:.2f} 毫秒")
                            segment_start_time = None

                            if result is not None:
                                response = TranscriptionResponse(
                                    code=0,
                                    info=json.dumps(result[0], ensure_ascii=False),
                                    data=format_str_v3(result[0]['text'])
                                )
                                await websocket.send_json(response.model_dump())
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")                            
    except Exception as e:
        logger.error(f"Unexpected error: {e}\nCall stack:\n{traceback.format_exc()}")
        await websocket.close()
    finally:
        audio_buffer = np.array([], dtype=np.float32)
        cache.clear()
        logger.info("Cleaned up resources after WebSocket disconnect")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=27000, help='Port number to run the FastAPI app on.')
    # parser.add_argument('--certfile', type=str, default='path_to_your_SSL_certificate_file.crt', help='SSL certificate file')
    # parser.add_argument('--keyfile', type=str, default='path_to_your_SSL_certificate_file.key', help='SSL key file')
    args = parser.parse_args()
    # uvicorn.run(app, host="0.0.0.0", port=args.port, ssl_certfile=args.certfile, ssl_keyfile=args.keyfile)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
