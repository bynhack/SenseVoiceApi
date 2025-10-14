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
from difflib import SequenceMatcher
from collections import defaultdict
import scipy.signal as signal
from scipy.signal import butter, lfilter, wiener

logger.remove()
log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level}] {file}:{line} - {message}"
logger.add(sys.stdout, format=log_format, level="DEBUG", filter=lambda record: record["level"].no < 40)
logger.add(sys.stderr, format=log_format, level="ERROR", filter=lambda record: record["level"].no >= 40)


class Config(BaseSettings):
    sd_thr: float = Field(0.25, description="说话人分离相似度阈值")  # 降低阈值，避免同一人被识别成多人
    chunk_size_ms: int = Field(300, description="分片大小（毫秒）")
    sample_rate: int = Field(16000, description="采样率（Hz）")
    bit_depth: int = Field(16, description="位深")
    channels: int = Field(1, description="声道数")
    avg_logprob_thr: float = Field(-0.25, description="平均对数概率阈值")
    min_confidence: float = Field(0.6, description="最小置信度阈值")
    enable_error_correction: bool = Field(True, description="启用错误修正")
    filter_noise_words: bool = Field(True, description="过滤噪音误识别词")
    min_text_length: int = Field(2, description="最小有效文本长度")
    enable_audio_denoise: bool = Field(True, description="启用音频降噪")
    denoise_method: str = Field("spectral", description="降噪方法: spectral/wiener/bandpass")
    noise_reduce_strength: float = Field(0.5, description="降噪强度(0.0-1.0)")

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


# ============ 音频降噪模块 ============

class AudioDenoiser:
    """
    音频降噪器,用于在ASR识别前对音频信号进行降噪处理
    支持多种降噪算法:
    1. 频谱减法(Spectral Subtraction)
    2. 维纳滤波(Wiener Filter)
    3. 带通滤波(Bandpass Filter)
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        # 语音频率范围: 300Hz - 3400Hz
        self.speech_lowcut = 300.0
        self.speech_highcut = 3400.0
        
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        """
        设计带通滤波器
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        应用带通滤波
        """
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    def spectral_subtraction(self, audio, noise_profile=None, strength=0.5):
        """
        频谱减法降噪
        
        Args:
            audio: 音频信号
            noise_profile: 噪声频谱(如果为None,使用前10%作为噪声估计)
            strength: 降噪强度(0.0-1.0)
        """
        # 计算FFT
        fft_audio = np.fft.rfft(audio)
        power_spectrum = np.abs(fft_audio) ** 2
        
        # 估计噪声频谱(使用前10%的音频)
        if noise_profile is None:
            noise_len = len(audio) // 10
            noise_audio = audio[:noise_len]
            fft_noise = np.fft.rfft(noise_audio)
            noise_power = np.abs(fft_noise) ** 2
            # 扩展到与音频相同长度
            noise_profile = np.mean(noise_power) * np.ones_like(power_spectrum)
        
        # 频谱减法
        clean_power = power_spectrum - strength * noise_profile
        clean_power = np.maximum(clean_power, 0.1 * power_spectrum)  # 避免过度减法
        
        # 重建信号
        phase = np.angle(fft_audio)
        clean_fft = np.sqrt(clean_power) * np.exp(1j * phase)
        clean_audio = np.fft.irfft(clean_fft, n=len(audio))
        
        return clean_audio.astype(np.float32)
    
    def wiener_filter_denoise(self, audio, noise_power=None):
        """
        维纳滤波降噪
        """
        try:
            # 使用scipy的wiener滤泥
            denoised = wiener(audio)
            return denoised.astype(np.float32)
        except Exception as e:
            logger.warning(f"[维纳滤波] 失败: {e}, 返回原始音频")
            return audio
    
    def energy_based_filter(self, audio, threshold_percentile=20):
        """
        基于能量的简单降噪
        将低能量帧设为零(可能是噪声)
        """
        # 计算短时能量
        frame_length = 400  # 25ms @ 16kHz
        hop_length = 160    # 10ms @ 16kHz
        
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            energy.append(np.sum(frame ** 2))
        
        if len(energy) == 0:
            return audio
        
        # 计算能量阈值
        threshold = np.percentile(energy, threshold_percentile)
        
        # 应用阈值
        denoised = audio.copy()
        for i, e in enumerate(energy):
            if e < threshold:
                start = i * hop_length
                end = start + frame_length
                if end <= len(denoised):
                    denoised[start:end] *= 0.1  # 衰减但不完全消除
        
        return denoised
    
    def denoise(self, audio, method="spectral", strength=0.5):
        """
        对音频进行降噪处理
        
        Args:
            audio: 音频信号(numpy array)
            method: 降噪方法 ("spectral", "wiener", "bandpass", "energy")
            strength: 降噪强度(0.0-1.0)
        
        Returns:
            denoised_audio: 降噪后的音频
        """
        if len(audio) == 0:
            return audio
        
        try:
            if method == "spectral":
                # 频谱减法
                denoised = self.spectral_subtraction(audio, strength=strength)
                logger.debug(f"[音频降噪] 使用频谱减法,强度={strength}")
            
            elif method == "wiener":
                # 维纳滤波
                denoised = self.wiener_filter_denoise(audio)
                logger.debug(f"[音频降噪] 使用维纳滤波")
            
            elif method == "bandpass":
                # 带通滤波
                denoised = self.bandpass_filter(
                    audio, 
                    self.speech_lowcut, 
                    self.speech_highcut, 
                    self.sample_rate
                )
                logger.debug(f"[音频降噪] 使用带通滤波 {self.speech_lowcut}-{self.speech_highcut}Hz")
            
            elif method == "energy":
                # 能量阈值法
                denoised = self.energy_based_filter(audio)
                logger.debug(f"[音频降噪] 使用能量阈值法")
            
            else:
                logger.warning(f"[音频降噪] 未知方法: {method}, 返回原始音频")
                return audio
            
            # 确保输出范围在[-1, 1]
            denoised = np.clip(denoised, -1.0, 1.0)
            
            return denoised
            
        except Exception as e:
            logger.error(f"[音频降噪] 失败: {e}")
            return audio


# 初始化音频降噪器
audio_denoiser = AudioDenoiser(sample_rate=16000)


# ============ ASR错误修正模块 ============

class ASRErrorCorrector:
    """
    ASR错误修正器,用于修正三种常见错误:
    1. 替换错误(Substitution): 同音字、形近字
    2. 插入错误(Insertion): 多余的语气词、重复字
    3. 删除错误(Deletion): 缺失的字词
    """
    
    def __init__(self):
        # 同音字替换词典(替换错误)
        self.homophone_dict = {
            "小右": "小佑",
            "小优": "小佑",
            "小有": "小佑",
            "在那": "再拿",
            "在拿": "再拿",
            "的话": "得话",
            "做到": "做到",
            "在见": "再见",
            "在会": "再会",
            # 可以继续添加更多同音字规则
        }
        
        # 专有名词词典(替换错误)
        self.proper_noun_dict = {
            "阿里巴巴": ["阿里爸爸", "阿里吧吧"],
            "腾讯": ["腾信", "腾讯"],
            "ChatGPT": ["chat gpt", "chatgpt", "chat GPT"],
            "SenseVoice": ["sense voice", "sensevoice"],
            # 可以继续添加更多专有名词
        }
        
        # 常见插入错误(语气词、重复)
        self.insertion_patterns = [
            r'(.)\1{2,}',  # 连续重复3次以上的字符
            r'(呃|啊|嗯|哦){2,}',  # 重复的语气词
        ]
        
        # 噪音误识别词列表(通常是单个字或语气词)
        self.noise_words = {
            "嗯", "啊", "呃", "哦", "唔", "嘛", "呢", "哪",
            "额", "诶", "哎", "嘿", "喂", "咦", "哟", "嗨",
            "嘘", "咳", "哼", "嗳", "哇", "噢", "嗷", "咿",
            # 单个标点或无意义字符
            "。", "，", "、", "？", "！", " ", "　",
        }
        
        # 噪音模式(正则表达式)
        self.noise_patterns = [
            r'^[嗯啊呃哦唔嘛呢哪额诶哎嘿喂咦哟嗨]+$',  # 纯语气词
            r'^[。，、？！\s]+$',  # 纯标点
            r'^[a-zA-Z]$',  # 单个英文字母
        ]
        
        # 常见删除错误的补全规则
        self.completion_rules = {
            r'不是': '不是吗',
            r'可以': '可以的',
            r'没有': '没有了',
        }
        
        # 上下文N-gram统计(用于基于概率的修正)
        self.bigram_freq = defaultdict(int)
        self.trigram_freq = defaultdict(int)
        
    def correct_homophones(self, text: str) -> str:
        """
        修正同音字错误(替换错误)
        """
        corrected = text
        for wrong, correct in self.homophone_dict.items():
            if wrong in corrected:
                corrected = corrected.replace(wrong, correct)
                logger.debug(f"[同音字修正] '{wrong}' -> '{correct}'")
        return corrected
    
    def correct_proper_nouns(self, text: str) -> str:
        """
        修正专有名词错误(替换错误)
        """
        corrected = text
        for correct_noun, wrong_variants in self.proper_noun_dict.items():
            for wrong in wrong_variants:
                if wrong in corrected.lower():
                    # 使用正则表达式进行大小写不敏感的替换
                    pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                    corrected = pattern.sub(correct_noun, corrected)
                    logger.debug(f"[专有名词修正] '{wrong}' -> '{correct_noun}'")
        return corrected
    
    def is_noise_text(self, text: str, confidence: float = 1.0) -> bool:
        """
        判断文本是否为噪音误识别
        """
        if not text or len(text.strip()) == 0:
            return True
        
        text = text.strip()
        
        # 检查是否在噪音词列表中
        if text in self.noise_words:
            return True
        
        # 检查是否匹配噪音模式
        for pattern in self.noise_patterns:
            if re.match(pattern, text):
                return True
        
        # 检查是否只包含语气词和标点
        clean_text = re.sub(r'[嗯啊呃哦唔嘛呢哪额诶哎嘿喂咦哟嗨。，、？！\s]', '', text)
        if len(clean_text) == 0:
            return True
        
        # 特殊检查: 单个语气词+标点(如"嗯。"、"啊!")
        if len(text) <= 2:
            # 移除标点后检查
            text_no_punct = re.sub(r'[。，、？！\s]', '', text)
            if text_no_punct in self.noise_words:
                return True
        
        # 单字+标点的情况,结合置信度判断
        # 移除标点后只有1个字,且置信度不高,可能是噪音
        text_no_punct_final = re.sub(r'[。，、？！\s]', '', text)
        if len(text_no_punct_final) == 1 and confidence < 0.85:
            logger.debug(f"[噪音检测] 单字低置信度: '{text}' (置信度={confidence:.2f})")
            return True
        
        return False
    
    def remove_insertions(self, text: str) -> str:
        """
        移除插入错误(多余字符)
        """
        corrected = text
        for pattern in self.insertion_patterns:
            matches = re.finditer(pattern, corrected)
            for match in matches:
                original = match.group(0)
                # 保留一个字符,删除重复的
                replacement = match.group(1)
                corrected = corrected.replace(original, replacement, 1)
                logger.debug(f"[插入错误修正] 删除重复: '{original}' -> '{replacement}'")
        return corrected
    
    def filter_noise(self, text: str, confidence: float = 1.0) -> tuple[str, bool]:
        """
        过滤噪音误识别的文本
        
        Returns:
            filtered_text: 过滤后的文本
            is_filtered: 是否被过滤
        """
        if self.is_noise_text(text, confidence):
            logger.info(f"[噪音过滤] 检测到噪音文本: '{text}'")
            return "", True
        
        # 移除文本开头和结尾的语气词
        original = text
        text = re.sub(r'^[嗯啊呃哦唔]+', '', text)
        text = re.sub(r'[嗯啊呃哦唔]+$', '', text)
        
        if text != original:
            logger.debug(f"[噪音过滤] 移除首尾语气词: '{original}' -> '{text}'")
        
        return text.strip(), False
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        计算两个字符串的相似度(用于检测替换错误)
        """
        return SequenceMatcher(None, str1, str2).ratio()
    
    def correct_by_context(self, text: str, prev_text: str = "") -> str:
        """
        基于上下文进行修正(可以检测删除错误)
        这里使用简单的规则,实际应用中可以使用语言模型
        """
        corrected = text
        
        # 检查是否缺少标点符号(删除错误)
        if len(corrected) > 10 and not re.search(r'[,。!?、]', corrected):
            # 简单规则:在句子中间添加逗号
            mid_point = len(corrected) // 2
            corrected = corrected[:mid_point] + ',' + corrected[mid_point:]
            logger.debug(f"[删除错误修正] 添加标点符号")
        
        return corrected
    
    def correct_text(self, text: str, prev_text: str = "", confidence: float = 1.0, filter_noise: bool = True) -> tuple[str, list[dict], bool]:
        """
        综合修正文本中的各种错误
        
        Args:
            text: 待修正的文本
            prev_text: 前一句文本(用于上下文分析)
            confidence: 识别置信度
            filter_noise: 是否过滤噪音
        
        Returns:
            corrected_text: 修正后的文本
            corrections: 修正记录列表
            is_noise: 是否为噪音(应该被丢弃)
        """
        original_text = text
        corrections = []
        
        # 0. 噪音过滤(最优先)
        if filter_noise:
            text, is_filtered = self.filter_noise(text, confidence)
            if is_filtered:
                corrections.append({
                    "type": "noise_filter",
                    "original": original_text,
                    "corrected": "",
                    "method": "noise_detection"
                })
                return "", corrections, True  # 标记为噪音
            
            if text != original_text:
                corrections.append({
                    "type": "noise_filter",
                    "original": original_text,
                    "corrected": text,
                    "method": "remove_edge_noise"
                })
                original_text = text
        
        # 1. 修正同音字(替换错误)
        text = self.correct_homophones(text)
        if text != original_text:
            corrections.append({
                "type": "substitution",
                "original": original_text,
                "corrected": text,
                "method": "homophone"
            })
            original_text = text
        
        # 2. 修正专有名词(替换错误)
        text = self.correct_proper_nouns(text)
        if text != original_text:
            corrections.append({
                "type": "substitution",
                "original": original_text,
                "corrected": text,
                "method": "proper_noun"
            })
            original_text = text
        
        # 3. 移除插入错误
        text = self.remove_insertions(text)
        if text != original_text:
            corrections.append({
                "type": "insertion",
                "original": original_text,
                "corrected": text,
                "method": "remove_repetition"
            })
            original_text = text
        
        # 4. 基于上下文修正(可以检测删除错误)
        if prev_text:
            text = self.correct_by_context(text, prev_text)
            if text != original_text:
                corrections.append({
                    "type": "deletion",
                    "original": original_text,
                    "corrected": text,
                    "method": "context"
                })
        
        # 5. 最终检查:如果修正后文本太短或为空,标记为噪音
        if filter_noise and len(text.strip()) < 1:
            return "", corrections, True
        
        return text, corrections, False


# 初始化错误修正器
error_corrector = ASRErrorCorrector()


# sv_pipeline = pipeline(
#     task='speaker-verification',
#     model='iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k',
#     model_revision='v1.0.0'
# )

# ============ 模型加载配置 ============
# 优先使用本地缓存,避免网络问题

# 1. 说话人分离模型 - 用于提取声纹特征和说话人识别
sd_model_cache = os.path.expanduser('~/.cache/modelscope/hub/iic/speech_campplus_sv_zh-cn_16k-common')
if not os.path.exists(sd_model_cache):
    sd_model_cache = 'iic/speech_campplus_sv_zh-cn_16k-common'
    logger.warning(f"本地缓存不存在,将尝试从 ModelScope 下载: {sd_model_cache}")
else:
    logger.info(f"使用本地缓存的说话人分离模型: {sd_model_cache}")

sd_pipeline = pipeline(
    task='speaker-verification',
    model=sd_model_cache,
    model_revision='v1.0.0' if sd_model_cache.startswith('iic/') else None
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

# 旧的预注册说话人验证功能已移除，现在使用 identify_speaker() 进行动态说话人识别


def extract_speaker_embedding(audio):
    """
    提取音频的说话人声纹特征向量
    """
    try:
        # 使用说话人验证模型提取声纹特征
        result = sd_pipeline([audio, audio], 0.0)
        # 返回声纹特征(embedding)
        if 'emb' in result:
            return result['emb']
        else:
            # 如果没有直接返回embedding,使用模型内部方法
            return sd_pipeline.model.forward(audio)
    except Exception as e:
        logger.error(f"[extract_speaker_embedding] 提取声纹特征失败: {e}")
        return None


def compute_similarity(emb1, emb2):
    """
    计算两个声纹特征向量的余弦相似度
    """
    if emb1 is None or emb2 is None:
        return 0.0
    try:
        # 转换为numpy数组
        if not isinstance(emb1, np.ndarray):
            emb1 = np.array(emb1)
        if not isinstance(emb2, np.ndarray):
            emb2 = np.array(emb2)
        
        # 展平为一维数组（处理 (1, 192) 这样的二维数组）
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        
        # 计算余弦相似度
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    except Exception as e:
        logger.error(f"[compute_similarity] 计算相似度失败: {e}")
        return 0.0


def identify_speaker(audio, speaker_embeddings, sd_thr):
    """
    识别说话人,返回说话人ID
    
    Args:
        audio: 音频数据
        speaker_embeddings: 已知说话人的声纹特征字典 {speaker_id: embedding}
        sd_thr: 相似度阈值
    
    Returns:
        speaker_id: 说话人ID (speaker_1, speaker_2, ...)
        is_new: 是否是新说话人
    """
    # 提取当前音频的声纹特征
    current_emb = extract_speaker_embedding(audio)
    if current_emb is None:
        return None, False
    
    # 如果还没有注册的说话人,这是第一个说话人
    if len(speaker_embeddings) == 0:
        speaker_id = "speaker_1"
        speaker_embeddings[speaker_id] = current_emb
        logger.info(f"[identify_speaker] 注册新说话人: {speaker_id}")
        return speaker_id, True
    
    # 与已知说话人进行比对
    max_similarity = -1
    matched_speaker = None
    
    for spk_id, spk_emb in speaker_embeddings.items():
        similarity = compute_similarity(current_emb, spk_emb)
        logger.debug(f"[identify_speaker] 与 {spk_id} 的相似度: {similarity:.4f}")
        
        if similarity > max_similarity:
            max_similarity = similarity
            matched_speaker = spk_id
    
    # 如果最大相似度超过阈值,认为是已知说话人
    if max_similarity >= sd_thr:
        logger.info(f"[identify_speaker] 识别为已知说话人: {matched_speaker}, 相似度: {max_similarity:.4f}")
        return matched_speaker, False
    else:
        # 否则注册为新说话人
        new_speaker_id = f"speaker_{len(speaker_embeddings) + 1}"
        speaker_embeddings[new_speaker_id] = current_emb
        logger.info(f"[identify_speaker] 注册新说话人: {new_speaker_id}, 最大相似度: {max_similarity:.4f}")
        return new_speaker_id, True


def asr(audio, lang, cache, use_itn=False, prev_text="", enable_correction=True, enable_denoise=True):
    """
    执行ASR识别并进行错误修正
    
    Args:
        audio: 音频数据
        lang: 语言
        cache: 缓存
        use_itn: 是否使用ITN
        prev_text: 前一句文本(用于上下文修正)
        enable_correction: 是否启用错误修正
        enable_denoise: 是否启用音频降噪
    
    Returns:
        result: ASR识别结果
    """
    start_time = time.time()
    
    # 音频降噪处理
    if enable_denoise and config.enable_audio_denoise:
        denoise_start = time.time()
        audio = audio_denoiser.denoise(
            audio, 
            method=config.denoise_method,
            strength=config.noise_reduce_strength
        )
        denoise_elapsed = (time.time() - denoise_start) * 1000
        logger.debug(f"[音频降噪] 耗时: {denoise_elapsed:.2f} ms")
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
    
    # 错误修正处理
    if result and len(result) > 0 and 'text' in result[0] and enable_correction:
        original_text = result[0]['text']
        
        # 先移除特殊标记,得到纯文本
        clean_text = format_str_v3(original_text)
        logger.debug(f"[文本清理] 原始: '{original_text}' -> 清理: '{clean_text}'")
        
        # 获取置信度(如果有)
        confidence = 1.0
        if 'confidence' in result[0]:
            confidence = result[0]['confidence']
        elif 'avg_logprob' in result[0]:
            # 将对数概率转换为置信度
            confidence = min(1.0, max(0.0, (result[0]['avg_logprob'] + 1.0)))
        
        # 使用错误修正器对清理后的文本进行修正
        corrected_text, corrections, is_noise = error_corrector.correct_text(
            clean_text,  # 使用清理后的文本
            prev_text=prev_text,
            confidence=confidence,
            filter_noise=config.filter_noise_words
        )
        
        # 如果检测到噪音,返回空结果
        if is_noise:
            logger.info(f"[噪音过滤] 丢弃噪音文本: '{clean_text}' (原始: '{original_text}')")
            return None  # 返回 None 表示应该丢弃此结果
        
        # 更新结果(使用修正后的文本重新构建带标记的文本)
        if corrected_text != clean_text or clean_text != original_text:
            # 重新构建带标记的文本(保留原始标记)
            # 提取原始标记
            tags = re.findall(r'<\|[^|]+\|>', original_text)
            if tags:
                # 重新组合: 标记 + 修正后的文本
                new_text_with_tags = ''.join(tags) + corrected_text
            else:
                new_text_with_tags = corrected_text
            
            logger.info(f"[ASR错误修正] 原文: '{original_text}'")
            logger.info(f"[ASR错误修正] 清理: '{clean_text}'")
            logger.info(f"[ASR错误修正] 修正: '{corrected_text}'")
            if corrections:
                logger.info(f"[ASR错误修正] 修正详情: {corrections}")
            
            result[0]['text'] = new_text_with_tags
            result[0]['original_text'] = original_text  # 保留原始文本
            result[0]['corrections'] = corrections  # 保存修正记录
        
        # 最终长度检查
        if config.filter_noise_words and len(corrected_text.strip()) < config.min_text_length:
            logger.info(f"[噪音过滤] 文本过短,丢弃: '{corrected_text}'")
            return None
    
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
    client: str | None = None
    speaker: str | None = None  # 说话人标识

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    client_id = None
    try:
        query_params = parse_qs(websocket.scope['query_string'].decode())
        sv = query_params.get('sv', ['false'])[0].lower() in ['true', '1', 't', 'y', 'yes']
        lang = query_params.get('lang', ['auto'])[0].lower()
        # 优先使用请求参数中的 client_id，否则回退到 IP:端口
        client_id = query_params.get('client_id', [None])[0]
        if not client_id and websocket.client:
            client_id = f"{websocket.client.host}:{websocket.client.port}"
        logger.info(f"[client] 建立连接，client_id={client_id}")
        
        await websocket.accept()
        chunk_size = int(config.chunk_size_ms * config.sample_rate / 1000)
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)

        cache = {}
        cache_asr = {}
        # 说话人声纹特征字典 {speaker_id: embedding}
        speaker_embeddings = {}
        # 当前说话人ID
        current_speaker = None
        # 记录语音段的起始时间戳
        segment_start_time = None
        last_vad_beg = last_vad_end = -1
        offset = 0
        # 用于上下文修正的前一句文本
        prev_text = ""
        
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
                    
                # 旧的 sv 参数验证逻辑已移除，现在统一使用 identify_speaker() 进行说话人识别

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
                                logger.info(f"[segment] 检测到语音段起点，起始采样点={start_sample}，时间戳={segment_start_time:.3f}")                           
                        if segment[1] > -1: # speech end
                            last_vad_end = segment[1]
                        if last_vad_beg > -1 and last_vad_end > -1:
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            beg = int(last_vad_beg * config.sample_rate / 1000)
                            end = int(last_vad_end * config.sample_rate / 1000)
                            logger.info(f"[vad segment] audio_len: {end - beg}")
                            
                            # 说话人识别
                            segment_audio = audio_vad[beg:end]
                            speaker_id, is_new_speaker = identify_speaker(
                                segment_audio, 
                                speaker_embeddings, 
                                config.sd_thr
                            )
                            
                            # 如果识别到新说话人,发送通知
                            if is_new_speaker and speaker_id is not None:
                                try:
                                    new_speaker_response = TranscriptionResponse(
                                        code=4,
                                        info="new_speaker_detected",
                                        data=speaker_id,
                                        client=client_id,
                                        speaker=speaker_id
                                    )
                                    logger.info(f"[client] 向客户端发送：{new_speaker_response.model_dump()}")
                                    await websocket.send_json(new_speaker_response.model_dump())
                                except Exception as e:
                                    logger.debug(f"send new_speaker event failed: {e}")
                            
                            current_speaker = speaker_id
                            
                            # 执行 ASR 识别(启用错误修正和降噪)
                            result = asr(
                                segment_audio, 
                                lang.strip(), 
                                cache_asr, 
                                use_itn=True,
                                prev_text=prev_text,
                                enable_correction=config.enable_error_correction,
                                enable_denoise=config.enable_audio_denoise
                            )
                            segment_elapsed_ms = None
                            if segment_start_time is not None:
                                segment_elapsed_ms = (time.time() - segment_start_time) * 1000
                            audio_vad = audio_vad[end:]
                            last_vad_beg = last_vad_end = -1
                            segment_start_time = None
                            logger.info("[segment] 语音段结束，已重置状态")
                            if segment_elapsed_ms is not None:
                                logger.info(f"[segment] 语音段总耗时：{segment_elapsed_ms:.2f} 毫秒")
                            
                            # 检查 ASR 结果是否有效(可能被噪音过滤器丢弃)
                            if result is None:
                                logger.info("[segment] ASR结果为空(已被噪音过滤),跳过此段")
                                continue
                            
                            logger.info(f"asr response: {result}")
                            
                            # 更新前一句文本(用于下一次上下文修正)
                            if 'text' in result[0]:
                                prev_text = result[0]['text']
                            
                            response = TranscriptionResponse(
                                code=0,
                                info=json.dumps(result[0], ensure_ascii=False),
                                data=format_str_v3(result[0]['text']),
                                client=client_id,
                                speaker=current_speaker  # 添加说话人信息
                            )
                            logger.info(f"[client] 向客户端发送：{response.model_dump()}")
                            await websocket.send_json(response.model_dump())
                                
                        # logger.debug(f'last_vad_beg: {last_vad_beg}; last_vad_end: {last_vad_end} len(audio_vad): {len(audio_vad)}')

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected, client_id={client_id}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}\nCall stack:\n{traceback.format_exc()}")
        await websocket.close()
    finally:
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)
        cache.clear()
        logger.info(f"Cleaned up resources after WebSocket disconnect, client_id={client_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=27100, help='Port number to run the FastAPI app on.')
    # parser.add_argument('--certfile', type=str, default='path_to_your_SSL_certificate_file.crt', help='SSL certificate file')
    # parser.add_argument('--keyfile', type=str, default='path_to_your_SSL_certificate_file.key', help='SSL key file')
    args = parser.parse_args()
    # uvicorn.run(app, host="0.0.0.0", port=args.port, ssl_certfile=args.certfile, ssl_keyfile=args.keyfile)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
