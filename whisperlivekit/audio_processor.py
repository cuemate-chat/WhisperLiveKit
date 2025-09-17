import asyncio
import numpy as np
from time import time
import logging
import traceback
from datetime import timedelta
from whisperlivekit.timed_objects import ASRToken, Silence
from whisperlivekit.core import TranscriptionEngine, online_factory

try:
    from whisperlivekit.remove_silences import handle_silences
except ImportError:
    def handle_silences(*args, **kwargs):
        """Fallback silence handler"""
        pass

try:
    from whisperlivekit.silero_vad_iterator import FixedVADIterator
except ImportError:
    FixedVADIterator = None

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SENTINEL = object()  # unique sentinel object for end of stream marker

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

class AudioProcessor:
    """简化的音频处理器，专注于PCM音频转录"""

    def __init__(self, **kwargs):
        if 'transcription_engine' in kwargs and isinstance(kwargs['transcription_engine'], TranscriptionEngine):
            models = kwargs['transcription_engine']
        else:
            models = TranscriptionEngine(**kwargs)

        # 音频处理设置
        self.args = models.args
        self.sample_rate = 16000
        self.channels = 1
        self.samples_per_sec = int(self.sample_rate * self.args.min_chunk_size)
        self.bytes_per_sample = 2
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz
        self.debug = False

        # 状态管理
        self.is_stopping = False
        self.silence = False
        self.tokens = []
        self.translated_segments = []
        self.buffer_transcription = ""
        self.buffer_diarization = ""
        self.end_buffer = 0
        self.end_attributed_speaker = 0
        self.sep = " "
        self.beg_loop = time()
        self.lock = asyncio.Lock()

        # 模型设置
        self.asr = models.asr
        self.tokenizer = models.tokenizer
        self.diarization = models.diarization
        self.vac_model = models.vac_model
        if self.args.vac and FixedVADIterator:
            self.vac = FixedVADIterator(models.vac_model)
        else:
            self.vac = None

        self.transcription_queue = asyncio.Queue() if self.args.transcription else None
        self.diarization_queue = asyncio.Queue() if self.args.diarization else None
        self.translation_queue = asyncio.Queue() if self.args.target_language else None
        self.pcm_buffer = bytearray()

        # Task references
        self.transcription_task = None
        self.diarization_task = None
        self.watchdog_task = None
        self.all_tasks_for_cleanup = []

        # Initialize transcription engine if enabled
        if self.args.transcription:
            self.online = online_factory(self.args, models.asr, models.tokenizer)

    def convert_pcm_to_float(self, pcm_buffer):
        """将PCM缓冲区转换为标准化的NumPy数组"""
        return np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0

    async def update_transcription(self, new_tokens, buffer, end_buffer, sep):
        """Thread-safe update of transcription with new data."""
        async with self.lock:
            self.tokens.extend(new_tokens)
            self.buffer_transcription = buffer
            self.end_buffer = end_buffer
            self.sep = sep

    async def update_diarization(self, end_attributed_speaker, buffer_diarization=""):
        """Thread-safe update of diarization with new data."""
        async with self.lock:
            self.end_attributed_speaker = end_attributed_speaker
            if buffer_diarization:
                self.buffer_diarization = buffer_diarization

    async def add_dummy_token(self):
        """Placeholder token when no transcription is available."""
        async with self.lock:
            current_time = time() - self.beg_loop
            self.tokens.append(ASRToken(
                start=current_time, end=current_time + 1,
                text=".", speaker=-1, is_dummy=True
            ))

    async def get_current_state(self):
        """Get current state."""
        async with self.lock:
            current_time = time()

            # Calculate remaining times
            remaining_transcription = 0
            if self.end_buffer > 0:
                remaining_transcription = max(0, round(current_time - self.beg_loop - self.end_buffer, 1))

            remaining_diarization = 0
            if self.tokens:
                latest_end = max(self.end_buffer, self.tokens[-1].end if self.tokens else 0)
                remaining_diarization = max(0, round(latest_end - self.end_attributed_speaker, 1))

            return {
                "tokens": self.tokens.copy(),
                "buffer_transcription": self.buffer_transcription,
                "buffer_diarization": self.buffer_diarization,
                "end_buffer": self.end_buffer,
                "end_attributed_speaker": self.end_attributed_speaker,
                "sep": self.sep,
                "remaining_time_transcription": remaining_transcription,
                "remaining_time_diarization": remaining_diarization
            }

    async def reset(self):
        """Reset all state variables to initial values."""
        async with self.lock:
            self.tokens = []
            self.buffer_transcription = self.buffer_diarization = ""
            self.end_buffer = self.end_attributed_speaker = 0
            self.beg_loop = time()

    async def process_pcm_chunks(self):
        """处理PCM音频数据块的主要循环"""
        buffer_size = self.bytes_per_sec

        while not self.is_stopping:
            try:
                if len(self.pcm_buffer) >= buffer_size:
                    # 提取音频块
                    chunk = bytes(self.pcm_buffer[:buffer_size])
                    self.pcm_buffer = self.pcm_buffer[buffer_size:]

                    # 转换为浮点数数组
                    audio_chunk = self.convert_pcm_to_float(chunk)

                    # 处理转录
                    if self.args.transcription and self.transcription_queue:
                        await self.transcription_queue.put(audio_chunk)

                    # 处理说话人分离
                    if self.args.diarization and self.diarization_queue:
                        await self.diarization_queue.put(audio_chunk)
                else:
                    # 等待更多数据
                    await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"PCM处理错误: {e}")
                await asyncio.sleep(0.1)

    async def transcription_worker(self):
        """转录工作器"""
        if not self.args.transcription:
            return

        logger.info("转录工作器启动")

        while not self.is_stopping:
            try:
                # 从队列获取音频数据
                audio_chunk = await asyncio.wait_for(
                    self.transcription_queue.get(), timeout=1.0
                )

                # 使用在线转录引擎处理
                if hasattr(self, 'online') and self.online:
                    new_tokens = self.online.process(audio_chunk)
                    if new_tokens:
                        current_time = time() - self.beg_loop
                        await self.update_transcription(
                            new_tokens,
                            " ".join([t.text for t in new_tokens]),
                            current_time,
                            " "
                        )

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"转录工作器错误: {e}")
                await asyncio.sleep(0.1)

    async def diarization_worker(self):
        """说话人分离工作器"""
        if not self.args.diarization:
            return

        logger.info("说话人分离工作器启动")

        while not self.is_stopping:
            try:
                # 从队列获取音频数据
                audio_chunk = await asyncio.wait_for(
                    self.diarization_queue.get(), timeout=1.0
                )

                # 这里可以添加说话人分离逻辑
                # 暂时使用简单的时间戳
                current_time = time() - self.beg_loop
                await self.update_diarization(current_time)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"说话人分离工作器错误: {e}")
                await asyncio.sleep(0.1)

    async def format_output(self):
        """格式化输出结果"""
        state = await self.get_current_state()

        # 简单返回当前状态
        return (
            [],  # lines
            [token.text for token in state["tokens"]],  # undiarized_text
            state["buffer_transcription"],
            state["buffer_diarization"]
        )

    async def start(self):
        """启动音频处理器"""
        logger.info("启动PCM音频处理器")
        self.is_stopping = False
        self.beg_loop = time()

        # 启动工作任务
        tasks = []

        if self.args.transcription:
            self.transcription_task = asyncio.create_task(self.transcription_worker())
            tasks.append(self.transcription_task)

        if self.args.diarization:
            self.diarization_task = asyncio.create_task(self.diarization_worker())
            tasks.append(self.diarization_task)

        # 启动PCM处理任务
        pcm_task = asyncio.create_task(self.process_pcm_chunks())
        tasks.append(pcm_task)

        self.all_tasks_for_cleanup.extend(tasks)

        return True

    async def stop(self):
        """停止音频处理器"""
        logger.info("停止PCM音频处理器")
        self.is_stopping = True

        # 取消所有任务
        for task in self.all_tasks_for_cleanup:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.all_tasks_for_cleanup.clear()

    async def write_data(self, message):
        """接收PCM音频数据"""
        if isinstance(message, bytes):
            self.pcm_buffer.extend(message)
            logger.debug(f"接收PCM数据: {len(message)} 字节, 缓冲区总大小: {len(self.pcm_buffer)}")
            return True
        return False