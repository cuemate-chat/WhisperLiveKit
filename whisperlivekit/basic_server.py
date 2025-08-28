from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from whisperlivekit import TranscriptionEngine, AudioProcessor, get_web_interface_html, parse_args
import asyncio
import logging
import logging.handlers
from starlette.staticfiles import StaticFiles
import pathlib
import whisperlivekit.web as webpkg
import os
from datetime import datetime, timezone, timedelta
import sys
import warnings

# 设置东八区时区
CHINA_TZ = timezone(timedelta(hours=8))

def get_china_time():
    """获取东八区当前时间"""
    return datetime.now(CHINA_TZ)

# 设置日志格式和基础配置
def setup_file_logging(service_name: str = None):
    """设置文件日志输出，按照 CueMate 项目的日志格式"""
    if service_name is None:
        # 根据环境变量确定服务名称
        if os.environ.get('ASR_SERVICE_TYPE') == 'interviewer':
            service_name = "asr-interviewer"
        else:
            service_name = "asr-user"
    
    log_base_dir = os.environ.get('CUEMATE_LOG_DIR', '/opt/cuemate/logs')
    today = get_china_time().strftime('%Y-%m-%d')
    
    # 为不同级别创建文件处理器
    levels = ['debug', 'info', 'warn', 'error']
    handlers = []
    
    for level_name in levels:
        level_dir = os.path.join(log_base_dir, level_name, service_name, today)
        os.makedirs(level_dir, exist_ok=True)
        
        log_file = os.path.join(level_dir, f'{level_name}.log')
        
        # 创建文件处理器
        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # 设置日志级别过滤
        if level_name == 'debug':
            handler.setLevel(logging.DEBUG)
        elif level_name == 'info':
            handler.setLevel(logging.INFO)
        elif level_name == 'warn':
            handler.setLevel(logging.WARNING)
        elif level_name == 'error':
            handler.setLevel(logging.ERROR)
            
        handlers.append(handler)
    
    return handlers

# 获取服务名称（根据环境变量 ASR_SERVICE_TYPE 判断）
if os.environ.get('ASR_SERVICE_TYPE') == 'interviewer':
    service_name = "asr-interviewer"
else:
    service_name = "asr-user"

# 设置控制台输出
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# 设置文件日志输出
file_handlers = setup_file_logging(service_name)

# 获取根日志记录器并添加文件处理器
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

for handler in file_handlers:
    root_logger.addHandler(handler)

# 设置应用日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 捕获 warnings 到日志
warnings.simplefilter("default")
logging.captureWarnings(True)

# 全局捕获未处理异常并写入日志文件
def _global_excepthook(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # 让 KeyboardInterrupt 走默认行为，便于优雅退出
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.getLogger("unhandled_exception").error(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )

sys.excepthook = _global_excepthook

# 捕获 asyncio 事件循环中的未处理异常
def _asyncio_exception_handler(loop, context):
    message = context.get("message")
    exception = context.get("exception")
    if exception is not None:
        logging.getLogger("asyncio").error(message or "Asyncio exception", exc_info=exception)
    else:
        logging.getLogger("asyncio").error(message or f"Asyncio error context: {context}")

try:
    asyncio.get_event_loop().set_exception_handler(_asyncio_exception_handler)
except RuntimeError:
    # 在新事件循环创建前设置可能抛错，忽略并在 lifespan 中由 uvicorn 管理
    pass



# 记录启动信息
logger.info(f"WhisperLiveKit {service_name} starting up")
_log_base_dir_display = os.environ.get('CUEMATE_LOG_DIR', '/opt/cuemate/logs')
logger.info(
    "Log files will be written to: %s/[level]/%s/%s/[level].log",
    _log_base_dir_display,
    service_name,
    get_china_time().strftime('%Y-%m-%d'),
)

args = parse_args()
transcription_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
    
    # 重新配置 uvicorn 日志记录器，让它们也写入到我们的日志文件
    import logging
    
    # 为 uvicorn 日志记录器添加文件处理器，避免重复
    uvicorn_loggers = ['uvicorn', 'uvicorn.error', 'uvicorn.access']
    
    for logger_name in uvicorn_loggers:
        logger_obj = logging.getLogger(logger_name)
        
        # 清除现有的文件处理器，避免重复
        for handler in logger_obj.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger_obj.removeHandler(handler)
        
        # 添加 debug 级别文件处理器
        debug_dir = os.path.join(
            os.environ.get('CUEMATE_LOG_DIR', '/opt/cuemate/logs'),
            'debug', service_name, get_china_time().strftime('%Y-%m-%d')
        )
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, 'debug.log')
        
        debug_handler = logging.FileHandler(debug_file, encoding='utf-8')
        debug_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        debug_handler.setLevel(logging.DEBUG)
        logger_obj.addHandler(debug_handler)
        
        # 添加 info 级别文件处理器
        info_dir = os.path.join(
            os.environ.get('CUEMATE_LOG_DIR', '/opt/cuemate/logs'),
            'info', service_name, get_china_time().strftime('%Y-%m-%d')
        )
        os.makedirs(info_dir, exist_ok=True)
        info_file = os.path.join(info_dir, 'info.log')
        
        info_handler = logging.FileHandler(info_file, encoding='utf-8')
        info_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        info_handler.setLevel(logging.INFO)
        logger_obj.addHandler(info_handler)
        
        # 添加 warn 级别文件处理器
        warn_dir = os.path.join(
            os.environ.get('CUEMATE_LOG_DIR', '/opt/cuemate/logs'),
            'warn', service_name, get_china_time().strftime('%Y-%m-%d')
        )
        os.makedirs(warn_dir, exist_ok=True)
        warn_file = os.path.join(warn_dir, 'warn.log')
        
        warn_handler = logging.FileHandler(warn_file, encoding='utf-8')
        warn_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        warn_handler.setLevel(logging.WARNING)
        logger_obj.addHandler(warn_handler)
        
        # 添加 error 级别文件处理器
        error_dir = os.path.join(
            os.environ.get('CUEMATE_LOG_DIR', '/opt/cuemate/logs'),
            'error', service_name, get_china_time().strftime('%Y-%m-%d')
        )
        os.makedirs(error_dir, exist_ok=True)
        error_file = os.path.join(error_dir, 'error.log')
        
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        error_handler.setLevel(logging.ERROR)
        logger_obj.addHandler(error_handler)
        
        logger_obj.setLevel(logging.DEBUG)
        logger_obj.propagate = False  # 防止传播到根日志记录器
    
    transcription_engine = TranscriptionEngine(
        **vars(args),
    )
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
web_dir = pathlib.Path(webpkg.__file__).parent
app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")

@app.get("/")
async def get():
    return HTMLResponse(get_web_interface_html())

@app.get("/config")
async def get_config():
    """获取当前转录引擎的配置"""
    global transcription_engine
    if transcription_engine is None:
        return {"error": "Transcription engine not initialized"}
    
    # 将 Namespace 转换为字典
    config = vars(transcription_engine.args)
    
    # 过滤掉一些内部参数，只返回可配置的参数
    filterable_keys = {
        'host', 'port', 'warmup_file', 'ssl_certfile', 'ssl_keyfile', 
        'model_cache_dir', 'model_dir', 'model_path', 'cif_ckpt_path',
        'segmentation_model', 'embedding_model'
    }
    
    # 创建返回的配置字典
    public_config = {k: v for k, v in config.items() if k not in filterable_keys}
    
    return {"config": public_config}

@app.post("/config")
async def update_config(new_config: dict):
    """更新转录引擎配置（需要重启生效）"""
    global transcription_engine
    if transcription_engine is None:
        return {"error": "Transcription engine not initialized"}
    
    try:
        # 获取当前配置
        current_config = vars(transcription_engine.args)
        
        # 验证并更新可修改的配置项
        updatable_params = {
            'diarization', 'punctuation_split', 'min_chunk_size', 'model', 
            'lan', 'task', 'backend', 'vac', 'vac_chunk_size', 'log_level',
            'transcription', 'vad', 'buffer_trimming', 'confidence_validation', 
            'buffer_trimming_sec', 'frame_threshold', 'beams', 'decoder_type',
            'audio_max_len', 'audio_min_len', 'never_fire', 'init_prompt',
            'static_init_prompt', 'max_context_tokens', 'diarization_backend'
        }
        
        updated_params = {}
        for key, value in new_config.items():
            if key in updatable_params:
                # 更新配置
                setattr(transcription_engine.args, key, value)
                updated_params[key] = value
        
        return {
            "success": True,
            "message": f"配置已更新，共更新 {len(updated_params)} 个参数",
            "updated_params": updated_params,
            "note": "配置已保存到内存中，部分参数可能需要重启服务才能生效"
        }
        
    except Exception as e:
        logger.error(f"更新配置失败: {e}", exc_info=True)
        return {"error": f"更新配置失败: {str(e)}"}


async def handle_websocket_results(websocket, results_generator):
    """Consumes results from the audio processor and sends them via WebSocket."""
    result_count = 0
    try:
        async for response in results_generator:
            result_count += 1
            logger.debug(f"Sending result #{result_count}: {response}")
            await websocket.send_json(response)
        # when the results_generator finishes it means all audio has been processed
        logger.info(f"Results generator finished after {result_count} responses. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected while handling results after {result_count} responses (client likely closed connection).")
    except Exception as e:
        logger.error(f"Error in WebSocket results handler after {result_count} responses: {e}", exc_info=True)


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    client_ip = websocket.client.host if websocket.client else "unknown"
    logger.info(f"New WebSocket connection from {client_ip}")
    
    global transcription_engine
    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
    )
    
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection accepted for client {client_ip}")
            
        results_generator = await audio_processor.create_tasks()
        websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

        message_count = 0
        bytes_received = 0
        
        while True:
            message = await websocket.receive_bytes()
            message_count += 1
            bytes_received += len(message)
            
            if len(message) == 0:
                logger.info(f"Received empty message #{message_count} from {client_ip} - end of audio signal")
            else:
                logger.debug(f"Received audio message #{message_count} from {client_ip}: {len(message)} bytes")
                
            await audio_processor.process_audio(message)
            
    except KeyError as e:
        if 'bytes' in str(e):
            logger.warning(f"Client {client_ip} has closed the connection after {message_count} messages, {bytes_received} bytes.")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint for {client_ip}: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected by client {client_ip} during message receiving loop after {message_count} messages, {bytes_received} bytes.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop for {client_ip}: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaning up WebSocket endpoint for client {client_ip}...")
        if 'websocket_task' in locals() and not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")
            
        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")

def main():
    """Entry point for the CLI command."""
    # 在 main 函数中初始化日志，确保在 uvicorn 启动前执行
    global service_name
    
    # 重新获取服务名称（确保使用最新的环境变量）
    if os.environ.get('ASR_SERVICE_TYPE') == 'interviewer':
        service_name = "asr-interviewer"  
    else:
        service_name = "asr-user"
    
    # 设置文件日志输出
    file_handlers = setup_file_logging(service_name)
    root_logger = logging.getLogger()
    for handler in file_handlers:
        root_logger.addHandler(handler)
    
    # 记录启动信息
    logger.info(f"WhisperLiveKit {service_name} starting up in main()")
    logger.info(f"Log files will be written to: {os.environ.get('CUEMATE_LOG_DIR', '/opt/cuemate/logs')}/[level]/{service_name}/{get_china_time().strftime('%Y-%m-%d')}/[level].log")
    
    # 在 uvicorn 启动前记录启动信息
    logger.info("Starting uvicorn server...")
    
    import uvicorn
    
    # 配置 uvicorn 日志，使用我们的文件处理器
    uvicorn_log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "file_info": {
                "formatter": "default",
                "class": "logging.FileHandler",
                "filename": f"/opt/cuemate/logs/info/{service_name}/{get_china_time().strftime('%Y-%m-%d')}/info.log",
                "encoding": "utf-8",
                "level": "INFO",
            },
            "file_error": {
                "formatter": "default", 
                "class": "logging.FileHandler",
                "filename": f"/opt/cuemate/logs/error/{service_name}/{get_china_time().strftime('%Y-%m-%d')}/error.log",
                "encoding": "utf-8",
                "level": "ERROR",
            },
        },
        "loggers": {
            "uvicorn": {
                "level": "INFO",
                "handlers": ["default", "file_info"],
                "propagate": False,
            },
            "uvicorn.error": {
                "level": "ERROR", 
                "handlers": ["default", "file_error"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["default", "file_info"],
                "propagate": False,
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["default", "file_info"],
        },
    }
    
    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host":args.host, 
        "port":args.port, 
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
        "log_config": None,  # 完全禁用 uvicorn 的日志配置
        "access_log": False,  # 禁用访问日志
    }
    
    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}

    uvicorn.run(**uvicorn_kwargs)

if __name__ == "__main__":
    main()
