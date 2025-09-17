from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from whisperlivekit import TranscriptionEngine, AudioProcessor, parse_args
from whisperlivekit.cuemate_logger import get_logger, get_service_name, setup_exception_handlers
import asyncio
import os
import sys
import warnings

# 使用 CueMate 统一日志系统
service_name = get_service_name()
logger = get_logger(__name__)

# 设置全局异常处理器
setup_exception_handlers(logger)

# 捕获 warnings 到日志
warnings.simplefilter("default")
import logging
logging.captureWarnings(True)

# 记录启动信息
logger.info(f"{service_name} starting up")
from whisperlivekit.cuemate_logger import get_china_time
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
    
    #to remove after 0.2.8
    if args.backend == "simulstreaming" and not args.disable_fast_encoder:
        logger.warning(f"""
{'='*50}
WhisperLiveKit 0.2.8 has introduced a new fast encoder feature using MLX Whisper or Faster Whisper for improved speed. Use --disable-fast-encoder to disable if you encounter issues.
{'='*50}
    """)
    
    global transcription_engine
    
    # 为 uvicorn 使用统一的日志配置
    from whisperlivekit.cuemate_logger import create_logger_for_service

    uvicorn_loggers = ['uvicorn', 'uvicorn.error', 'uvicorn.access']
    for logger_name in uvicorn_loggers:
        # 为每个 uvicorn logger 创建统一配置
        create_logger_for_service(service_name, logger_name)
    
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
@app.get("/")
async def get():
    return {"message": "WhisperLiveKit ASR Service", "service": service_name, "endpoints": ["/asr", "/config"]}

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
            await websocket.send_json(response.to_dict())
            result_count += 1
            logger.debug(f"Sending result #{result_count}: {response}")
        # when the results_generator finishes it means all audio has been processed
        logger.info(f"Results generator finished after {result_count} responses. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected while handling results after {result_count} responses (client likely closed connection).")
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")
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
    logger.info(f"{service_name} starting up in main()")
    logger.info("Starting uvicorn server...")

    import uvicorn
    
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
