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
from datetime import datetime
import sys
import warnings

# 设置日志格式和基础配置
def setup_file_logging(service_name: str = "asr-service"):
    """设置文件日志输出，按照 CueMate 项目的日志格式"""
    log_base_dir = os.environ.get('CUEMATE_LOG_DIR', '/opt/cuemate/logs')
    today = datetime.now().strftime('%Y-%m-%d')
    
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

# 获取服务名称（从端口号推断）
service_name = "asr-service"
for arg in sys.argv:
    if arg == "--port" and sys.argv.index(arg) + 1 < len(sys.argv):
        port = sys.argv[sys.argv.index(arg) + 1]
        if port == "8000":
            # 从环境变量或其他方式确定是 user 还是 interviewer
            if os.environ.get('ASR_SERVICE_TYPE') == 'interviewer':
                service_name = "asr-interviewer"
            else:
                service_name = "asr-user"
        break

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
    datetime.now().strftime('%Y-%m-%d'),
)

args = parse_args()
transcription_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
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
    import uvicorn
    
    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host":args.host, 
        "port":args.port, 
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
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
