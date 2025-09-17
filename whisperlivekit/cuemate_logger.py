"""
CueMate 统一日志系统
按照 CueMate 标准配置 logging，支持按日期和服务分类的日志文件结构
"""
import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_china_time():
    """获取中国时间"""
    import pytz
    china_tz = pytz.timezone('Asia/Shanghai')
    return datetime.now(china_tz)


def create_logger_for_service(service_name: str, logger_name: Optional[str] = None) -> logging.Logger:
    """
    为指定服务创建标准化的 logger

    Args:
        service_name: 服务名称，如 'asr-user', 'asr-interviewer'
        logger_name: logger 名称，如果为 None 则使用调用文件的 __name__

    Returns:
        配置好的 logger 实例
    """
    # 获取日志根目录
    log_base_dir = os.environ.get('CUEMATE_LOG_DIR', '/opt/cuemate/logs')

    # 获取当前日期
    current_date = get_china_time().strftime('%Y-%m-%d')

    # 确定 logger 名称
    if logger_name is None:
        import inspect
        frame = inspect.currentframe().f_back
        logger_name = frame.f_globals.get('__name__', 'unknown')

    # 创建 logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # 清除已有的 handlers 避免重复
    logger.handlers.clear()

    # 创建不同级别的日志处理器
    levels = [
        ('debug', logging.DEBUG),
        ('info', logging.INFO),
        ('warn', logging.WARNING),
        ('error', logging.ERROR)
    ]

    for level_name, level_value in levels:
        # 创建日志目录
        log_dir = Path(log_base_dir) / level_name / service_name / current_date
        log_dir.mkdir(parents=True, exist_ok=True)

        # 创建日志文件路径
        log_file = log_dir / f"{level_name}.log"

        # 创建文件处理器
        handler = logging.FileHandler(log_file, encoding='utf-8')

        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        # 设置级别过滤
        handler.setLevel(level_value)
        handler.addFilter(lambda record, target_level=level_value: record.levelno == target_level)

        # 添加到 logger
        logger.addHandler(handler)

    # 添加控制台输出处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def get_service_name() -> str:
    """
    根据环境变量自动获取服务名称
    """
    if os.environ.get('ASR_SERVICE_TYPE') == 'interviewer':
        return "asr-interviewer"
    else:
        return "asr-user"


def get_logger(logger_name: Optional[str] = None) -> logging.Logger:
    """
    获取或创建当前服务的 logger
    这是主要的对外接口函数，自动根据环境变量判断服务名称

    Args:
        logger_name: logger 名称，如果为 None 则使用调用文件的 __name__

    Returns:
        配置好的 logger 实例
    """
    service_name = get_service_name()
    return create_logger_for_service(service_name, logger_name)


def setup_exception_handlers(logger: logging.Logger):
    """设置全局异常处理器"""
    import sys
    import asyncio

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    def handle_asyncio_exception(loop, context):
        exception = context.get('exception')
        if exception:
            logger.error(f"Asyncio exception: {context.get('message', '')}", exc_info=exception)
        else:
            logger.error(f"Asyncio error context: {context}")

    sys.excepthook = handle_exception
    asyncio.get_event_loop().set_exception_handler(handle_asyncio_exception)