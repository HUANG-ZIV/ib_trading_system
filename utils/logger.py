"""
Logger 模組 - 日誌配置

使用 loguru 提供統一的日誌管理
"""

import sys
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
import threading

from loguru import logger


class LogLevel(Enum):
    """日誌等級"""
    
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# 預設配置
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_ROTATION = "00:00"  # 每日午夜輪轉
DEFAULT_RETENTION = "30 days"  # 保留 30 天
DEFAULT_COMPRESSION = "zip"  # 壓縮格式

# 控制台格式（帶顏色）
CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# 簡化的控制台格式
CONSOLE_FORMAT_SIMPLE = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

# 檔案格式（完整資訊）
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)

# 錯誤檔案格式（包含額外資訊）
ERROR_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}\n"
    "{exception}"
)

# 全局狀態
_is_configured = False
_config_lock = threading.Lock()
_loggers: Dict[str, Any] = {}


def setup_logger(
    log_dir: str = DEFAULT_LOG_DIR,
    log_level: str = DEFAULT_LOG_LEVEL,
    console_output: bool = True,
    file_output: bool = True,
    error_file: bool = True,
    rotation: str = DEFAULT_ROTATION,
    retention: str = DEFAULT_RETENTION,
    compression: str = DEFAULT_COMPRESSION,
    simple_console: bool = False,
    json_logs: bool = False,
    app_name: str = "ib_trading",
) -> None:
    """
    配置日誌系統
    
    Args:
        log_dir: 日誌目錄
        log_level: 日誌等級
        console_output: 是否輸出到控制台
        file_output: 是否輸出到檔案
        error_file: 是否獨立錯誤日誌檔案
        rotation: 輪轉時間 ("00:00", "1 day", "100 MB")
        retention: 保留時間 ("30 days", "10 files")
        compression: 壓縮格式 ("zip", "gz", "tar.gz")
        simple_console: 是否使用簡化控制台格式
        json_logs: 是否輸出 JSON 格式日誌
        app_name: 應用名稱（用於檔案命名）
    
    使用方式:
        # 基本配置
        setup_logger()
        
        # 自訂配置
        setup_logger(
            log_dir="logs",
            log_level="DEBUG",
            rotation="100 MB",
            retention="7 days",
        )
    """
    global _is_configured
    
    with _config_lock:
        if _is_configured:
            logger.warning("日誌已配置，跳過重複配置")
            return
        
        # 移除預設處理器
        logger.remove()
        
        # 建立日誌目錄
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 添加控制台輸出
        if console_output:
            console_format = CONSOLE_FORMAT_SIMPLE if simple_console else CONSOLE_FORMAT
            logger.add(
                sys.stderr,
                format=console_format,
                level=log_level,
                colorize=True,
                backtrace=True,
                diagnose=True,
            )
        
        # 添加主日誌檔案
        if file_output:
            main_log_file = log_path / f"{app_name}_{{time:YYYY-MM-DD}}.log"
            
            if json_logs:
                logger.add(
                    str(main_log_file),
                    format=FILE_FORMAT,
                    level=log_level,
                    rotation=rotation,
                    retention=retention,
                    compression=compression,
                    serialize=True,  # JSON 格式
                    encoding="utf-8",
                    backtrace=True,
                    diagnose=True,
                )
            else:
                logger.add(
                    str(main_log_file),
                    format=FILE_FORMAT,
                    level=log_level,
                    rotation=rotation,
                    retention=retention,
                    compression=compression,
                    encoding="utf-8",
                    backtrace=True,
                    diagnose=True,
                )
        
        # 添加錯誤日誌檔案
        if error_file:
            error_log_file = log_path / f"{app_name}_error_{{time:YYYY-MM-DD}}.log"
            logger.add(
                str(error_log_file),
                format=ERROR_FORMAT,
                level="ERROR",
                rotation=rotation,
                retention=retention,
                compression=compression,
                encoding="utf-8",
                backtrace=True,
                diagnose=True,
            )
        
        _is_configured = True
        logger.info(f"日誌系統已配置: level={log_level}, dir={log_dir}")


def get_logger(name: str = "") -> "BoundLogger":
    """
    取得帶名稱的 logger
    
    Args:
        name: logger 名稱
        
    Returns:
        BoundLogger 實例
        
    使用方式:
        logger = get_logger("my_module")
        logger.info("Hello world")
    """
    # 確保日誌已配置
    if not _is_configured:
        setup_logger()
    
    # 使用 bind 建立帶名稱的 logger
    if name:
        return logger.bind(name=name)
    return logger


def set_log_level(level: str) -> None:
    """
    動態設定日誌等級
    
    Args:
        level: 日誌等級 ("DEBUG", "INFO", "WARNING", "ERROR")
    """
    # 注意：loguru 不支持直接修改等級
    # 需要重新配置
    logger.warning(f"動態修改日誌等級功能有限，建議重新配置")


def add_file_handler(
    filepath: str,
    level: str = "DEBUG",
    format_string: str = FILE_FORMAT,
    rotation: str = DEFAULT_ROTATION,
    retention: str = DEFAULT_RETENTION,
    compression: str = DEFAULT_COMPRESSION,
) -> int:
    """
    添加額外的檔案處理器
    
    Args:
        filepath: 檔案路徑
        level: 日誌等級
        format_string: 格式字串
        rotation: 輪轉設定
        retention: 保留設定
        compression: 壓縮設定
        
    Returns:
        處理器 ID（可用於移除）
    """
    handler_id = logger.add(
        filepath,
        format=format_string,
        level=level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )
    return handler_id


def remove_handler(handler_id: int) -> None:
    """移除處理器"""
    logger.remove(handler_id)


class BoundLogger:
    """
    帶名稱的 Logger 包裝類
    
    提供與標準 logging 相似的介面
    """
    
    def __init__(self, name: str):
        self._name = name
        self._logger = logger.bind(name=name)
    
    @property
    def name(self) -> str:
        return self._name
    
    def trace(self, message: str, *args, **kwargs) -> None:
        self._logger.trace(message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        self._logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        self._logger.info(message, *args, **kwargs)
    
    def success(self, message: str, *args, **kwargs) -> None:
        self._logger.success(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        self._logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        self._logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        self._logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs) -> None:
        self._logger.exception(message, *args, **kwargs)
    
    def log(self, level: str, message: str, *args, **kwargs) -> None:
        self._logger.log(level, message, *args, **kwargs)
    
    def bind(self, **kwargs) -> "BoundLogger":
        """綁定額外的上下文"""
        new_logger = BoundLogger(self._name)
        new_logger._logger = self._logger.bind(**kwargs)
        return new_logger
    
    def opt(self, **kwargs):
        """設定選項"""
        return self._logger.opt(**kwargs)


class LoggerAdapter:
    """
    標準 logging 適配器
    
    用於需要標準 logging.Logger 介面的第三方庫
    """
    
    def __init__(self, name: str):
        self._name = name
        self._logger = logger.bind(name=name)
    
    def setLevel(self, level) -> None:
        pass  # loguru 不使用此方法
    
    def addHandler(self, handler) -> None:
        pass  # loguru 不使用此方法
    
    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg, *args, **kwargs):
        self._logger.exception(msg, *args, **kwargs)


def intercept_standard_logging(level: str = "DEBUG") -> None:
    """
    攔截標準 logging 模組的輸出
    
    將標準 logging 的輸出重定向到 loguru
    """
    import logging
    
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # 取得對應的 loguru 等級
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # 找到呼叫者
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    logging.basicConfig(handlers=[InterceptHandler()], level=0)


# ============================================================
# 便捷函數
# ============================================================

def log_function_call(func):
    """
    裝飾器：記錄函數呼叫
    """
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼叫 {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} 返回: {result}")
            return result
        except Exception as e:
            logger.exception(f"{func.__name__} 發生錯誤: {e}")
            raise
    
    return wrapper


def log_execution_time(func):
    """
    裝飾器：記錄執行時間
    """
    from functools import wraps
    import time
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} 執行時間: {elapsed:.4f}s")
        return result
    
    return wrapper


# ============================================================
# 模組初始化
# ============================================================

# 提供直接的 logger 存取
__all__ = [
    "setup_logger",
    "get_logger",
    "set_log_level",
    "add_file_handler",
    "remove_handler",
    "LogLevel",
    "BoundLogger",
    "LoggerAdapter",
    "intercept_standard_logging",
    "log_function_call",
    "log_execution_time",
    "logger",  # 直接匯出 loguru logger
]