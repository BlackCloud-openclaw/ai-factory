import gzip
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    Fore = None
    Style = None


def gzip_compress(filepath: str, output_path: Optional[str] = None) -> str:
    """Compress a file with gzip.

    Args:
        filepath: Path to the file to compress
        output_path: Optional output path (default: filepath + ".gz")

    Returns:
        Path to the compressed file
    """
    if output_path is None:
        output_path = filepath + ".gz"

    with open(filepath, "rb") as f_in:
        with gzip.open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(filepath)
    return output_path


_LOG_COLORS = {}
if HAS_COLORAMA:
    _LOG_COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }


def _get_color(levelname: str) -> str:
    if not HAS_COLORAMA:
        return ""
    return _LOG_COLORS.get(levelname, "")


def _build_color_formatter() -> logging.Formatter:
    if not HAS_COLORAMA:
        return logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)-8s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    class ColorFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            color = _get_color(record.levelname)
            record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
            return super().format(record)

    return ColorFormatter(
        "%(asctime)s - %(name)s - %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class JsonFormatter(logging.Formatter):
    """JSON 格式日志输出，用于文件日志。"""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "agent"):
            log_data["agent"] = record.agent
        if hasattr(record, "duration"):
            log_data["duration"] = record.duration
        if hasattr(record, "status"):
            log_data["status"] = record.status
        if hasattr(record, "step"):
            log_data["step"] = record.step
        if hasattr(record, "task_id"):
            log_data["task_id"] = record.task_id
        if record.exc_info and record.exc_info[1]:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)


class CompressedRotatingFileHandler(RotatingFileHandler):
    """Rotating file handler that compresses old log files with gzip."""

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None

        if self.backupCount == 0:
            return

        # Rotate backup files (shift .N to .N+1)
        for i in range(self.backupCount - 1, 0, -1):
            src = self.rotation_filename(f"{self.baseFilename}.{i}")
            dst = self.rotation_filename(f"{self.baseFilename}.{i + 1}")

            src_gz = src + ".gz"
            dst_gz = dst + ".gz"

            if os.path.exists(src_gz):
                if os.path.exists(dst_gz):
                    os.remove(dst_gz)
                shutil.move(src_gz, dst_gz)
            elif os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)

        # Compress the first backup (baseFilename.1)
        first_backup = self.rotation_filename(self.baseFilename)
        if os.path.exists(first_backup):
            gz_path = first_backup + ".gz"
            if os.path.exists(gz_path):
                os.remove(gz_path)
            gzip_compress(first_backup, gz_path)

        if not self.delay:
            self.stream = self._open()


def setup_logging(
    name: str = "ai_factory",
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_max_bytes: Optional[int] = None,
    log_backup_count: Optional[int] = None,
    log_compress: bool = True,
) -> logging.Logger:
    """设置日志，同时输出到控制台和文件。

    Args:
        name: Logger 名称
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR)，默认 INFO
        log_file: 日志文件路径，默认 logs/ai_factory.log
        log_max_bytes: 单个日志文件最大字节数，默认 10MB
        log_backup_count: 保留的备份文件数，默认 5
        log_compress: 是否压缩旧日志文件

    Returns:
        配置好的 Logger 实例
    """
    from src.config import config

    logger = logging.getLogger(name)

    # Phase 16.0: 确保 root logger 也有 file handler。
    # 项目中 53 个模块直接用 logging.getLogger(__name__) 而没有调用 setup_logging，
    # 它们的 logger.info/error 因 root logger 无 handler 而全部丢失。
    _log_file = log_file or config.log_file
    _log_max_bytes = log_max_bytes or config.log_max_bytes
    _log_backup_count = log_backup_count or config.log_backup_count
    _log_path = Path(_log_file)
    _log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if not getattr(root, "_ai_factory_handler_installed", False):
        root._ai_factory_handler_installed = True
        root.setLevel(logging.DEBUG)
        _root_fh = CompressedRotatingFileHandler(
            filename=str(_log_path),
            maxBytes=_log_max_bytes,
            backupCount=_log_backup_count,
            encoding="utf-8",
        )
        _root_fh.setLevel(logging.DEBUG)
        _root_fh.setFormatter(JsonFormatter())

        # Phase 16.0: 过滤第三方库 DEBUG 噪音
        class _ThirdPartyFilter(logging.Filter):
            _BLOCK = ("httpcore", "httpx", "urllib3", "asyncio", "openai", "aiohttp", "charset_normalizer")
            def filter(self, record: logging.LogRecord) -> bool:
                return not any(record.name.startswith(p) for p in self._BLOCK)

        _root_fh.addFilter(_ThirdPartyFilter())
        root.addHandler(_root_fh)

    if logger.handlers:
        return logger

    level = getattr(logging, (log_level or config.log_level).upper(), logging.INFO)
    logger.setLevel(level)

    log_file = log_file or config.log_file
    log_max_bytes = log_max_bytes or config.log_max_bytes
    log_backup_count = log_backup_count or config.log_backup_count

    # Console handler (colored)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(_build_color_formatter())
    logger.addHandler(console_handler)

    # File handler (rotating with compression)
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = CompressedRotatingFileHandler(
        filename=str(log_path),
        maxBytes=log_max_bytes,
        backupCount=log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)
    logger.propagate = False  # 避免与 root handler 重复写

    logger.info(f"Logging initialized: level={config.log_level}, file={log_file}")

    return logger
