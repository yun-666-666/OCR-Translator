import atexit
import os
import sys
import tempfile
import threading
import time
from pathlib import Path


DEBUG_LOG_FILENAME = "translator_debug.log"
DEBUG_LOG_MAX_BYTES = 5 * 1024 * 1024
DEBUG_LOG_BACKUP_COUNT = 3

_debug_logging_enabled = True
_writer_registry = {}
_writer_registry_lock = threading.RLock()


class _RotatingTextWriter:
    """Thread-safe, line-buffered text writer with bounded file rotation."""

    def __init__(self, path, max_bytes, backup_count):
        self.path = Path(path)
        self.max_bytes = max(0, int(max_bytes))
        self.backup_count = max(0, int(backup_count))
        self._lock = threading.RLock()
        self._stream = None
        self._size_bytes = None

    def _ensure_open_locked(self):
        if self._stream is not None and not self._stream.closed:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open(
            "a",
            encoding="utf-8-sig",
            buffering=1,
        )
        if self._size_bytes is None:
            self._size_bytes = self.path.stat().st_size

    def _close_locked(self):
        if self._stream is None:
            return
        try:
            self._stream.flush()
            self._stream.close()
        finally:
            self._stream = None

    def _rotate_locked(self):
        self._close_locked()
        if not self.path.exists():
            return
        if self.backup_count == 0:
            self.path.unlink()
            self._size_bytes = 0
            return

        for index in range(self.backup_count - 1, 0, -1):
            source = Path(f"{self.path}.{index}")
            destination = Path(f"{self.path}.{index + 1}")
            if source.exists():
                os.replace(source, destination)
        os.replace(self.path, Path(f"{self.path}.1"))
        self._size_bytes = 0

    def _encoded_write_size(self, text):
        platform_text = text.replace("\n", os.linesep)
        bom_size = 3 if self._size_bytes == 0 else 0
        return bom_size + len(platform_text.encode("utf-8"))

    def _should_rotate_locked(self, incoming_size):
        if self.max_bytes <= 0:
            return False
        if not self._size_bytes:
            return False
        return self._size_bytes + incoming_size > self.max_bytes

    def write(self, text):
        text = str(text)
        if not text:
            return
        with self._lock:
            self._ensure_open_locked()
            incoming_size = self._encoded_write_size(text)
            if self._should_rotate_locked(incoming_size):
                self._rotate_locked()
                self._ensure_open_locked()
                incoming_size = self._encoded_write_size(text)
            self._stream.write(text)
            self._stream.flush()
            self._size_bytes += incoming_size

    def clear(self, marker=""):
        with self._lock:
            self._close_locked()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open(
                "w",
                encoding="utf-8-sig",
                buffering=1,
            )
            self._size_bytes = 0
            if marker:
                marker = str(marker)
                self._stream.write(str(marker))
                self._stream.flush()
                self._size_bytes = self._encoded_write_size(marker)

    def close(self):
        with self._lock:
            self._close_locked()


def _is_test_process():
    main_module = sys.modules.get("__main__")
    main_spec = getattr(main_module, "__spec__", None)
    main_name = str(getattr(main_spec, "name", "") or "").lower()
    executable_name = Path(sys.argv[0]).name.lower() if sys.argv else ""
    return (
        "unittest" in main_name
        or "pytest" in main_name
        or executable_name in {"pytest", "pytest.exe", "py.test", "py.test.exe"}
        or "pytest" in sys.modules
    )


def ensure_test_log_environment():
    """Export the test log directory so child processes inherit isolation."""
    existing = os.environ.get("OCR_TRANSLATOR_LOG_DIR", "").strip()
    if existing:
        return existing
    if not _is_test_process():
        return None

    test_log_dir = (
        Path(tempfile.gettempdir())
        / f"ocr-translator-tests-{os.getpid()}"
    )
    os.environ["OCR_TRANSLATOR_LOG_DIR"] = str(test_log_dir)
    return str(test_log_dir)


def resolve_runtime_log_path(filename, test_process=None):
    """Resolve a runtime log path without mixing test and application sessions."""
    filename_path = Path(filename)
    if filename_path.is_absolute():
        return filename_path

    if test_process is None:
        test_process = _is_test_process()
    if test_process:
        ensure_test_log_environment()

    override_dir = os.environ.get("OCR_TRANSLATOR_LOG_DIR", "").strip()
    if override_dir:
        base_dir = Path(override_dir)
    else:
        if test_process:
            base_dir = (
                Path(tempfile.gettempdir())
                / f"ocr-translator-tests-{os.getpid()}"
            )
        else:
            base_dir = Path.cwd()
    return base_dir / filename_path


def _get_writer(path, max_bytes, backup_count):
    path = Path(path).absolute()
    key = (str(path), int(max_bytes), int(backup_count))
    with _writer_registry_lock:
        writer = _writer_registry.get(key)
        if writer is None:
            writer = _RotatingTextWriter(path, max_bytes, backup_count)
            _writer_registry[key] = writer
        return writer


def append_rotating_text(
    filename,
    text,
    max_bytes=DEBUG_LOG_MAX_BYTES,
    backup_count=DEBUG_LOG_BACKUP_COUNT,
):
    """Append raw text to a bounded runtime log."""
    path = resolve_runtime_log_path(filename)
    writer = _get_writer(path, max_bytes, backup_count)
    writer.write(text)


def close_log_writers():
    """Flush and close every process-local log writer."""
    with _writer_registry_lock:
        writers = list(_writer_registry.values())
        _writer_registry.clear()
    for writer in writers:
        writer.close()


def set_debug_logging_enabled(enabled):
    """Enable or disable debug logging."""
    global _debug_logging_enabled
    _debug_logging_enabled = enabled


def is_debug_logging_enabled():
    """Check if debug logging is currently enabled."""
    return _debug_logging_enabled


def log_debug(message):
    """Append a timestamped message to the bounded debug log."""
    if not _debug_logging_enabled:
        return

    try:
        append_rotating_text(
            DEBUG_LOG_FILENAME,
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {message}\n",
            max_bytes=DEBUG_LOG_MAX_BYTES,
            backup_count=DEBUG_LOG_BACKUP_COUNT,
        )
    except Exception as e:
        print(f"Error writing to log file: {e}")


def clear_debug_log():
    """Safely clear the active debug log while preserving the shared writer."""
    marker = f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Debug log cleared by user.\n"
    path = resolve_runtime_log_path(DEBUG_LOG_FILENAME)
    writer = _get_writer(
        path,
        DEBUG_LOG_MAX_BYTES,
        DEBUG_LOG_BACKUP_COUNT,
    )
    writer.clear(marker)


ensure_test_log_environment()
atexit.register(close_log_writers)
