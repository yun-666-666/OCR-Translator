"""
Rust Overlay Manager — Python wrapper for ocr_overlay.exe
通过 subprocess + stdin JSON 通信，零额外 IPC 依赖
"""
import subprocess
import json
import os
import threading
from pathlib import Path
from logger import log_debug


class RustOverlayProcess:
    """管理单个 Rust overlay 进程的生命周期和通信"""

    def __init__(self, x, y, width, height, opacity=0.88, bg_color="#1e1e1e", text_color="#f0f0f0"):
        self.process = None
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.opacity = opacity
        self.bg_color = bg_color
        self.text_color = text_color
        self.visible = False
        self._lock = threading.Lock()

    def start(self):
        """启动 Rust overlay 进程"""
        if self.process and self.process.poll() is None:
            log_debug("RustOverlay 进程已存在，跳过启动")
            return

        # 查找 ocr_overlay.exe 路径
        rust_dir = Path(__file__).parent.parent / "rust" / "target" / "release"
        overlay_exe = rust_dir / "ocr_overlay.exe"

        if not overlay_exe.exists():
            raise FileNotFoundError(f"找不到 Rust overlay 可执行文件: {overlay_exe}")

        # 解析颜色
        bg_r, bg_g, bg_b = self._hex_to_rgb(self.bg_color)
        text_r, text_g, text_b = self._hex_to_rgb(self.text_color)

        # 启动参数
        args = [
            str(overlay_exe),
            "--x", str(self.x),
            "--y", str(self.y),
            "--w", str(self.width),
            "--h", str(self.height),
            "--opacity", str(self.opacity),
        ]

        try:
            self.process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            log_debug(f"RustOverlay 进程启动成功: PID={self.process.pid}")
        except Exception as e:
            log_debug(f"启动 RustOverlay 失败: {e}")
            raise

    def send_command(self, cmd_dict):
        """发送 JSON 指令到 overlay stdin"""
        if not self.process or self.process.poll() is not None:
            log_debug("RustOverlay 进程未运行，无法发送指令")
            return False

        try:
            with self._lock:
                json_line = json.dumps(cmd_dict, ensure_ascii=False) + "\n"
                self.process.stdin.write(json_line)
                self.process.stdin.flush()
            log_debug(f"RustOverlay 发送指令: {cmd_dict}")
            return True
        except Exception as e:
            log_debug(f"RustOverlay 发送指令失败: {e}")
            return False

    def update_text(self, text):
        """更新显示文本"""
        self.visible = True
        return self.send_command({"type": "update", "text": text})

    def hide(self):
        """隐藏 overlay"""
        self.visible = False
        return self.send_command({"type": "hide"})

    def show(self):
        """显示 overlay"""
        self.visible = True
        return self.send_command({"type": "show"})

    def stop(self):
        """停止 overlay 进程"""
        if not self.process:
            return

        try:
            self.send_command({"type": "exit"})
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            log_debug("RustOverlay 正常退出超时，强制终止")
            self.process.kill()
        except Exception as e:
            log_debug(f"停止 RustOverlay 失败: {e}")
        finally:
            self.process = None

    @staticmethod
    def _hex_to_rgb(hex_color):
        """#RRGGBB → (r, g, b)"""
        hex_color = hex_color.lstrip('#')
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )


# ────────────────────────────────────────────────────────────────
# 兼容现有 overlay_manager.py 接口的适配函数
# ────────────────────────────────────────────────────────────────

_rust_source_overlay: RustOverlayProcess | None = None
_rust_target_overlay: RustOverlayProcess | None = None


def is_rust_overlay_available() -> bool:
    """检查 Rust overlay 二进制文件是否存在"""
    rust_dir = Path(__file__).parent.parent / "rust" / "target" / "release"
    return (rust_dir / "ocr_overlay.exe").exists()


def _get_area_from_app(app, area_key: str):
    """从 app 配置中读取区域坐标"""
    cfg = getattr(app, "config", None)
    if cfg is None:
        return 100, 100, 400, 120
    x = int(cfg.get(f"{area_key}_x1", 100))
    y = int(cfg.get(f"{area_key}_y1", 100))
    w = int(cfg.get(f"{area_key}_x2", 500)) - x
    h = int(cfg.get(f"{area_key}_y2", 220)) - y
    return x, y, max(w, 100), max(h, 40)


# ── 目标 overlay（显示翻译结果）────────────────────────────────

def create_target_overlay_rust(app) -> bool:
    """创建/重建 Rust 目标 overlay。返回 True 表示成功。"""
    global _rust_target_overlay

    if not is_rust_overlay_available():
        log_debug("Rust overlay 不可用，回退到 tkinter")
        return False

    # 停掉旧进程
    if _rust_target_overlay is not None:
        try:
            _rust_target_overlay.stop()
        except Exception:
            pass

    x, y, w, h = _get_area_from_app(app, "target")
    opacity = float(getattr(app, "target_opacity_var", type("_", (), {"get": lambda _: 0.88})()).get())
    bg = getattr(app, "target_bg_colour_var", type("_", (), {"get": lambda _: "#1e1e1e"})()).get()
    fg = getattr(app, "target_fg_colour_var", type("_", (), {"get": lambda _: "#f0f0f0"})()).get()

    _rust_target_overlay = RustOverlayProcess(x, y, w, h, opacity, bg, fg)
    try:
        _rust_target_overlay.start()
        return True
    except Exception as e:
        log_debug(f"create_target_overlay_rust 失败: {e}")
        _rust_target_overlay = None
        return False


def update_target_text_rust(text: str) -> bool:
    """更新目标 overlay 的翻译文本"""
    if _rust_target_overlay is None:
        return False
    return _rust_target_overlay.update_text(text)


def toggle_target_visibility_rust():
    """切换目标 overlay 可见状态"""
    if _rust_target_overlay is None:
        return
    if _rust_target_overlay.visible:
        _rust_target_overlay.hide()
    else:
        _rust_target_overlay.show()


# ── 来源 overlay（显示 OCR 原文）────────────────────────────────

def create_source_overlay_rust(app) -> bool:
    """创建/重建 Rust 来源 overlay。返回 True 表示成功。"""
    global _rust_source_overlay

    if not is_rust_overlay_available():
        return False

    if _rust_source_overlay is not None:
        try:
            _rust_source_overlay.stop()
        except Exception:
            pass

    x, y, w, h = _get_area_from_app(app, "source")
    _rust_source_overlay = RustOverlayProcess(x, y, w, h, opacity=0.75)
    try:
        _rust_source_overlay.start()
        return True
    except Exception as e:
        log_debug(f"create_source_overlay_rust 失败: {e}")
        _rust_source_overlay = None
        return False


def update_source_text_rust(text: str) -> bool:
    """更新来源 overlay 文本"""
    if _rust_source_overlay is None:
        return False
    return _rust_source_overlay.update_text(text)


def toggle_source_visibility_rust():
    """切换来源 overlay 可见状态"""
    if _rust_source_overlay is None:
        return
    if _rust_source_overlay.visible:
        _rust_source_overlay.hide()
    else:
        _rust_source_overlay.show()


# ── 清理 ─────────────────────────────────────────────────────────

def cleanup_rust_overlays():
    """进程退出时调用，确保子进程被终止"""
    global _rust_source_overlay, _rust_target_overlay
    for overlay in (_rust_source_overlay, _rust_target_overlay):
        if overlay is not None:
            try:
                overlay.stop()
            except Exception:
                pass
    _rust_source_overlay = None
    _rust_target_overlay = None
