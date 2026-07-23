# handlers/configuration_handler.py
from config_manager import load_main_window_geometry, save_main_window_geometry


class ConfigurationHandler:
    """Configuration helpers for the live PaddleOCR + Custom AI path."""

    def __init__(self, app):
        self.app = app

    def load_window_geometry(self):
        load_main_window_geometry(self.app.config, self.app.root, self.app.root.minsize())

    def on_window_configure(self, event):
        if event.widget == self.app.root:
            if getattr(self.app, "_save_timer", None):
                try:
                    self.app.root.after_cancel(self.app._save_timer)
                except Exception:
                    pass
            self.app._save_timer = self.app.root.after(500, self.save_current_window_geometry)

    def save_current_window_geometry(self):
        save_main_window_geometry(self.app.config, self.app.root)
