"""Lightweight styling helpers for the Tkinter UI."""

import tkinter as tk
from tkinter import ttk

WHITE_CLEAN_PALETTE = {
    "window": "#f7f8fb",
    "surface": "#ffffff",
    "surface_variant": "#f1f5f9",
    "primary": "#2563eb",
    "primary_container": "#dbeafe",
    "secondary": "#64748b",
    "text": "#1f2937",
    "text_muted": "#64748b",
    "outline": "#d8dee8",
    "outline_variant": "#edf2f7",
    "success": "#16a34a",
    "warning": "#d97706",
    "danger": "#dc2626",
}

MD3_LIGHT_PALETTE = WHITE_CLEAN_PALETTE


def _safe_configure(widget, **kwargs):
    try:
        widget.configure(**kwargs)
    except Exception:
        pass


def _configure_spinbox_style(style, palette):
    style.configure(
        "TSpinbox",
        fieldbackground=palette["surface"],
        foreground=palette["text"],
        bordercolor=palette["outline"],
        arrowcolor=palette["text_muted"],
        arrowsize=16,
        padding=(5, 3),
    )


def apply_white_clean_theme(root, palette=None):
    """Apply the app-wide white, minimal Tk/ttk theme."""
    palette = dict(WHITE_CLEAN_PALETTE if palette is None else palette)
    _safe_configure(root, bg=palette["window"])

    try:
        root.option_add("*Background", palette["window"])
        root.option_add("*Foreground", palette["text"])
        root.option_add("*selectBackground", palette["primary_container"])
        root.option_add("*selectForeground", palette["text"])
        root.option_add("*insertBackground", palette["primary"])
        root.option_add("*font", ("Segoe UI", 10))
    except Exception:
        pass

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure(".", background=palette["window"], foreground=palette["text"])
    style.configure("TFrame", background=palette["window"])
    style.configure("TLabelframe", background=palette["window"], bordercolor=palette["outline"], relief="solid")
    style.configure("TLabelframe.Label", background=palette["window"], foreground=palette["text_muted"])
    style.configure("TLabel", background=palette["window"], foreground=palette["text"])
    style.configure("TCheckbutton", background=palette["window"], foreground=palette["text"])
    style.configure("TButton", background=palette["surface"], foreground=palette["text"],
                    bordercolor=palette["outline"], focusthickness=1, focuscolor=palette["primary"],
                    padding=(10, 5), relief="flat")
    style.map("TButton",
              background=[("active", "#eef4ff"), ("pressed", palette["primary_container"]),
                          ("disabled", palette["surface_variant"])],
              foreground=[("disabled", palette["text_muted"])],
              bordercolor=[("focus", palette["primary"]), ("active", palette["primary"])])
    _configure_action_button_style(style, "ProfileAdd.TButton", palette["success"], "#15803d")
    _configure_action_button_style(style, "ProfileSave.TButton", palette["primary"], "#1d4ed8")
    _configure_action_button_style(style, "ProfileDelete.TButton", palette["danger"], "#b91c1c")
    _configure_action_button_style(style, "ProfileTest.TButton", palette["warning"], "#b45309")
    # Compact square refresh control for the model-list fetch action.
    style.configure(
        "ProfileRefresh.TButton",
        background=palette["primary"],
        foreground="#ffffff",
        bordercolor=palette["primary"],
        focuscolor="#1d4ed8",
        padding=(0, 0),
        relief="flat",
        font=("Segoe UI Symbol", 13, "bold"),
        anchor="center",
    )
    style.map(
        "ProfileRefresh.TButton",
        background=[
            ("pressed", "#1d4ed8"),
            ("active", "#1d4ed8"),
            ("disabled", "#e5e7eb"),
        ],
        foreground=[("disabled", "#94a3b8"), ("!disabled", "#ffffff")],
        bordercolor=[
            ("pressed", "#1d4ed8"),
            ("active", "#1d4ed8"),
            ("!disabled", palette["primary"]),
        ],
    )

    style.configure("TNotebook", background=palette["window"], borderwidth=0, tabmargins=(0, 4, 0, 0))
    style.configure("TNotebook.Tab", background=palette["surface_variant"], foreground=palette["text_muted"],
                    bordercolor=palette["outline"], padding=(10, 7))
    style.map("TNotebook.Tab",
              background=[("selected", palette["surface"]), ("active", "#eef4ff")],
              foreground=[("selected", palette["text"]), ("active", palette["text"])])

    style.configure("TEntry", fieldbackground=palette["surface"], foreground=palette["text"],
                    bordercolor=palette["outline"], lightcolor=palette["outline"], darkcolor=palette["outline"])
    _configure_spinbox_style(style, palette)
    style.configure("TCombobox", fieldbackground=palette["surface"], background=palette["surface"],
                    foreground=palette["text"], bordercolor=palette["outline"], arrowcolor=palette["text_muted"])
    style.map("TCombobox",
              fieldbackground=[("readonly", palette["surface"]), ("focus", palette["surface"])],
              foreground=[("readonly", palette["text"]), ("focus", palette["text"])],
              selectbackground=[("readonly", palette["primary_container"])],
              selectforeground=[("readonly", palette["text"])])

    style.configure("Vertical.TScrollbar", background=palette["surface_variant"],
                    troughcolor=palette["window"], bordercolor=palette["window"],
                    arrowcolor=palette["text_muted"])
    style.configure("Horizontal.TScrollbar", background=palette["surface_variant"],
                    troughcolor=palette["window"], bordercolor=palette["window"],
                    arrowcolor=palette["text_muted"])
    root.md3_palette = palette
    root.white_clean_palette = palette
    return palette


def _configure_action_button_style(style, style_name, base_color, active_color):
    style.configure(
        style_name,
        background=base_color,
        foreground="#ffffff",
        bordercolor=base_color,
        focuscolor=active_color,
        padding=(14, 7),
        relief="flat",
    )
    style.map(
        style_name,
        background=[("pressed", active_color), ("active", active_color), ("disabled", "#e5e7eb")],
        foreground=[("disabled", "#94a3b8"), ("!disabled", "#ffffff")],
        bordercolor=[("pressed", active_color), ("active", active_color), ("!disabled", base_color)],
    )


def style_tk_canvas(canvas, palette=None):
    palette = dict(WHITE_CLEAN_PALETTE if palette is None else palette)
    _safe_configure(canvas, bg=palette["window"], highlightthickness=0, borderwidth=0)
    return palette


def style_tk_text_widget(widget, palette=None):
    palette = dict(WHITE_CLEAN_PALETTE if palette is None else palette)
    _safe_configure(
        widget,
        bg=palette["surface"],
        fg=palette["text"],
        insertbackground=palette["primary"],
        selectbackground=palette["primary_container"],
        selectforeground=palette["text"],
        relief="solid",
        bd=1,
        highlightthickness=1,
        highlightbackground=palette["outline"],
        highlightcolor=palette["primary"],
    )
    return palette


def style_selection_window(window, palette=None):
    palette = dict(WHITE_CLEAN_PALETTE if palette is None else palette)
    _safe_configure(window, bg=palette["window"])
    return palette


class SquareIconButton(tk.Canvas):
    """Fixed-size square icon button with optically centered glyph text."""

    def __init__(
        self,
        master,
        text="⟳",
        command=None,
        size=28,
        background=None,
        activebackground=None,
        foreground="#ffffff",
        disabledbackground="#e5e7eb",
        disabledforeground="#94a3b8",
        font=("Segoe UI Symbol", 14, "bold"),
        # Many symbol-font refresh glyphs sit low in the em-box; nudge up by default.
        text_offset_y=-1,
        **kwargs,
    ):
        palette = WHITE_CLEAN_PALETTE
        background = palette["primary"] if background is None else background
        activebackground = "#1d4ed8" if activebackground is None else activebackground
        super().__init__(
            master,
            width=int(size),
            height=int(size),
            highlightthickness=0,
            bd=0,
            bg=background,
            cursor="hand2",
            **kwargs,
        )
        self._command = command
        self._size = int(size)
        self._bg = background
        self._active_bg = activebackground
        self._fg = foreground
        self._disabled_bg = disabledbackground
        self._disabled_fg = disabledforeground
        self._enabled = True
        self._icon_id = self.create_text(
            self._size / 2,
            self._size / 2 + float(text_offset_y),
            text=text,
            fill=foreground,
            font=font,
            anchor="center",
        )
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def configure(self, cnf=None, **kwargs):
        if isinstance(cnf, str):
            return super().configure(cnf)
        options = {}
        if isinstance(cnf, dict):
            options.update(cnf)
        options.update(kwargs)
        if "state" in options:
            state = str(options.pop("state")).lower()
            self._set_enabled(state not in ("disabled", "0"))
        if "command" in options:
            self._command = options.pop("command")
        if "text" in options:
            self.itemconfigure(self._icon_id, text=options.pop("text"))
        if options:
            return super().configure(**options)
        return None

    config = configure

    def _set_enabled(self, enabled):
        self._enabled = bool(enabled)
        if self._enabled:
            super().configure(bg=self._bg, cursor="hand2")
            self.itemconfigure(self._icon_id, fill=self._fg)
        else:
            super().configure(bg=self._disabled_bg, cursor="arrow")
            self.itemconfigure(self._icon_id, fill=self._disabled_fg)

    def _on_enter(self, _event=None):
        if self._enabled:
            super().configure(bg=self._active_bg)

    def _on_leave(self, _event=None):
        if self._enabled:
            super().configure(bg=self._bg)

    def _on_press(self, _event=None):
        if self._enabled:
            super().configure(bg=self._active_bg)

    def _on_release(self, _event=None):
        if not self._enabled:
            return
        try:
            x = self.winfo_pointerx() - self.winfo_rootx()
            y = self.winfo_pointery() - self.winfo_rooty()
            inside = 0 <= x < self._size and 0 <= y < self._size
        except Exception:
            inside = True
        if inside and callable(self._command):
            self._command()
        if self._enabled:
            try:
                px = self.winfo_pointerx() - self.winfo_rootx()
                py = self.winfo_pointery() - self.winfo_rooty()
                hovered = 0 <= px < self._size and 0 <= py < self._size
            except Exception:
                hovered = False
            super().configure(bg=self._active_bg if hovered else self._bg)
