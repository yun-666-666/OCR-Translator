"""Lightweight styling helpers for the Tkinter UI."""

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
