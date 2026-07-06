"""Lightweight styling helpers for the Tkinter UI."""

from tkinter import ttk

WHITE_CLEAN_PALETTE = {
    "window": "#f6f8fb",
    "surface": "#ffffff",
    "surface_variant": "#edf2f7",
    "primary": "#0f766e",
    "primary_container": "#e7f5f2",
    "secondary": "#5c6b7a",
    "text": "#18212f",
    "text_muted": "#5c6b7a",
    "outline": "#ccd6e0",
    "outline_variant": "#e3e9ef",
    "success": "#15803d",
    "warning": "#c2410c",
    "danger": "#b42318",
}

MD3_LIGHT_PALETTE = WHITE_CLEAN_PALETTE


def _safe_configure(widget, **kwargs):
    try:
        widget.configure(**kwargs)
    except Exception:
        pass


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

    style.configure(".", background=palette["window"], foreground=palette["text"], font=("Segoe UI", 10))
    style.configure("TFrame", background=palette["window"])
    style.configure("Surface.TFrame", background=palette["surface"])
    style.configure("TLabelframe", background=palette["window"], bordercolor=palette["outline"], relief="solid")
    style.configure("TLabelframe.Label", background=palette["window"], foreground=palette["text_muted"], font=("Segoe UI Semibold", 10))
    style.configure("Section.TLabelframe", background=palette["surface"], bordercolor=palette["outline"], relief="solid")
    style.configure("Section.TLabelframe.Label", background=palette["surface"], foreground=palette["text"], font=("Segoe UI Semibold", 10))
    style.configure("TLabel", background=palette["window"], foreground=palette["text"])
    style.configure("Muted.TLabel", background=palette["window"], foreground=palette["text_muted"], font=("Segoe UI", 9))
    style.configure("SectionMuted.TLabel", background=palette["surface"], foreground=palette["text_muted"], font=("Segoe UI", 9))
    style.configure("Header.TLabel", background=palette["window"], foreground=palette["text"], font=("Segoe UI Semibold", 10))
    style.configure("TCheckbutton", background=palette["window"], foreground=palette["text"])
    style.configure("TButton", background=palette["surface_variant"], foreground=palette["text"],
                    bordercolor=palette["outline"], focusthickness=1, focuscolor=palette["primary"],
                    padding=(10, 5), relief="solid", borderwidth=1)
    style.map("TButton",
              background=[("active", palette["primary_container"]), ("pressed", palette["primary_container"]),
                          ("disabled", palette["surface_variant"])],
              foreground=[("disabled", palette["text_muted"])],
              bordercolor=[("focus", palette["primary"]), ("active", palette["primary"])])
    style.configure("Secondary.TButton", background=palette["surface_variant"], foreground=palette["text"],
                    bordercolor=palette["outline"], focusthickness=1, focuscolor=palette["primary"],
                    padding=(12, 6), relief="solid", borderwidth=1)
    style.map("Secondary.TButton",
              background=[("active", palette["primary_container"]), ("pressed", palette["primary_container"]),
                          ("disabled", palette["surface_variant"])],
              foreground=[("disabled", palette["text_muted"])],
              bordercolor=[("focus", palette["primary"]), ("active", palette["primary"])])
    _configure_action_button_style(style, "Primary.TButton", palette["primary"], "#0d5f59")
    _configure_action_button_style(style, "ProfileAdd.TButton", palette["success"], "#166534")
    _configure_action_button_style(style, "ProfileSave.TButton", palette["primary"], "#0d5f59")
    _configure_action_button_style(style, "ProfileDelete.TButton", palette["danger"], "#9f1d14")
    _configure_action_button_style(style, "ProfileTest.TButton", palette["warning"], "#9a3412")

    style.configure("TNotebook", background=palette["window"], borderwidth=0, tabmargins=(0, 4, 0, 0))
    style.configure("TNotebook.Tab", background=palette["surface_variant"], foreground=palette["text_muted"],
                    bordercolor=palette["outline"], padding=(10, 7))
    style.map("TNotebook.Tab",
              background=[("selected", palette["surface"]), ("active", palette["primary_container"])],
              foreground=[("selected", palette["text"]), ("active", palette["text"])])

    style.configure("TEntry", fieldbackground=palette["surface"], foreground=palette["text"],
                    bordercolor=palette["outline"], lightcolor=palette["outline"], darkcolor=palette["outline"])
    style.configure("TSpinbox", fieldbackground=palette["surface"], foreground=palette["text"],
                    bordercolor=palette["outline"], arrowcolor=palette["text_muted"])
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
    style.configure("StatusTrack.TFrame", background=palette["primary_container"])
    style.configure("StatusTrackStep.TLabel", background=palette["primary_container"],
                    foreground=palette["text_muted"], font=("Segoe UI Semibold", 9))
    style.configure("StatusTrackDone.TLabel", background=palette["primary_container"],
                    foreground=palette["primary"], font=("Segoe UI Semibold", 9))
    style.configure("StatusTrackArrow.TLabel", background=palette["primary_container"],
                    foreground=palette["text_muted"], font=("Segoe UI", 9))
    style.configure("SettingsGroup.TLabel", background=palette["surface"], foreground=palette["primary"],
                    font=("Segoe UI Semibold", 9))
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
