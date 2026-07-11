"""Lightweight styling helpers for the Tkinter UI."""

import tkinter as tk
from tkinter import ttk

WHITE_CLEAN_PALETTE = {
    # Phase 1.5 Amber Signal palette — 6 token changes from prior teal theme
    "window": "#f0f2f4",           # was #f6f8fb — cooler neutral, avoids Tailwind defaults
    "surface": "#ffffff",
    "surface_variant": "#e4e9ee",   # was #edf2f7 — slightly cooler
    "primary": "#d4700c",           # was #0f766e — amber signal replaces teal
    "primary_container": "#fdf2e6", # was #e7f5f2 — amber light container
    "secondary": "#5c6b7a",
    "text": "#1c2b3a",              # was #18212f — deeper ink navy
    "text_muted": "#5c6b7a",
    "outline": "#ccd6e0",
    "outline_variant": "#e3e8ed",
    "success": "#15803d",
    "warning": "#c2410c",
    "danger": "#c63535",            # was #b42318 — slightly deeper
}

MD3_LIGHT_PALETTE = WHITE_CLEAN_PALETTE


def _safe_configure(widget, **kwargs):
    try:
        widget.configure(**kwargs)
    except Exception:
        pass


def apply_white_clean_theme(root, palette=None):
    """Apply the app-wide Amber Signal Tkinter/ttk theme."""
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

    # Primary (idle Start button) — dark ink
    _configure_action_button_style(style, "Primary.TButton", palette["text"], "#0d1c2a")
    # StartRunning — amber, used when translation is active
    _configure_action_button_style(style, "StartRunning.TButton", palette["primary"], "#b05a08")

    _configure_action_button_style(style, "ProfileAdd.TButton", palette["success"], "#166534")
    _configure_action_button_style(style, "ProfileSave.TButton", palette["text"], "#0d1c2a")
    _configure_action_button_style(style, "ProfileDelete.TButton", palette["danger"], "#a02828")
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

    # Status track strip (workflow bar background)
    style.configure("StatusTrack.TFrame", background=palette["surface_variant"])
    style.configure("StatusTrackStep.TLabel", background=palette["surface_variant"],
                    foreground=palette["text_muted"], font=("Segoe UI Semibold", 9))
    style.configure("StatusTrackDone.TLabel", background=palette["surface_variant"],
                    foreground=palette["primary"], font=("Segoe UI Semibold", 9))
    style.configure("StatusTrackArrow.TLabel", background=palette["surface_variant"],
                    foreground=palette["text_muted"], font=("Segoe UI", 9))

    # Settings group heading label
    style.configure("SettingsGroup.TLabel", background=palette["surface"], foreground=palette["primary"],
                    font=("Segoe UI Semibold", 9))
    # Settings group heading frame (holds amber bar + separator)
    style.configure("SettingsGroupRow.TFrame", background=palette["surface"])

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


# ---------------------------------------------------------------------------
# Pipeline pulse helpers — amber dots while translation is running
# ---------------------------------------------------------------------------

_PULSE_INTERVAL_MS = 550  # ms per step


def _widget_exists_safe(widget):
    try:
        return widget is not None and bool(widget.winfo_exists())
    except Exception:
        return False


def _reset_pipeline_dots(dot_canvases, palette):
    """Set all three pipeline dots to the neutral (idle) colour."""
    idle_color = palette.get("surface_variant", "#e4e9ee")
    for canvas in dot_canvases:
        if _widget_exists_safe(canvas):
            try:
                canvas.itemconfig("dot", fill=idle_color, outline=idle_color)
            except Exception:
                pass


def _pulse_pipeline_step(app, dot_canvases, palette, step):
    """Advance one step of the amber pulse animation.

    Reads ``app.is_running`` each tick — stops itself the moment
    the translation is no longer running or widgets are gone.
    """
    if not getattr(app, "is_running", False):
        _reset_pipeline_dots(dot_canvases, palette)
        app._pipeline_pulse_id = None
        return

    if not all(_widget_exists_safe(c) for c in dot_canvases):
        app._pipeline_pulse_id = None
        return

    amber = palette.get("primary", "#d4700c")
    idle = palette.get("surface_variant", "#e4e9ee")

    for i, canvas in enumerate(dot_canvases):
        try:
            color = amber if i == step else idle
            canvas.itemconfig("dot", fill=color, outline=color)
        except Exception:
            pass

    try:
        app._pipeline_pulse_id = app.root.after(
            _PULSE_INTERVAL_MS,
            _pulse_pipeline_step,
            app,
            dot_canvases,
            palette,
            (step + 1) % len(dot_canvases),
        )
    except Exception:
        app._pipeline_pulse_id = None


def start_pipeline_pulse(app, dot_canvases, palette):
    """Start the amber pulse.  Cancel any existing timer first."""
    cancel_pipeline_pulse(app)
    if not getattr(app, "is_running", False):
        return
    _pulse_pipeline_step(app, dot_canvases, palette, 0)


def cancel_pipeline_pulse(app, dot_canvases=None, palette=None):
    """Cancel the pulse timer and optionally reset dots to idle."""
    after_id = getattr(app, "_pipeline_pulse_id", None)
    if after_id is not None:
        try:
            app.root.after_cancel(after_id)
        except Exception:
            pass
    app._pipeline_pulse_id = None

    if dot_canvases is not None and palette is not None:
        _reset_pipeline_dots(dot_canvases, palette)
