"""Debug and custom-prompt tab builders extracted from gui_builder."""

import tkinter as tk
from tkinter import ttk

import gui_builder as _gui_builder
from gui_builder import (
    _cancel_runtime_metrics_refresh,
    _copy_runtime_metrics_summary,
    _refresh_runtime_metrics_panel,
    _reset_runtime_metrics_panel,
)
from modern_ui import style_tk_text_widget
from ui_elements import create_scrollable_tab


def create_debug_tab(app):
    # Create a scrollable tab content frame
    scrollable_content = create_scrollable_tab(app.tab_control, app.ui_lang.get_label("debug_tab_title"))
    app.tab_debug = scrollable_content

    # Create the debug frame inside the scrollable area
    frame = ttk.LabelFrame(scrollable_content, text=app.ui_lang.get_label("debug_tab_title"))
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    image_frame = ttk.Frame(frame)
    image_frame.pack(fill="x", padx=5, pady=5)

    original_frame = ttk.LabelFrame(image_frame, text=app.ui_lang.get_label("original_image_label"))
    original_frame.pack(side=tk.LEFT, padx=5, fill="both", expand=True)
    app.original_image_label = ttk.Label(original_frame, text=app.ui_lang.get_label("no_image_captured", "No image captured yet"))
    app.original_image_label.pack(padx=5, pady=5)

    processed_frame = ttk.LabelFrame(image_frame, text=app.ui_lang.get_label("processed_image_label"))
    processed_frame.pack(side=tk.RIGHT, padx=5, fill="both", expand=True)
    app.processed_image_label = ttk.Label(processed_frame, text=app.ui_lang.get_label("no_image_processed", "No image processed yet"))
    app.processed_image_label.pack(padx=5, pady=5)

    ocr_frame = ttk.LabelFrame(frame, text=app.ui_lang.get_label("ocr_results_label"))
    ocr_frame.pack(fill="x", padx=5, pady=5)
    app.ocr_results_text = tk.Text(ocr_frame, height=16, width=50, wrap=tk.WORD)
    style_tk_text_widget(app.ocr_results_text, getattr(app, "md3_palette", None))
    app.ocr_results_text.pack(fill="both", expand=True, padx=5, pady=5)
    app.ocr_results_text.insert(tk.END, app.ui_lang.get_label("ocr_results_placeholder"))
    app.ocr_results_text.config(state=tk.DISABLED)

    button_frame = ttk.Frame(frame)
    button_frame.pack(fill="x", padx=5, pady=5)
    ttk.Button(button_frame, text=app.ui_lang.get_label("save_debug_images_btn"), command=app.save_debug_images).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text=app.ui_lang.get_label("refresh_log_btn"), command=app.refresh_debug_log).pack(side=tk.LEFT, padx=5)

    diagnostics_frame = ttk.LabelFrame(
        frame,
        text=app.ui_lang.get_label("performance_diagnostics_title", "Performance Diagnostics"),
    )
    diagnostics_frame.pack(fill="both", expand=False, padx=5, pady=5)

    diagnostics_button_frame = ttk.Frame(diagnostics_frame)
    diagnostics_button_frame.pack(fill="x", padx=5, pady=(5, 0))
    ttk.Button(
        diagnostics_button_frame,
        text=app.ui_lang.get_label("reset_metrics_btn", "Reset metrics"),
        command=lambda: _reset_runtime_metrics_panel(app),
    ).pack(side=tk.LEFT, padx=(0, 5))
    ttk.Button(
        diagnostics_button_frame,
        text=app.ui_lang.get_label("copy_metrics_summary_btn", "Copy summary"),
        command=lambda: _copy_runtime_metrics_summary(app),
    ).pack(side=tk.LEFT, padx=5)

    app.runtime_metrics_text = tk.Text(
        diagnostics_frame,
        height=14,
        width=72,
        wrap=tk.WORD,
    )
    style_tk_text_widget(app.runtime_metrics_text, getattr(app, "md3_palette", None))
    app.runtime_metrics_text.pack(fill="both", expand=True, padx=5, pady=5)
    app.runtime_metrics_text.config(state=tk.DISABLED)
    _cancel_runtime_metrics_refresh(app)
    _refresh_runtime_metrics_panel(app)

    log_frame = ttk.LabelFrame(frame, text=app.ui_lang.get_label("app_log_label"))
    log_frame.pack(fill="both", expand=True, padx=5, pady=5)

    app.log_text = tk.Text(log_frame, wrap=tk.WORD)
    style_tk_text_widget(app.log_text, getattr(app, "md3_palette", None))
    scrollbar = ttk.Scrollbar(log_frame, command=app.log_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    app.log_text.config(yscrollcommand=scrollbar.set, state=tk.DISABLED)
    app.log_text.pack(fill="both", expand=True, padx=5, pady=5)

    app.refresh_debug_log()

def create_custom_prompt_tab(app):
    # Create a scrollable tab content frame
    scrollable_content = create_scrollable_tab(app.tab_control, app.ui_lang.get_label("custom_prompt_tab_title", "Custom Prompt"))
    app.tab_custom_prompt = scrollable_content

    # Create the main frame inside the scrollable area
    frame = ttk.LabelFrame(scrollable_content, text=app.ui_lang.get_label("custom_prompt_tab_title", "Custom Prompt"))
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Info label
    info_text = app.ui_lang.get_label("custom_prompt_info", "This text will be added at the beginning of the custom AI translation instruction.")
    info_label = ttk.Label(frame, text=info_text, wraplength=550)
    info_label.pack(fill="x", padx=5, pady=5)

    # Text area
    text_frame = ttk.Frame(frame)
    text_frame.pack(fill="both", expand=True, padx=5, pady=5)

    app.custom_prompt_text_widget = tk.Text(text_frame, wrap=tk.WORD, height=15)
    style_tk_text_widget(app.custom_prompt_text_widget, getattr(app, "md3_palette", None))
    scrollbar = ttk.Scrollbar(text_frame, command=app.custom_prompt_text_widget.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    app.custom_prompt_text_widget.pack(side=tk.LEFT, fill="both", expand=True)
    app.custom_prompt_text_widget.config(yscrollcommand=scrollbar.set)

    # Load initial text
    if hasattr(app, "custom_prompt_text"):
        app.custom_prompt_text_widget.insert("1.0", app.custom_prompt_text)

    # Buttons
    button_frame = ttk.Frame(frame)
    button_frame.pack(fill="x", padx=5, pady=10)

    def save_prompt():
        text = app.custom_prompt_text_widget.get("1.0", "end-1c")
        if not app.save_custom_prompt(text):
            _gui_builder.messagebox.showerror(app.ui_lang.get_label("error_title", "Error"),
                                 app.ui_lang.get_label("custom_prompt_save_error", "Failed to save custom prompt."))

    def reload_prompt():
        app.load_custom_prompt()
        app.custom_prompt_text_widget.delete("1.0", tk.END)
        app.custom_prompt_text_widget.insert("1.0", app.custom_prompt_text)

    app.save_custom_prompt_btn = ttk.Button(button_frame, text=app.ui_lang.get_label("save_btn", "Save"), command=save_prompt)
    app.save_custom_prompt_btn.pack(side=tk.LEFT, padx=5)

    app.reload_custom_prompt_btn = ttk.Button(button_frame, text=app.ui_lang.get_label("reload_btn", "Reload"), command=reload_prompt)
    app.reload_custom_prompt_btn.pack(side=tk.LEFT, padx=5)

    # Function to update labels on language change
    def update_custom_prompt_labels_for_language():
        if hasattr(app, 'tab_custom_prompt') and app.tab_custom_prompt.winfo_exists():
            try:
                frame.config(text=app.ui_lang.get_label("custom_prompt_tab_title", "Custom Prompt"))
                info_label.config(text=app.ui_lang.get_label("custom_prompt_info", "This text will be added at the beginning of the custom AI translation instruction."))
                app.save_custom_prompt_btn.config(text=app.ui_lang.get_label("save_btn", "Save"))
                app.reload_custom_prompt_btn.config(text=app.ui_lang.get_label("reload_btn", "Reload"))
            except Exception as e:
                _gui_builder.log_debug(f"Error updating custom prompt UI language: {e}")

    app.update_custom_prompt_labels_for_language = update_custom_prompt_labels_for_language
