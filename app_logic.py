# --- Configuration ---
ENABLE_PROCESS_CPU_AFFINITY = False  # Set to False to disable process-level CPU core limiting

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import queue
import sys
import os
import re
import gc
import traceback
import concurrent.futures

from logger import log_debug, set_debug_logging_enabled, is_debug_logging_enabled
from resource_handler import get_resource_path
from config_manager import (
    get_provider_api_key,
    load_app_config,
    save_app_config,
    load_ocr_preview_geometry,
    save_ocr_preview_geometry,
)
from gui_builder import create_main_tab, create_settings_tab, create_custom_prompt_tab, create_debug_tab
from overlay_manager import (
    select_source_area_om, select_target_area_om,
    create_source_overlay_om, create_target_overlay_om,
    toggle_source_visibility_om, toggle_target_visibility_om, load_areas_from_config_om
)
from language_manager import LanguageManager
from language_ui import UILanguageManager
from modern_ui import (
    apply_white_clean_theme,
    style_tk_canvas,
    style_tk_text_widget,
    start_pipeline_pulse,
    cancel_pipeline_pulse,
)
from runtime_metrics import RuntimeMetrics
from custom_ai import (
    CustomAIProfileManager,
    CUSTOM_AI_LATENCY_MODE_SAFE,
    normalize_custom_ai_latency_mode,
)
from ocr_utils import (
    CaptureBackendSelector,
    OCRFrameCache,
    API_OCR_IMAGE_DETAIL_DEFAULT,
    API_OCR_IMAGE_FORMAT_DEFAULT,
    API_OCR_IMAGE_MODE_DEFAULT,
    API_OCR_IMAGE_QUALITY_DEFAULT,
    encode_image_for_api_ocr,
    encode_image_for_api_ocr_payload,
    normalize_api_ocr_image_detail,
    normalize_api_ocr_image_format,
    normalize_api_ocr_image_mode,
    normalize_api_ocr_image_quality,
)
from paddle_ocr_backend import (
    PADDLEOCR_DISPLAY_NAME,
    PADDLEOCR_MODEL_CODE,
    clear_paddleocr_engines,
    get_paddleocr_engine,
    get_paddleocr_text_recognition_engine,
    prepare_paddleocr_image,
    recognize_with_paddleocr,
)

from handlers import (
    CacheManager,
    ConfigurationHandler,
    DisplayManager,
    HotkeyHandler,
    TranslationHandler,
    UIInteractionHandler
)

DEFAULT_CUSTOM_PROMPT = (
    "Use context to resolve ambiguity. Translate naturally and concisely while preserving meaning, "
    "tone, and character voice. Keep names and game terms consistent."
)

KEYBOARD_AVAILABLE = False
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    pass

GOOGLE_TRANSLATE_API_AVAILABLE = False
DEEPL_API_AVAILABLE = False
GEMINI_API_AVAILABLE = False
OPENAI_API_AVAILABLE = False
MARIANMT_AVAILABLE = False


class _DisabledModelManager:
    def get_translation_model_names(self):
        return []

    def get_ocr_model_names(self):
        return []

    def get_api_name_by_display_name(self, display_name):
        return None

    def get_model_costs(self, api_name):
        return {'input_cost': 0.0, 'output_cost': 0.0}

class GameChangingTranslator:
    def __init__(self, root):
        self.root = root
        self.root.title("Game-Changing Translator")
        self.root.geometry("750x480")
        self.root.minsize(650, 430)
        self.root.resizable(True, True)
        self.md3_palette = apply_white_clean_theme(self.root)

        self._fully_initialized = False # Flag for settings save callback
        self.toggle_in_progress = False

        self.KEYBOARD_AVAILABLE = KEYBOARD_AVAILABLE
        self.GOOGLE_TRANSLATE_API_AVAILABLE = GOOGLE_TRANSLATE_API_AVAILABLE
        self.DEEPL_API_AVAILABLE = DEEPL_API_AVAILABLE
        self.GEMINI_API_AVAILABLE = GEMINI_API_AVAILABLE
        self.OPENAI_API_AVAILABLE = OPENAI_API_AVAILABLE
        self.MARIANMT_AVAILABLE = MARIANMT_AVAILABLE

        # Debug: Log execution environment information
        import sys
        is_compiled = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
        log_debug(f"Application execution environment:")
        log_debug(f"  Compiled/Frozen: {is_compiled}")
        if is_compiled:
            log_debug(f"  Executable path: {sys.executable}")
            log_debug(f"  Bundle dir: {getattr(sys, '_MEIPASS', 'Unknown')}")
        else:
            log_debug(f"  Python script mode")
            log_debug(f"  Script path: {__file__}")

        log_debug(f"Library availability check:")
        if not KEYBOARD_AVAILABLE: log_debug("  Keyboard library not available. Hotkeys disabled.")
        else: log_debug("  Keyboard library: available")
        log_debug("  Built-in cloud translation providers disabled; using custom OpenAI-compatible endpoints.")
        log_debug("  MarianMT disabled to avoid heavy startup imports.")

        # Process-Level CPU Affinity: Limit application to exactly 3 cores
        if ENABLE_PROCESS_CPU_AFFINITY:
            try:
                import psutil
                cpu_count = os.cpu_count() or 2
                if cpu_count >= 3:  # Only limit if system has 3+ cores
                    # Use exactly 3 cores: [0, 1, 2]
                    available_cores = [0, 1, 2]
                    psutil.Process().cpu_affinity(available_cores)

                    # Also set environment variable for OpenMP-backed OCR libraries.
                    os.environ['OMP_NUM_THREADS'] = '3'

                    log_debug(f"Limited application to exactly 3 CPU cores: {available_cores} (out of {cpu_count} total)")
                    log_debug("Set OMP_NUM_THREADS=3 for OCR thread limiting")
                else:
                    log_debug(f"System has {cpu_count} cores - no CPU limiting applied (need 3+ cores)")
            except ImportError:
                log_debug("psutil not available - CPU affinity not set. Install psutil for CPU core limiting.")
            except Exception as e_cpu:
                log_debug(f"Error setting CPU affinity: {e_cpu}")
        else:
            log_debug("Process-level CPU affinity DISABLED via configuration flag")

        self.source_area = None
        self.target_area = None
        self.is_running = False
        self.threads = []
        self.last_image_hash = None
        self.source_overlay = None
        self.target_overlay = None
        self.translation_text = None
        self.text_stability_counter = 0
        self.previous_text = ""
        self.last_screenshot = None
        self.last_processed_image = None
        self.raw_image_for_gemini = None  # WebP bytes ready for Gemini API

        # Gemini OCR Batch Infrastructure (Phase 1)
        self.last_processed_subtitle = None  # Store last processed subtitle for successive comparison
        self.batch_sequence_counter = 0  # Track batch sequence numbers
        self.clear_timeout_timer_start = None  # Timer for clear translation timeout
        self.active_ocr_calls = set()  # Track active async OCR calls
        self.max_concurrent_ocr_calls = 8  # Limit concurrent OCR API calls (8 for Gemini)

        # Gemini OCR Simple Management (No Queue for Gemini)
        self.last_displayed_batch_sequence = 0  # Track chronological order

        # Translation Async Processing Infrastructure (Phase 2)
        self.translation_sequence_counter = 0  # Track translation sequence numbers
        self.last_displayed_translation_sequence = 0  # Track chronological order for translations
        self.active_translation_calls = set()  # Track active async translation calls
        self.active_translation_inflight_keys = set()  # Track unique in-flight translation requests
        self.active_translation_started_monotonic = {}  # Track request age for bounded stale-call supersession
        self.max_concurrent_translation_calls = 6  # Limit concurrent translation API calls
        self.translation_supersede_after_seconds = 1.5  # Let the newest subtitle bypass one stale slow call
        self.last_translation_submit_monotonic = 0.0  # Track the last time we submitted a translation request
        self.pending_translation_request = None  # Latest queued translation request while throttled
        self.pending_translation_flush_scheduled = False  # Whether a queued translation flush is scheduled
        self.pending_translation_flush_deadline_monotonic = 0.0  # Deadline for the current flush timer
        self.pending_translation_flush_generation = 0  # Invalidates superseded flush timers
        self.last_local_ocr_submitted_text = None  # Last successfully displayed local OCR source text
        self.last_local_ocr_submitted_norm = None  # Normalized local OCR source text used for resubmit dedup
        self.last_local_ocr_submitted_scope = None  # Translation scope associated with the last local OCR submit
        self.runtime_metrics = RuntimeMetrics(max_events=240, max_age_seconds=60.0)
        self.runtime_metrics_refresh_after_id = None
        self.capture_backend_selector = CaptureBackendSelector()
        self.ocr_stability_gate = None

        # Initialize thread pools for optimized performance (especially for compiled version)
        self.ocr_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=8,
            thread_name_prefix="ApiOCR"
        )
        self.translation_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=6,
            thread_name_prefix="Translation"
        )
        log_debug("Initialized thread pools for OCR and translation processing")
        self._paddleocr_prewarm_lock = threading.RLock()
        self._paddleocr_prewarm_thread = None
        self._paddleocr_prewarm_settings = None
        self._paddleocr_prewarmed_settings = None
        self._paddleocr_prewarm_generation = 0

        # Adaptive Scan Interval Infrastructure
        self.base_scan_interval = 500  # User's preferred setting (will be updated from config)
        self.current_scan_interval = 500  # Dynamic value used by capture thread
        self.load_check_timer = 0
        self.overload_detected = False
        self._last_adaptive_log_state = None
        self._last_adaptive_log_time = 0.0
        log_debug("Initialized adaptive scan interval infrastructure")

        # OCR Preview window
        self.ocr_preview_window = None

        self.config = load_app_config()
        self.language_manager = LanguageManager()

        # Initialize UI language manager with the saved language if available
        self.ui_lang = UILanguageManager()
        saved_language_display = self.ui_lang.normalize_display_name(
            self.config['Settings'].get('gui_language', 'English')
        )
        if saved_language_display != 'English':
            lang_code = self.ui_lang.get_language_code_from_name(saved_language_display)
            if lang_code:
                self.ui_lang.load_language(lang_code)
                log_debug(f"Loaded UI language from config: {lang_code}")

        self.root.bind('<Configure>', self.on_window_configure)
        self._save_timer = None
        self._save_settings_timer = None

        # Initialize Tkinter Variables FIRST ---
        self.source_colour_var = tk.StringVar(value=self.config['Settings'].get('source_area_colour', '#FFFF99'))
        self.target_colour_var = tk.StringVar(value=self.config['Settings'].get('target_area_colour', '#663399'))
        self.target_text_colour_var = tk.StringVar(value=self.config['Settings'].get('target_text_colour', '#FFFFFF'))
        self.debug_logging_enabled_var = tk.BooleanVar(value=self.config.getboolean('Settings', 'debug_logging_enabled', fallback=True))
        self.gui_language_var = tk.StringVar(value=saved_language_display)
        self.keep_linebreaks_var = tk.BooleanVar(value=self.config.getboolean('Settings', 'keep_linebreaks', fallback=False))
        self.capture_backend_var = tk.StringVar(value=self.config['Settings'].get('capture_backend', 'auto'))
        self.ocr_frame_cache_size_var = tk.IntVar(value=int(self.config['Settings'].get('ocr_frame_cache_size', '64')))
        self.enable_instant_cache_display_var = tk.BooleanVar(value=self.config.getboolean('Settings', 'enable_instant_cache_display', fallback=True))

        # OCR Model Selection
        configured_ocr_model = self.config['Settings'].get('ocr_model', PADDLEOCR_MODEL_CODE)
        if configured_ocr_model not in [PADDLEOCR_MODEL_CODE, 'custom_ai']:
            configured_ocr_model = PADDLEOCR_MODEL_CODE
        self.ocr_model_var = tk.StringVar(value=configured_ocr_model)

        self.google_api_key_var = tk.StringVar(value=get_provider_api_key(self.config, 'google_translate_api_key'))
        self.deepl_api_key_var = tk.StringVar(value=get_provider_api_key(self.config, 'deepl_api_key'))
        self.gemini_api_key_var = tk.StringVar(value=get_provider_api_key(self.config, 'gemini_api_key'))
        self.deepl_model_type_var = tk.StringVar(value=self.config['Settings'].get('deepl_model_type', 'latency_optimized'))
        translation_model_val = self.config['Settings'].get('translation_model', 'custom_ai')
        if translation_model_val != 'custom_ai':
            log_debug(f"Configured legacy translation model '{translation_model_val}' migrated to custom_ai")
            translation_model_val = 'custom_ai'
        self.translation_model_var = tk.StringVar(value=translation_model_val)

        # Define translation model names and values earlier
        # Initialize with default values, will be updated with localized versions
        self.translation_model_names = {'custom_ai': 'Custom AI Translation'}
        self.gemini_models_manager = _DisabledModelManager()
        self.openai_models_manager = _DisabledModelManager()
        self.custom_ai_profiles = CustomAIProfileManager(self.config['Settings'].get('custom_ai_profiles_file', 'custom_ai_profiles.json'))

        # Update with localized names after UI language is loaded
        self.update_translation_model_names()
        self.translation_model_values = {v: k for k, v in self.translation_model_names.items()}

        self.models_file_var = tk.StringVar(value=self.config['Settings'].get('marian_models_file'))
        self.num_beams_var = tk.IntVar(value=int(self.config['Settings'].get('num_beams', '2')))
        self.marian_model_var = tk.StringVar(value=self.config['Settings'].get('marian_model', '')) # Stores path

        self.google_file_cache_var = tk.BooleanVar(value=self.config.getboolean('Settings', 'google_file_cache', fallback=True))
        self.deepl_file_cache_var = tk.BooleanVar(value=self.config.getboolean('Settings', 'deepl_file_cache', fallback=True))
        self.deepl_context_window_var = tk.IntVar(value=int(self.config['Settings'].get('deepl_context_window', '2')))
        self.gemini_file_cache_var = tk.BooleanVar(value=self.config.getboolean('Settings', 'gemini_file_cache', fallback=True))
        self.gemini_context_window_var = tk.IntVar(value=int(self.config['Settings'].get('gemini_context_window', '1')))
        self.gemini_api_log_enabled_var = tk.BooleanVar(value=self.config.getboolean('Settings', 'gemini_api_log_enabled', fallback=True))

        # OpenAI API variables
        self.openai_file_cache_var = tk.BooleanVar(value=self.config.getboolean('Settings', 'openai_file_cache', fallback=True))
        self.openai_context_window_var = tk.IntVar(value=int(self.config['Settings'].get('openai_context_window', '2')))
        self.openai_api_log_enabled_var = tk.BooleanVar(value=self.config.getboolean('Settings', 'openai_api_log_enabled', fallback=True))
        self.openai_api_key_var = tk.StringVar(value=get_provider_api_key(self.config, 'openai_api_key'))
        self.custom_context_window_var = tk.IntVar(value=int(self.config['Settings'].get('custom_context_window', '5')))
        self.custom_ai_latency_mode_var = tk.StringVar(
            value=normalize_custom_ai_latency_mode(
                self.config['Settings'].get('custom_ai_latency_mode', CUSTOM_AI_LATENCY_MODE_SAFE)
            )
        )
        try:
            custom_ai_submit_interval_ms = int(
                self.config['Settings'].get('custom_ai_submit_interval_ms', '300')
            )
        except (TypeError, ValueError):
            custom_ai_submit_interval_ms = 300
        self.custom_ai_submit_interval_ms_var = tk.IntVar(
            value=max(0, min(5000, custom_ai_submit_interval_ms))
        )
        self.custom_ai_ocr_image_format_var = tk.StringVar(
            value=normalize_api_ocr_image_format(
                self.config['Settings'].get('custom_ai_ocr_image_format', API_OCR_IMAGE_FORMAT_DEFAULT)
            )
        )
        self.custom_ai_ocr_image_mode_var = tk.StringVar(
            value=normalize_api_ocr_image_mode(
                self.config['Settings'].get('custom_ai_ocr_image_mode', API_OCR_IMAGE_MODE_DEFAULT)
            )
        )
        self.custom_ai_ocr_image_quality_var = tk.IntVar(
            value=normalize_api_ocr_image_quality(
                self.config['Settings'].get('custom_ai_ocr_image_quality', str(API_OCR_IMAGE_QUALITY_DEFAULT))
            )
        )
        self.custom_ai_ocr_image_detail_var = tk.StringVar(
            value=normalize_api_ocr_image_detail(
                self.config['Settings'].get('custom_ai_ocr_image_detail', API_OCR_IMAGE_DETAIL_DEFAULT)
            )
        )

        self.paddleocr_source_dir_var = tk.StringVar(
            value=self.config['Settings'].get('paddleocr_source_dir', 'PaddleOCR-3.7.0')
        )
        self.paddleocr_lang_var = tk.StringVar(
            value=self.config['Settings'].get('paddleocr_lang', 'en')
        )
        self.paddleocr_ocr_version_var = tk.StringVar(
            value=self.config['Settings'].get('paddleocr_ocr_version', 'PP-OCRv6')
        )
        self.paddleocr_model_size_var = tk.StringVar(
            value=self.config['Settings'].get('paddleocr_model_size', 'tiny')
        )
        self.paddleocr_device_var = tk.StringVar(
            value=self.config['Settings'].get('paddleocr_device', 'cpu')
        )
        self.paddleocr_min_score_var = tk.StringVar(
            value=self.config['Settings'].get('paddleocr_min_score', '0.35')
        )
        self.paddleocr_upscale_var = tk.StringVar(
            value=self.config['Settings'].get('paddleocr_upscale', '1.0')
        )
        self.paddleocr_text_det_limit_side_len_var = tk.StringVar(
            value=self.config['Settings'].get('paddleocr_text_det_limit_side_len', '960')
        )
        self.paddleocr_text_det_limit_type_var = tk.StringVar(
            value=self.config['Settings'].get('paddleocr_text_det_limit_type', 'max')
        )
        self.paddleocr_use_textline_orientation_var = tk.BooleanVar(
            value=self.config.getboolean('Settings', 'paddleocr_use_textline_orientation', fallback=False)
        )

        # Separate Gemini model selection for OCR and Translation
        self.gemini_translation_model_var = tk.StringVar(value=self.config['Settings'].get('gemini_translation_model', 'Gemini 2.5 Flash-Lite'))
        self.gemini_ocr_model_var = tk.StringVar(value=self.config['Settings'].get('gemini_ocr_model', 'Gemini 2.5 Flash-Lite'))

        # OpenAI model selection for OCR and Translation
        self.openai_translation_model_var = tk.StringVar(value=self.config['Settings'].get('openai_translation_model', 'GPT-4o Mini'))
        self.openai_ocr_model_var = tk.StringVar(value=self.config['Settings'].get('openai_ocr_model', 'GPT-4o'))

        # Gemini statistics variables (initialized by GUI builder)
        self.gemini_total_words_var = None
        self.gemini_total_cost_var = None

        # OpenAI statistics variables (initialized by GUI builder)
        self.openai_total_words_var = None
        self.openai_total_cost_var = None

        self.scan_interval_var = tk.IntVar(value=int(self.config['Settings'].get('scan_interval', '100')))

        # Initialize adaptive scan interval values from user configuration
        initial_scan_interval = self.scan_interval_var.get()
        self.base_scan_interval = initial_scan_interval  # Update with user's actual setting
        self.current_scan_interval = initial_scan_interval  # Start with user's setting
        log_debug(f"Initialized adaptive scan interval: base={self.base_scan_interval}ms, current={self.current_scan_interval}ms")

        self.clear_translation_timeout_var = tk.IntVar(value=int(self.config['Settings'].get('clear_translation_timeout', '3')))
        self.stability_var = tk.IntVar(value=int(self.config['Settings'].get('stability_threshold', '2')))
        self.ocr_debugging_var = tk.BooleanVar(value=self.config.getboolean('Settings', 'ocr_debugging', fallback=False))
        self.target_font_size_var = tk.IntVar(value=int(self.config['Settings'].get('target_font_size', '12')))
        self.target_font_type_var = tk.StringVar(value=self.config['Settings'].get('target_font_type', 'Arial'))
        self.target_opacity_var = tk.DoubleVar(value=float(self.config['Settings'].get('target_opacity', '0.15')))
        self.target_text_opacity_var = tk.DoubleVar(value=float(self.config['Settings'].get('target_text_opacity', '1.0')))

        # Initialize OCR model display variable here to ensure it persists across UI rebuilds
        self.ocr_model_display_var = tk.StringVar()
        self.custom_translation_profile_display_var = tk.StringVar()
        self.custom_ocr_profile_display_var = tk.StringVar()
        initial_ocr_model_code = self.ocr_model_var.get()
        initial_ocr_display_name = ""
        if initial_ocr_model_code not in [PADDLEOCR_MODEL_CODE, 'custom_ai']:
            log_debug(f"Configured legacy OCR model '{initial_ocr_model_code}' migrated to paddleocr")
            self.ocr_model_var.set(PADDLEOCR_MODEL_CODE)
            initial_ocr_model_code = PADDLEOCR_MODEL_CODE
        if initial_ocr_model_code == PADDLEOCR_MODEL_CODE:
            initial_ocr_display_name = self.ui_lang.get_label("ocr_model_paddleocr", PADDLEOCR_DISPLAY_NAME)
        elif initial_ocr_model_code == 'custom_ai':
            active_ocr_profile = self.custom_ai_profiles.get_active_profile("ocr")
            initial_ocr_display_name = active_ocr_profile["name"] if active_ocr_profile else self.ui_lang.get_label("custom_ai_no_profiles", "Add an AI model profile")
        elif self.is_gemini_model(initial_ocr_model_code):
            saved_gemini_ocr_model = self.config['Settings'].get('gemini_ocr_model', '')
            if saved_gemini_ocr_model and self.GEMINI_API_AVAILABLE and saved_gemini_ocr_model in self.gemini_models_manager.get_ocr_model_names():
                initial_ocr_display_name = saved_gemini_ocr_model
        elif self.is_openai_model(initial_ocr_model_code):
            saved_openai_ocr_model = self.config['Settings'].get('openai_ocr_model', '')
            if saved_openai_ocr_model and self.OPENAI_API_AVAILABLE and saved_openai_ocr_model in self.openai_models_manager.get_ocr_model_names():
                initial_ocr_display_name = saved_openai_ocr_model

        # Fallback if no specific display name was found
        if not initial_ocr_display_name:
            initial_ocr_display_name = self.ui_lang.get_label("ocr_model_paddleocr", PADDLEOCR_DISPLAY_NAME)

        self.ocr_model_display_var.set(initial_ocr_display_name)

        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            self.base_dir = os.path.dirname(sys.executable)
        else:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.custom_ai_translation_cache_file = os.path.join(
            self.base_dir,
            "custom_ai_translation_cache.sqlite3",
        )

        # Initialize Handlers
        # self.cache_manager = CacheManager(self)
        self.configuration_handler = ConfigurationHandler(self)
        self.display_manager = DisplayManager(self)
        self.hotkey_handler = HotkeyHandler(self)
        self.translation_handler = TranslationHandler(self)
        self.ui_interaction_handler = UIInteractionHandler(self) # Needs self.translation_model_names

        # Pre-initialize Gemini model for optimal performance (especially for compiled version)
        self._pre_initialize_gemini_model()

        # Initialize trace suppression mechanism and UI update detection
        self._suppress_traces = False
        self._ui_update_in_progress = False

        def _settings_changed_callback_internal(*args, **kwargs):
            if self._fully_initialized and not self._suppress_traces and not self._ui_update_in_progress:
                self.save_settings()
            elif self._suppress_traces:
                log_debug("StringVar trace suppressed during UI update")
            elif self._ui_update_in_progress:
                log_debug("StringVar trace suppressed during UI update operation")

        self.settings_changed_callback = _settings_changed_callback_internal

        def _scan_interval_changed_callback(*args, **kwargs):
            if self._fully_initialized and not self._suppress_traces and not self._ui_update_in_progress:
                # Update adaptive scan interval when user changes scan interval
                new_scan_interval = self.scan_interval_var.get()
                if hasattr(self, 'base_scan_interval') and new_scan_interval != self.base_scan_interval:
                    self.base_scan_interval = new_scan_interval
                    # Reset to new base if not currently overloaded, or update overloaded value
                    if not self.overload_detected:
                        self.current_scan_interval = new_scan_interval
                        log_debug(f"Adaptive scan interval updated: base={self.base_scan_interval}ms, current={self.current_scan_interval}ms")
                    else:
                        self.current_scan_interval = int(new_scan_interval * 1.5)  # Maintain 150% overload ratio
                        log_debug(f"Adaptive scan interval updated during overload: base={self.base_scan_interval}ms, current={self.current_scan_interval}ms")

                self.save_settings()
            elif self._suppress_traces:
                log_debug("Scan interval trace suppressed during UI update")
            elif self._ui_update_in_progress:
                log_debug("Scan interval trace suppressed during UI update operation")

        self.scan_interval_changed_callback = _scan_interval_changed_callback

        def _custom_context_window_changed_callback(*args, **kwargs):
            if self._fully_initialized and not self._suppress_traces and not self._ui_update_in_progress:
                if hasattr(self, 'translation_handler') and hasattr(self.translation_handler, '_clear_active_context'):
                    self.translation_handler._clear_active_context()
                self.save_settings()
            elif self._suppress_traces:
                log_debug("Custom context window trace suppressed during UI update")
            elif self._ui_update_in_progress:
                log_debug("Custom context window trace suppressed during UI update operation")

        self.custom_context_window_changed_callback = _custom_context_window_changed_callback

        # Add traces
        self.source_colour_var.trace_add("write", self.settings_changed_callback)
        self.target_colour_var.trace_add("write", self.settings_changed_callback)
        self.target_text_colour_var.trace_add("write", self.settings_changed_callback)
        self.debug_logging_enabled_var.trace_add("write", self.settings_changed_callback)
        self.keep_linebreaks_var.trace_add("write", self.settings_changed_callback)
        self.capture_backend_var.trace_add("write", self.settings_changed_callback)
        self.ocr_frame_cache_size_var.trace_add("write", self.settings_changed_callback)
        self.ocr_frame_cache_size_var.trace_add("write", self.on_ocr_frame_cache_size_change)
        self.enable_instant_cache_display_var.trace_add("write", self.settings_changed_callback)
        self.google_api_key_var.trace_add("write", self.settings_changed_callback)
        self.deepl_api_key_var.trace_add("write", self.settings_changed_callback)
        self.deepl_model_type_var.trace_add("write", self.settings_changed_callback)
        self.models_file_var.trace_add("write", self.settings_changed_callback)
        self.google_file_cache_var.trace_add("write", self.settings_changed_callback)
        self.deepl_file_cache_var.trace_add("write", self.settings_changed_callback)
        self.deepl_context_window_var.trace_add("write", self.settings_changed_callback)
        self.custom_context_window_var.trace_add("write", self.custom_context_window_changed_callback)
        self.custom_ai_latency_mode_var.trace_add("write", self.settings_changed_callback)
        self.custom_ai_submit_interval_ms_var.trace_add("write", self.settings_changed_callback)
        self.custom_ai_ocr_image_format_var.trace_add("write", self.settings_changed_callback)
        self.custom_ai_ocr_image_mode_var.trace_add("write", self.settings_changed_callback)
        self.custom_ai_ocr_image_quality_var.trace_add("write", self.settings_changed_callback)
        self.custom_ai_ocr_image_detail_var.trace_add("write", self.settings_changed_callback)
        for paddleocr_var in (
            self.paddleocr_source_dir_var,
            self.paddleocr_lang_var,
            self.paddleocr_ocr_version_var,
            self.paddleocr_model_size_var,
            self.paddleocr_device_var,
            self.paddleocr_min_score_var,
            self.paddleocr_upscale_var,
            self.paddleocr_text_det_limit_side_len_var,
            self.paddleocr_text_det_limit_type_var,
            self.paddleocr_use_textline_orientation_var,
        ):
            paddleocr_var.trace_add("write", self.settings_changed_callback)
            paddleocr_var.trace_add("write", self.on_ocr_parameter_change)
        self.ocr_debugging_var.trace_add("write", self.settings_changed_callback)
        self.scan_interval_var.trace_add("write", self.scan_interval_changed_callback)  # Special validation callback
        self.clear_translation_timeout_var.trace_add("write", self.settings_changed_callback)
        self.stability_var.trace_add("write", self.settings_changed_callback)
        self.target_font_size_var.trace_add("write", self.settings_changed_callback)
        self.target_font_type_var.trace_add("write", self.settings_changed_callback)
        self.target_opacity_var.trace_add("write", self.settings_changed_callback)
        self.target_text_opacity_var.trace_add("write", self.settings_changed_callback)
        self.num_beams_var.trace_add("write", self.settings_changed_callback)
        self.marian_model_var.trace_add("write", self.settings_changed_callback)
        self.gui_language_var.trace_add("write", self.settings_changed_callback)
        self.ocr_model_var.trace_add("write", self.settings_changed_callback)
        self.ocr_model_var.trace_add("write", self.on_ocr_model_change)

        # Other instance variables
        # Increased queue sizes from 4/3 to 8/6 to reduce queue management overhead
        self.ocr_queue = queue.Queue(maxsize=8)  # Increased from 4 for better buffering
        self.translation_queue = queue.Queue(maxsize=6)  # Increased from 3 for better buffering
        self.last_successful_translation_time = 0.0
        self.min_translation_interval = 0.3
        self.last_translation_time = time.monotonic()
        self.google_api_client = None
        self.deepl_api_client = None
        self.google_api_key_visible = False
        self.deepl_api_key_visible = False
        self.gemini_api_key_visible = False
        self.openai_api_key_visible = False
        self.marian_translator = None
        self.marian_source_lang = None
        self.marian_target_lang = None

        self.google_source_lang = self.config['Settings'].get('google_source_lang', 'auto')
        self.google_target_lang = self.config['Settings'].get('google_target_lang', 'en')
        self.deepl_source_lang = self.config['Settings'].get('deepl_source_lang', 'auto')
        self.deepl_target_lang = self.config['Settings'].get('deepl_target_lang', 'EN-GB')
        self.gemini_source_lang = self.config['Settings'].get('gemini_source_lang', 'en')
        self.gemini_target_lang = self.config['Settings'].get('gemini_target_lang', 'pl')

        # OpenAI language settings
        self.openai_source_lang = self.config['Settings'].get('openai_source_lang', 'en')
        self.openai_target_lang = self.config['Settings'].get('openai_target_lang', 'pl')
        self.custom_source_lang = self.config['Settings'].get('custom_source_lang', 'auto')
        self.custom_target_lang = self.config['Settings'].get('custom_target_lang', 'en')

        base_dir = self.base_dir

        self.google_cache_file = os.path.join(base_dir, "googletrans_cache.txt")
        self.deepl_cache_file = os.path.join(base_dir, "deepl_cache.txt")
        self.gemini_cache_file = os.path.join(base_dir, "gemini_cache.txt")
        self.openai_cache_file = os.path.join(base_dir, "openai_cache.txt")
        self.custom_prompt_file = os.path.join(base_dir, "custom_prompt.txt")
        log_debug(f"Cache file paths: Google: {self.google_cache_file}, DeepL: {self.deepl_cache_file}, Gemini: {self.gemini_cache_file}, OpenAI: {self.openai_cache_file}")

        self.custom_prompt_text = ""
        self.load_custom_prompt()

        self.google_file_cache = {}
        self.deepl_file_cache = {}
        self.gemini_file_cache = {}
        self.openai_file_cache = {}
        self.translation_cache = {}
        self.ocr_frame_cache = OCRFrameCache(self.ocr_frame_cache_size_var.get())

        self.cache_manager = CacheManager(self)

        log_debug(f"OCR model initialized: {self.ocr_model_var.get()}")

        self.stable_threshold = self.stability_var.get()
        self.clear_translation_timeout = self.clear_translation_timeout_var.get()

        if not self.google_source_lang: self.google_source_lang = 'auto'
        if not self.google_target_lang: self.google_target_lang = 'en'
        if not self.deepl_source_lang: self.deepl_source_lang = 'auto'
        if not self.deepl_target_lang: self.deepl_target_lang = 'EN-GB'

        self.cache_manager.load_file_caches()

        # Initialize debug logging state
        set_debug_logging_enabled(self.debug_logging_enabled_var.get())

        self.marian_models_dict, self.marian_models_list = self.configuration_handler.load_marian_models(localize_names=True)
        self.configuration_handler.load_window_geometry()

        # Initialize UI display StringVars here so they exist before create_settings_tab
        self.source_display_var = tk.StringVar()
        self.target_display_var = tk.StringVar()

        configured_marian_path = self.marian_model_var.get()
        initial_marian_display_name = ""
        if configured_marian_path:
            for display_name_iter, path_iter in self.marian_models_dict.items():
                if path_iter == configured_marian_path:
                    initial_marian_display_name = display_name_iter
                    break
        if not initial_marian_display_name and self.marian_models_list:
            initial_marian_display_name = self.marian_models_list[0]
            fallback_path = self.marian_models_dict.get(initial_marian_display_name, "")
            if self.marian_model_var.get() != fallback_path :
                 self.marian_model_var.set(fallback_path)
        self.marian_model_display_var = tk.StringVar(value=initial_marian_display_name)

        # This uses self.translation_model_names, so it must be after its definition
        initial_model_code_for_display = self.translation_model_var.get()
        initial_display_name_for_model_combo = self.translation_model_names.get(initial_model_code_for_display, list(self.translation_model_names.values())[0])
        self.translation_model_display_var = tk.StringVar(value=initial_display_name_for_model_combo)
        active_translation_profile = self.custom_ai_profiles.get_active_profile("translation")
        if active_translation_profile:
            self.custom_translation_profile_display_var.set(active_translation_profile["name"])
            self.translation_model_display_var.set(active_translation_profile["name"])
        active_ocr_profile = self.custom_ai_profiles.get_active_profile("ocr")
        if active_ocr_profile:
            self.custom_ocr_profile_display_var.set(active_ocr_profile["name"])


        self.tab_control = ttk.Notebook(root)
        self.tab_control.pack(expand=True, fill="both", padx=5, pady=5)

        # The tab frames will be created and assigned in the create_*_tab functions
        # We'll temporarily set them to None
        self.tab_main = None
        self.tab_settings = None
        self.tab_custom_prompt = None
        self.tab_debug = None

        active_model_for_init = self.translation_model_var.get()
        initial_source_val, initial_target_val = 'auto', 'en'

        if active_model_for_init == 'custom_ai':
            initial_source_val = self.custom_source_lang
            initial_target_val = self.custom_target_lang
        elif active_model_for_init == 'google_api':
            initial_source_val = self.google_source_lang
            initial_target_val = self.google_target_lang
        elif active_model_for_init == 'deepl_api':
            initial_source_val = self.deepl_source_lang
            initial_target_val = self.deepl_target_lang
        elif active_model_for_init == 'gemini_api':
            initial_source_val = self.gemini_source_lang
            initial_target_val = self.gemini_target_lang
        elif self.is_openai_model(active_model_for_init):
            initial_source_val = self.openai_source_lang
            initial_target_val = self.openai_target_lang
        elif active_model_for_init == 'marianmt':
            if self.marian_model_display_var.get():
                # ui_interaction_handler is now defined
                parsed_marian_langs_init = self.ui_interaction_handler.parse_marian_model_for_langs(self.marian_model_display_var.get()) or \
                                           self.ui_interaction_handler.parse_marian_model_for_langs(self.marian_model_var.get())
                if parsed_marian_langs_init:
                    initial_source_val = parsed_marian_langs_init[0]
                    initial_target_val = parsed_marian_langs_init[1]
                    self.marian_source_lang = initial_source_val
                    self.marian_target_lang = initial_target_val
                else:
                    initial_source_val, initial_target_val = '', ''
            else:
                 initial_source_val, initial_target_val = '', ''


        self.source_lang_var = tk.StringVar(value=initial_source_val)
        self.target_lang_var = tk.StringVar(value=initial_target_val)

        self.lang_code_to_name = self.language_manager

        # Create the main tabs
        create_main_tab(self)
        create_settings_tab(self)
        create_custom_prompt_tab(self)
        create_debug_tab(self)

        # Handle tab change events to set focus appropriately
        def on_tab_changed(event):
            selected_tab_index = self.tab_control.index(self.tab_control.select())
            if selected_tab_index == 0 and hasattr(self, 'main_tab_start_button') and self.main_tab_start_button.winfo_exists():
                self.main_tab_start_button.focus_set()
            elif selected_tab_index == 1 and hasattr(self, 'settings_tab_save_button') and self.settings_tab_save_button.winfo_exists():
                self.settings_tab_save_button.focus_set()

        self.tab_control.bind("<<NotebookTabChanged>>", on_tab_changed)

        self.ui_interaction_handler.on_translation_model_selection_changed(initial_setup=True)

        # Initialize localized dropdowns after everything is set up
        self.root.after(50, self.ui_interaction_handler.update_all_dropdowns_for_language_change)

        self.root.after(100, self.load_initial_overlay_areas)
        self.root.after(200, self.ensure_window_visible)
        self.hotkey_handler.setup_hotkeys()

        # Add periodic network cleanup
        self.setup_network_cleanup()

        log_debug(f"Application initialized. Stability: {self.stable_threshold}")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self._fully_initialized = True
        log_debug("GameChangingTranslator fully initialized.")

        # Ensure OCR model UI is correctly set up on initial load
        if hasattr(self, 'ui_interaction_handler'):
            self.ui_interaction_handler.update_ocr_model_ui()
        self.schedule_initial_paddleocr_prewarm()

    def ensure_window_visible(self):
        """Ensure the main window is visible after all initialization is complete."""
        try:
            if self.root.winfo_exists():
                self.root.deiconify()
                self.root.lift()
                log_debug("Main window visibility ensured after initialization")
        except Exception as e:
            log_debug(f"Error ensuring window visibility: {e}")

    def clear_paddleocr_runtime_cache(self, reason="runtime settings changed"):
        """Release cached PaddleOCR engine instances after settings or lifecycle changes."""
        try:
            self._invalidate_paddleocr_prewarm_state()
            clear_paddleocr_engines()
            log_debug(f"PaddleOCR runtime cache cleared ({reason})")
        except Exception as e:
            log_debug(f"PaddleOCR runtime cache clear failed ({reason}): {e}")

    def _ensure_paddleocr_prewarm_state(self):
        """Create prewarm bookkeeping for lightweight test doubles and old instances."""
        if not hasattr(self, '_paddleocr_prewarm_lock'):
            self._paddleocr_prewarm_lock = threading.RLock()
        if not hasattr(self, '_paddleocr_prewarm_thread'):
            self._paddleocr_prewarm_thread = None
        if not hasattr(self, '_paddleocr_prewarm_settings'):
            self._paddleocr_prewarm_settings = None
        if not hasattr(self, '_paddleocr_prewarmed_settings'):
            self._paddleocr_prewarmed_settings = None
        if not hasattr(self, '_paddleocr_prewarm_generation'):
            self._paddleocr_prewarm_generation = 0
        return self._paddleocr_prewarm_lock

    def _invalidate_paddleocr_prewarm_state(self):
        lock = self._ensure_paddleocr_prewarm_state()
        with lock:
            self._paddleocr_prewarm_generation += 1
            self._paddleocr_prewarm_settings = None
            self._paddleocr_prewarmed_settings = None

    def schedule_initial_paddleocr_prewarm(self):
        """Start local PaddleOCR loading after the UI is up when it is the selected OCR."""
        try:
            selected_ocr_model = self.get_ocr_model_setting()
        except Exception as e:
            log_debug(f"PaddleOCR startup prewarm skipped; OCR model unavailable: {e}")
            return False
        if selected_ocr_model != PADDLEOCR_MODEL_CODE:
            log_debug(
                "PaddleOCR startup prewarm skipped; selected OCR model is "
                f"{selected_ocr_model}"
            )
            return False
        try:
            self.root.after(
                0,
                lambda: self.ensure_paddleocr_ready_if_selected("application startup"),
            )
            log_debug("PaddleOCR startup prewarm scheduled")
            return True
        except Exception as e:
            log_debug(f"PaddleOCR startup prewarm scheduling failed: {e}")
            return self.ensure_paddleocr_ready_if_selected("application startup")

    def ensure_paddleocr_ready_if_selected(self, reason="PaddleOCR selected"):
        """Warm PaddleOCR engines in the background only when the local OCR is selected."""
        try:
            selected_ocr_model = self.get_ocr_model_setting()
        except Exception as e:
            log_debug(f"PaddleOCR prewarm skipped ({reason}); OCR model unavailable: {e}")
            return False
        if selected_ocr_model != PADDLEOCR_MODEL_CODE:
            return False
        try:
            from worker_threads import get_paddleocr_settings_from_app

            settings = get_paddleocr_settings_from_app(self)
        except Exception as e:
            log_debug(f"PaddleOCR prewarm skipped ({reason}); settings unavailable: {e}")
            return False
        return self.start_paddleocr_prewarm(settings, reason)

    def start_paddleocr_prewarm(self, settings, reason="PaddleOCR selected"):
        lock = self._ensure_paddleocr_prewarm_state()
        with lock:
            if self._paddleocr_prewarmed_settings == settings:
                log_debug(f"PaddleOCR prewarm skipped ({reason}); engine already ready")
                return False
            active_thread = self._paddleocr_prewarm_thread
            if (
                active_thread is not None
                and active_thread.is_alive()
                and self._paddleocr_prewarm_settings == settings
            ):
                log_debug(f"PaddleOCR prewarm already running ({reason})")
                return False

            self._paddleocr_prewarm_generation += 1
            generation = self._paddleocr_prewarm_generation
            self._paddleocr_prewarm_settings = settings
            prewarm_thread = threading.Thread(
                target=self._run_paddleocr_prewarm,
                args=(settings, generation, reason),
                name="PaddleOCRPrewarm",
                daemon=True,
            )
            self._paddleocr_prewarm_thread = prewarm_thread

        prewarm_thread.start()
        log_debug(
            "PaddleOCR prewarm started "
            f"reason={reason} version={settings.ocr_version} "
            f"size={settings.model_size} device={settings.device}"
        )
        return True

    def _run_paddleocr_prewarm(self, settings, generation, reason):
        start_time = time.monotonic()
        try:
            get_paddleocr_text_recognition_engine(settings)
            get_paddleocr_engine(settings)
        except Exception as e:
            lock = self._ensure_paddleocr_prewarm_state()
            with lock:
                if generation == self._paddleocr_prewarm_generation:
                    self._paddleocr_prewarmed_settings = None
                    self._paddleocr_prewarm_settings = None
            log_debug(f"PaddleOCR prewarm failed ({reason}): {e}")
            return

        duration = time.monotonic() - start_time
        lock = self._ensure_paddleocr_prewarm_state()
        with lock:
            if generation == self._paddleocr_prewarm_generation:
                self._paddleocr_prewarmed_settings = settings
        log_debug(f"PaddleOCR prewarm completed ({reason}) in {duration:.2f}s")

    def clear_ocr_stability_gate(self, reason="OCR state changed"):
        """Clear pending OCR text that has not yet been submitted for translation."""
        try:
            gate = getattr(self, 'ocr_stability_gate', None)
            clearer = getattr(gate, 'clear', None)
            if callable(clearer) and clearer():
                log_debug(f"OCR stability gate cleared ({reason})")
        except Exception as e:
            log_debug(f"OCR stability gate clear failed ({reason}): {e}")

    def on_ocr_frame_cache_size_change(self, *args):
        """Resize the live OCR frame cache when the setting changes."""
        try:
            new_size = self.ocr_frame_cache_size_var.get()
            if hasattr(self, 'ocr_frame_cache') and hasattr(self.ocr_frame_cache, 'resize'):
                self.ocr_frame_cache.resize(new_size)
            else:
                self.ocr_frame_cache = OCRFrameCache(new_size)
            log_debug(f"OCR frame cache resized to {new_size}")
        except Exception as e:
            log_debug(f"OCR frame cache resize failed: {e}")

    def on_ocr_parameter_change(self, *args):
        """Called when OCR parameters change to refresh preview if it's open."""
        try:
            current_ocr_model = self.get_ocr_model_setting() if hasattr(self, 'get_ocr_model_setting') else PADDLEOCR_MODEL_CODE
            if current_ocr_model == PADDLEOCR_MODEL_CODE:
                self.clear_paddleocr_runtime_cache("OCR parameter changed")
        except Exception as e:
            log_debug(f"Error clearing OCR runtime after OCR parameter change: {e}")
        if self.ocr_preview_window is not None:
            try:
                if self.ocr_preview_window.winfo_exists():
                    # Delay the refresh slightly to avoid too frequent updates
                    if hasattr(self, '_preview_refresh_timer'):
                        self.root.after_cancel(self._preview_refresh_timer)
                    self._preview_refresh_timer = self.root.after(200, self.refresh_ocr_preview)
                else:
                    # Window was destroyed but reference wasn't cleared
                    self.ocr_preview_window = None
            except tk.TclError:
                # Window was destroyed
                self.ocr_preview_window = None

    def on_ocr_model_change(self, *args):
        """Called when OCR model selection changes to update UI visibility."""
        try:
            self.clear_paddleocr_runtime_cache("OCR model changed")
            self.clear_ocr_stability_gate("OCR model changed")

            # End OCR session if switching away from API OCR while translation is running
            if (hasattr(self, 'translation_handler') and self.is_running and
                not self.is_api_based_ocr_model()):
                self.translation_handler.request_end_ocr_session()

            # Start OCR session if switching to custom API OCR while translation is running
            if (hasattr(self, 'translation_handler') and self.is_running and
                self.is_api_based_ocr_model()):
                self.translation_handler.start_ocr_session()

            # Update UI to show/hide OCR model-specific fields
            if hasattr(self, 'ui_interaction_handler'):
                self.ui_interaction_handler.update_ocr_model_ui()

            # Refresh OCR preview if it's open to use the new OCR model
            if self.ocr_preview_window is not None:
                try:
                    if self.ocr_preview_window.winfo_exists():
                        if hasattr(self, '_preview_refresh_timer'):
                            self.root.after_cancel(self._preview_refresh_timer)
                        self._preview_refresh_timer = self.root.after(200, self.refresh_ocr_preview)
                    else:
                        self.ocr_preview_window = None
                except tk.TclError:
                    self.ocr_preview_window = None

            log_debug(f"OCR model changed to: {self.ocr_model_var.get()}")
        except Exception as e:
            log_debug(f"Error in OCR model change callback: {e}")

    def save_settings(self):
        if self._fully_initialized:
            saved = self.ui_interaction_handler.save_settings()
            if saved and not getattr(self, "_app_is_closing", False):
                try:
                    if self.get_ocr_model_setting() == PADDLEOCR_MODEL_CODE:
                        self.ensure_paddleocr_ready_if_selected("settings saved")
                except Exception as e:
                    log_debug(f"PaddleOCR prewarm after settings save failed: {e}")
            return saved
        log_debug("Attempted to save settings before full initialization.")
        return False

    def suppress_traces(self):
        """Suppress StringVar traces during UI updates to prevent cascading saves"""
        self._suppress_traces = True
        log_debug("StringVar traces suppressed")

    def restore_traces(self):
        """Restore StringVar traces after UI updates complete"""
        self._suppress_traces = False
        log_debug("StringVar traces restored")

    def start_ui_update(self):
        """Mark the start of a UI update operation to suppress all saves"""
        self._ui_update_in_progress = True
        self.suppress_traces()
        log_debug("UI update operation started - all saves suppressed")

    def end_ui_update(self):
        """Mark the end of a UI update operation and restore normal save behavior"""
        self._ui_update_in_progress = False
        self.restore_traces()
        log_debug("UI update operation ended - saves restored")

    def on_window_configure(self, event):
        self.configuration_handler.on_window_configure(event)

    def save_current_window_geometry(self):
        self.configuration_handler.save_current_window_geometry()

    def browse_marian_models_file(self):
        self.configuration_handler.browse_marian_models_file()

    def update_translation_text(self, text_to_display):
        self.display_manager.update_translation_text(text_to_display)

    def update_debug_display(self, original_img_pil, processed_img_cv, ocr_text_content):
        self.display_manager.update_debug_display(original_img_pil, processed_img_cv, ocr_text_content)

    def _widget_exists_safely(self, widget):
        """Safely check if a widget exists - works with both tkinter and PySide widgets"""
        if not widget:
            return False
        try:
            # Try tkinter method first
            if hasattr(widget, 'winfo_exists'):
                return widget.winfo_exists()
            # For PySide widgets, check if they're accessible
            elif hasattr(widget, 'isVisible'):
                return True  # PySide widgets exist until destroyed
            else:
                return True  # Assume widget exists if we can't check
        except Exception as e:
            log_debug(f"Error checking widget existence: {e}")
            return False

    def convert_to_api_ocr_image(self, pil_image):
        """Convert a PIL image to configured API OCR bytes plus MIME metadata."""
        format_getter = getattr(self, 'get_custom_ai_ocr_image_format', None)
        mode_getter = getattr(self, 'get_custom_ai_ocr_image_mode', None)
        quality_getter = getattr(self, 'get_custom_ai_ocr_image_quality', None)
        detail_getter = getattr(self, 'get_custom_ai_ocr_image_detail', None)
        if callable(format_getter):
            image_format = format_getter()
        else:
            format_var = getattr(self, 'custom_ai_ocr_image_format_var', None)
            image_format = normalize_api_ocr_image_format(format_var.get() if format_var is not None else API_OCR_IMAGE_FORMAT_DEFAULT)
        if callable(mode_getter):
            mode = mode_getter()
        else:
            mode_var = getattr(self, 'custom_ai_ocr_image_mode_var', None)
            mode = normalize_api_ocr_image_mode(mode_var.get() if mode_var is not None else API_OCR_IMAGE_MODE_DEFAULT)
        if callable(quality_getter):
            quality = quality_getter()
        else:
            quality_var = getattr(self, 'custom_ai_ocr_image_quality_var', None)
            quality = normalize_api_ocr_image_quality(quality_var.get() if quality_var is not None else API_OCR_IMAGE_QUALITY_DEFAULT)
        if callable(detail_getter):
            detail = detail_getter()
        else:
            detail_var = getattr(self, 'custom_ai_ocr_image_detail_var', None)
            detail = normalize_api_ocr_image_detail(detail_var.get() if detail_var is not None else API_OCR_IMAGE_DETAIL_DEFAULT)
        start = time.monotonic()

        try:
            encoded_image = encode_image_for_api_ocr_payload(
                pil_image,
                mode=mode,
                quality=quality,
                image_format=image_format,
            )
        except Exception as e:
            log_debug(
                "API OCR image encoding failed "
                f"format={image_format} mode={mode} quality={quality} detail={detail}: "
                f"{type(e).__name__} - {e}; retrying format=webp mode=lossless_webp"
            )
            try:
                image_format = 'webp'
                mode = 'lossless_webp'
                encoded_image = encode_image_for_api_ocr_payload(
                    pil_image,
                    mode=mode,
                    quality=quality,
                    image_format=image_format,
                )
            except Exception as fallback_error:
                log_debug(
                    "API OCR image encoding failed "
                    f"format=webp mode=lossless_webp quality={quality} detail={detail}: "
                    f"{type(fallback_error).__name__} - {fallback_error}"
                )
                return None

        duration = time.monotonic() - start
        log_debug(
            "API OCR image encoded "
            f"format={encoded_image.image_format} mime={encoded_image.mime_type} "
            f"mode={mode} bytes={len(encoded_image.data)} detail={detail} duration={duration:.3f}s"
        )
        return encoded_image

    def convert_to_webp_for_api(self, pil_image):
        """Convert a PIL image for API OCR calls and return bytes for legacy callers."""
        encoded_image = GameChangingTranslator.convert_to_api_ocr_image(self, pil_image)
        return encoded_image.data if encoded_image is not None else None

    def _pre_initialize_gemini_model(self):
        """Pre-configure Gemini API at startup to avoid thread initialization delays."""
        log_debug("Gemini pre-initialization skipped; built-in Gemini provider is disabled.")

    # Gemini OCR Batch Processing Methods (Phase 1)
    def get_ocr_model_setting(self):
        """Get the current OCR model setting."""
        return self.ocr_model_var.get()

    def update_adaptive_scan_interval(self):
        """Adjust scan interval based on current OCR API load to prevent bottlenecks."""
        now = time.monotonic()

        # Check load every 2 seconds
        if now - self.load_check_timer < 2.0:
            return

        self.load_check_timer = now

        # Measure current OCR load
        active_ocr_count = len(self.active_ocr_calls)
        max_ocr_calls = self.max_concurrent_ocr_calls

        # Get user's preferred base interval
        base_interval = self.scan_interval_var.get()  # User's setting in milliseconds

        # Update base_scan_interval to track user changes
        self.base_scan_interval = base_interval

        # Apply the user's specific requirements:
        # If active OCR API calls > 5, increase scan interval to 150% of current value
        # If active OCR API calls fall below 5, restore original scan interval
        if active_ocr_count > 5:
            adaptive_state = "overloaded"
        elif active_ocr_count < 5:
            adaptive_state = "normal"
        else:
            adaptive_state = "moderate"

        previous_log_state = getattr(self, "_last_adaptive_log_state", None)
        previous_log_time = getattr(self, "_last_adaptive_log_time", 0.0)
        should_log_state = (
            adaptive_state != previous_log_state
            or now - previous_log_time >= 30.0
        )
        adaptive_log_message = None

        if active_ocr_count > 5:
            if not self.overload_detected:
                # First detection of overload
                self.current_scan_interval = int(base_interval * 1.5)  # 150%
                self.overload_detected = True
                adaptive_log_message = (
                    f"ADAPTIVE: OCR overload detected ({active_ocr_count} active calls), "
                    f"increasing scan interval to {self.current_scan_interval}ms"
                )
            elif should_log_state:
                # Already in overload state, maintain increased interval
                adaptive_log_message = (
                    f"ADAPTIVE: OCR still overloaded ({active_ocr_count} active calls), "
                    f"maintaining scan interval at {self.current_scan_interval}ms"
                )
            # Stay at increased interval while overloaded

        elif active_ocr_count < 5:
            if self.overload_detected:
                # Load has decreased, return to normal
                self.current_scan_interval = base_interval
                self.overload_detected = False
                adaptive_log_message = (
                    f"ADAPTIVE: OCR load normalized ({active_ocr_count} active calls), "
                    f"returning scan interval to {self.current_scan_interval}ms"
                )
            elif should_log_state:
                # Normal state, no change needed
                adaptive_log_message = (
                    f"ADAPTIVE: OCR load normal ({active_ocr_count} active calls), "
                    f"scan interval remains at {self.current_scan_interval}ms"
                )
        else:
            # At exactly 5 calls, maintain current state
            if should_log_state:
                adaptive_log_message = (
                    f"ADAPTIVE: OCR load moderate ({active_ocr_count} active calls), "
                    f"scan interval unchanged at {self.current_scan_interval}ms"
                )

        if adaptive_log_message:
            log_debug(adaptive_log_message)
            self._last_adaptive_log_state = adaptive_state
            self._last_adaptive_log_time = now

    def handle_empty_ocr_result(self):
        """Handle <EMPTY> OCR result and manage clear translation timeout."""
        current_time = time.monotonic()

        # Only start timeout if we have a timeout value configured
        if self.clear_translation_timeout_var.get() <= 0:
            return  # Timeout disabled, do nothing

        if self.clear_timeout_timer_start is None:
            # First EMPTY result - start timer
            self.clear_timeout_timer_start = current_time
            log_debug("Clear timeout timer started for <EMPTY> OCR result")
        else:
            # Check if timeout period exceeded
            elapsed = current_time - self.clear_timeout_timer_start
            timeout_seconds = self.clear_translation_timeout_var.get()

            if elapsed >= timeout_seconds:
                # Clear the translation display
                self.update_translation_text("")
                self.last_local_ocr_submitted_text = None
                self.last_local_ocr_submitted_norm = None
                self.last_local_ocr_submitted_scope = None
                self.reset_clear_timeout()
                log_debug(f"Translation cleared after {elapsed:.1f}s timeout")

    def handle_successive_identical_subtitle(self, reason):
        """Handle identical subtitles that are the SAME as the immediately previous one."""
        # 1. Do NOT update caches (LRU, file cache) - no new content
        # 2. Do NOT update context window - successive identical subtitle
        # 3. Keep displaying last translation (no API call needed)
        # 4. Reset clear timeout (text is still present)

        self.reset_clear_timeout()  # Text still present
        # Display remains unchanged (last translation stays)
        # self.last_processed_subtitle stays the same (no change)
        log_debug(f"Successive identical subtitle detected ({reason}), maintaining current translation")
        # No context window update - subtitle hasn't changed

    def reset_clear_timeout(self):
        """Reset clear translation timeout timer."""
        self.clear_timeout_timer_start = None
        log_debug("Clear timeout timer reset - text detected")

    def initialize_async_translation_infrastructure(self):
        """Initialize async translation infrastructure if not already present."""
        if not hasattr(self, 'translation_sequence_counter'):
            self.translation_sequence_counter = 0
            log_debug("Initialized translation_sequence_counter")

        if not hasattr(self, 'last_displayed_translation_sequence'):
            self.last_displayed_translation_sequence = 0
            log_debug("Initialized last_displayed_translation_sequence")

        if not hasattr(self, 'active_translation_calls'):
            self.active_translation_calls = set()
            log_debug("Initialized active_translation_calls")

        if not hasattr(self, 'active_translation_inflight_keys'):
            self.active_translation_inflight_keys = set()
            log_debug("Initialized active_translation_inflight_keys")

        if not hasattr(self, 'active_translation_started_monotonic'):
            self.active_translation_started_monotonic = {}
            log_debug("Initialized active_translation_started_monotonic")

        if not hasattr(self, 'max_concurrent_translation_calls'):
            self.max_concurrent_translation_calls = 6
            log_debug("Initialized max_concurrent_translation_calls")

        if not hasattr(self, 'translation_supersede_after_seconds'):
            self.translation_supersede_after_seconds = 1.5
            log_debug("Initialized translation_supersede_after_seconds")

        if not hasattr(self, 'last_translation_submit_monotonic'):
            self.last_translation_submit_monotonic = 0.0
            log_debug("Initialized last_translation_submit_monotonic")

        if not hasattr(self, 'pending_translation_request'):
            self.pending_translation_request = None
            log_debug("Initialized pending_translation_request")

        if not hasattr(self, 'pending_translation_flush_scheduled'):
            self.pending_translation_flush_scheduled = False
            log_debug("Initialized pending_translation_flush_scheduled")

        if not hasattr(self, 'pending_translation_flush_deadline_monotonic'):
            self.pending_translation_flush_deadline_monotonic = 0.0
            log_debug("Initialized pending_translation_flush_deadline_monotonic")

        if not hasattr(self, 'pending_translation_flush_generation'):
            self.pending_translation_flush_generation = 0
            log_debug("Initialized pending_translation_flush_generation")

    def check_clear_timeout(self):
        """Check if clear timeout should be triggered and return True if timeout exceeded."""
        if self.clear_timeout_timer_start is None:
            return False

        if self.clear_translation_timeout_var.get() <= 0:
            return False  # Timeout disabled

        current_time = time.monotonic()
        elapsed = current_time - self.clear_timeout_timer_start
        timeout_seconds = self.clear_translation_timeout_var.get()

        return elapsed >= timeout_seconds

    def translate_text(self, text_content):
        return self.translation_handler.translate_text(text_content)

    def is_placeholder_text(self, text_content):
        return self.translation_handler.is_placeholder_text(text_content)

    def calculate_text_similarity(self, text1, text2):
        return self.translation_handler.calculate_text_similarity(text1, text2)

    def update_marian_active_model(self, model_name, source_lang=None, target_lang=None):
        return self.translation_handler.update_marian_active_model(model_name, source_lang, target_lang)

    def update_marian_beam_value(self):
        self.translation_handler.update_marian_beam_value()

    def choose_color_for_settings(self, color_type):
        self.ui_interaction_handler.choose_color_for_settings(color_type)

    def update_stability_from_spinbox(self):
        self.ui_interaction_handler.update_stability_from_spinbox()

    def update_target_font_size(self):
        self.ui_interaction_handler.update_target_font_size()

    def update_target_font_type(self):
        self.ui_interaction_handler.update_target_font_type()

    def update_target_opacity(self):
        self.ui_interaction_handler.update_target_opacity()

    def update_target_text_opacity(self):
        self.ui_interaction_handler.update_target_text_opacity()

    def refresh_debug_log(self):
        self.ui_interaction_handler.refresh_debug_log()

    def save_debug_images(self):
        self.ui_interaction_handler.save_debug_images()

    def toggle_api_key_visibility(self, api_type):
        self.ui_interaction_handler.toggle_api_key_visibility(api_type)

    def update_translation_model_ui(self):
        self.ui_interaction_handler.update_translation_model_ui()

    def on_marian_model_selection_changed(self, event=None, preload=False, initial_setup=False):
        self.ui_interaction_handler.on_marian_model_selection_changed(event, preload, initial_setup)
        if not initial_setup and self._fully_initialized :
             self.save_settings()


    def on_translation_model_selection_changed(
        self,
        event=None,
        initial_setup=False,
        synchronize_ui_only=False,
    ):
        # Handle session management for translation method changes
        if (
            not synchronize_ui_only
            and hasattr(self, 'translation_handler')
            and self.is_running
            and not initial_setup
        ):
            current_model = self.translation_model_var.get()

            # End translation session if switching away from Gemini
            if current_model != 'gemini_api':
                self.translation_handler.request_end_translation_session()

            # Start translation session if switching to Gemini
            if current_model == 'gemini_api':
                self.translation_handler.start_translation_session()

            # Handle OpenAI session management if needed
            if current_model == 'openai_api':
                # OpenAI doesn't require special session management like Gemini
                # But we could add any OpenAI-specific initialization here if needed
                pass
                self.translation_handler.start_translation_session()

        self.ui_interaction_handler.on_translation_model_selection_changed(event, initial_setup)
        if (
            not synchronize_ui_only
            and not initial_setup
            and self._fully_initialized
        ):
            self.save_settings()

    def clear_debug_log(self):
        self.ui_interaction_handler.clear_debug_log()

    def reset_gemini_api_log(self):
        """Reset/clear the Gemini API call log file."""
        try:
            if hasattr(self.translation_handler, 'gemini_log_file'):
                log_file_path = self.translation_handler.gemini_log_file

                # Clear the file by truncating it
                if os.path.exists(log_file_path):
                    with open(log_file_path, 'w', encoding='utf-8') as f:
                        f.write('')  # Clear the file
                    log_debug(f"Gemini API log file cleared: {log_file_path}")

                    # Reinitialize the log with header
                    if hasattr(self.translation_handler, '_initialize_gemini_log'):
                        self.translation_handler._initialize_gemini_log()

                    messagebox.showinfo(
                        self.ui_lang.get_label("gemini_reset_success_title", "Success"),
                        self.ui_lang.get_label("gemini_reset_success_msg", "Gemini API log has been reset.")
                    )
                else:
                    log_debug(f"Gemini API log file does not exist: {log_file_path}")
                    messagebox.showwarning(
                        self.ui_lang.get_label("gemini_reset_warning_title", "Warning"),
                        self.ui_lang.get_label("gemini_reset_warning_msg", "Gemini API log file does not exist.")
                    )
            else:
                log_debug("Gemini log file path not available")
                messagebox.showerror(
                    self.ui_lang.get_label("gemini_reset_error_title", "Error"),
                    self.ui_lang.get_label("gemini_reset_error_msg", "Could not access Gemini log file.")
                )
        except Exception as e:
            log_debug(f"Error resetting Gemini API log: {e}")
            messagebox.showerror(
                self.ui_lang.get_label("gemini_reset_error_title", "Error"),
                f"{self.ui_lang.get_label('gemini_reset_error_failed', 'Failed to reset Gemini API log:')} {str(e)}"
            )

    def update_openai_stats(self):
        """Update the OpenAI statistics fields by reading the log file."""
        try:
            # Check if all required components are available
            if not hasattr(self, 'openai_total_words_var') or self.openai_total_words_var is None:
                log_debug("OpenAI stats variables not initialized yet")
                return

            if not hasattr(self, 'openai_total_cost_var') or self.openai_total_cost_var is None:
                log_debug("OpenAI total cost variable not initialized yet")
                return

            # Get cumulative totals from OpenAI log file
            total_words, total_cost = self._get_cumulative_openai_totals()

            # Update GUI fields
            self.openai_total_words_var.set(self.format_number_with_separators(total_words))
            self.openai_total_cost_var.set(self.format_cost_for_display(total_cost))

            log_debug(f"Updated OpenAI stats: {total_words} words, ${total_cost:.8f}")
        except Exception as e:
            log_debug(f"Error updating OpenAI stats: {e}")
            # Set default values if there's an error
            if hasattr(self, 'openai_total_words_var') and self.openai_total_words_var is not None:
                self.openai_total_words_var.set(self.format_number_with_separators(0))
            if hasattr(self, 'openai_total_cost_var') and self.openai_total_cost_var is not None:
                self.openai_total_cost_var.set(self.format_cost_for_display(0.0))

    def _get_cumulative_openai_totals(self):
        """Read the cumulative totals from the OpenAI API log file."""
        try:
            # Get the log file path
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))

            openai_log_file = os.path.join(base_dir, "OpenAI_API_call_logs.txt")

            if not os.path.exists(openai_log_file):
                log_debug(f"OpenAI log file does not exist: {openai_log_file}")
                return 0, 0.0

            # Read the most recent cumulative cost and words from the log
            cumulative_cost = 0.0
            cumulative_words = 0

            with open(openai_log_file, 'r', encoding='utf-8') as f:
                content = f.read()

                # Find all instances of cumulative totals
                cost_matches = re.findall(r'Cumulative Log Cost: \$([0-9.]+)', content)
                word_matches = re.findall(r'Total Translated Words \(so far\): ([0-9,]+)', content)

                if cost_matches:
                    cumulative_cost = float(cost_matches[-1])  # Get the last (most recent) value

                if word_matches:
                    # Remove commas from word count and convert to int
                    word_str = word_matches[-1].replace(',', '')
                    cumulative_words = int(word_str)

            log_debug(f"OpenAI cumulative totals: {cumulative_words} words, ${cumulative_cost:.8f}")
            return cumulative_words, cumulative_cost

        except Exception as e:
            log_debug(f"Error reading OpenAI cumulative totals: {e}")
            return 0, 0.0

    def reset_openai_api_log(self):
        """Reset/clear the OpenAI API call log file."""
        try:
            if hasattr(self.translation_handler, 'openai_log_file'):
                log_file_path = self.translation_handler.openai_log_file

                # Clear the file by truncating it
                if os.path.exists(log_file_path):
                    with open(log_file_path, 'w', encoding='utf-8') as f:
                        f.write('')  # Clear the file
                    log_debug(f"OpenAI API log file cleared: {log_file_path}")

                    # Reinitialize the log with header
                    if hasattr(self.translation_handler, '_initialize_openai_log'):
                        self.translation_handler._initialize_openai_log()

                    # Update the GUI fields
                    self.update_openai_stats()

                    messagebox.showinfo(
                        self.ui_lang.get_label("openai_reset_success_title", "Success"),
                        self.ui_lang.get_label("openai_reset_success_msg", "OpenAI API log has been reset.")
                    )
                else:
                    log_debug(f"OpenAI API log file does not exist: {log_file_path}")
                    messagebox.showwarning(
                        self.ui_lang.get_label("openai_reset_warning_title", "Warning"),
                        self.ui_lang.get_label("openai_reset_warning_msg", "OpenAI API log file does not exist.")
                    )
            else:
                log_debug("OpenAI log file path not available")
                messagebox.showerror(
                    self.ui_lang.get_label("openai_reset_error_title", "Error"),
                    self.ui_lang.get_label("openai_reset_error_msg", "Could not access OpenAI log file.")
                )
        except Exception as e:
            log_debug(f"Error resetting OpenAI API log: {e}")
            messagebox.showerror(
                self.ui_lang.get_label("openai_reset_error_title", "Error"),
                f"{self.ui_lang.get_label('openai_reset_error_failed', 'Failed to reset OpenAI API log:')} {str(e)}"
            )

    def format_currency_for_display(self, amount, unit_suffix=""):
        """Format currency amount according to current UI language."""
        try:
            prefix = "$" if not unit_suffix else "$"
            return f"{prefix}{amount:.8f}{unit_suffix}"
        except Exception as e:
            log_debug(f"Error formatting currency: {e}")
            return f"${amount:.8f}{unit_suffix}"

    def format_cost_for_display(self, cost_value):
        """Format cost value for display."""
        return self.format_currency_for_display(cost_value, "")

    def format_number_with_separators(self, number):
        """Format integer numbers with thousand separators according to current UI language."""
        try:
            # Convert to integer to avoid decimal formatting issues
            num = int(number)

            return f"{num:,}"
        except Exception as e:
            log_debug(f"Error formatting number with separators: {e}")
            return str(number)  # Fallback to string representation

    def toggle_debug_logging(self):
        """Toggle debug logging on/off and update button text."""
        current_state = self.debug_logging_enabled_var.get()
        new_state = not current_state

        # Log state change before changing the state
        if current_state:
            log_debug("Debug logging disabled by user")

        # Update the state
        self.debug_logging_enabled_var.set(new_state)
        set_debug_logging_enabled(new_state)

        # Log state change after enabling (if we're enabling)
        if new_state:
            log_debug("Debug logging enabled by user")

        # Update button text
        if hasattr(self, 'debug_log_toggle_btn') and self.debug_log_toggle_btn.winfo_exists():
            if new_state:
                button_text = self.ui_lang.get_label("toggle_debug_log_disable_btn")
            else:
                button_text = self.ui_lang.get_label("toggle_debug_log_enable_btn")
            self.debug_log_toggle_btn.config(text=button_text)

        # Save settings
        if self._fully_initialized:
            self.save_settings()

    def show_ocr_preview(self):
        """Show/create the OCR Preview window."""
        # Check if window already exists and is valid
        if self.ocr_preview_window is not None:
            try:
                if self.ocr_preview_window.winfo_exists():
                    # Window already exists, just bring to front
                    self.ocr_preview_window.lift()
                    self.ocr_preview_window.attributes('-topmost', True)
                    self.ocr_preview_window.after(100, lambda: self.ocr_preview_window.attributes('-topmost', False))
                    return
            except tk.TclError:
                # Window was destroyed but variable wasn't cleared
                self.ocr_preview_window = None

        # Create new preview window
        self.ocr_preview_window = tk.Toplevel(self.root)
        self.ocr_preview_window.title(self.ui_lang.get_label("ocr_preview_title", "OCR Preview"))
        self.ocr_preview_window.minsize(400, 500)

        # Load window geometry from config
        load_ocr_preview_geometry(self.config, self.ocr_preview_window)

        # Create main frame
        main_frame = ttk.Frame(self.ocr_preview_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Image section - with horizontal scrollbar (no extra space) - NEW APPROACH
        image_frame = ttk.LabelFrame(main_frame, text=self.ui_lang.get_label("processed_image_preview", "Processed Image (1:1 scale)"))
        image_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Create frame for image content that won't expand
        content_frame = ttk.Frame(image_frame)
        content_frame.pack(fill="x", padx=5, pady=5)

        # Create canvas with scrollbars - but don't let it expand vertically
        image_canvas = tk.Canvas(content_frame, bd=0, highlightthickness=0, relief='flat', height=200)
        style_tk_canvas(image_canvas, self.md3_palette)
        h_scrollbar = ttk.Scrollbar(content_frame, orient="horizontal", command=image_canvas.xview)
        v_scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=image_canvas.yview)

        image_canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        # Pack with no expand for vertical
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        image_canvas.pack(side="left", fill="both", expand=True)

        # Create label inside canvas for image display
        self.preview_image_label = ttk.Label(image_canvas, text=self.ui_lang.get_label("no_image_processed", "No image processed yet"),
                                            anchor="center", justify="center")

        # Add label to canvas
        self.preview_image_canvas_item = image_canvas.create_window(0, 0, anchor="nw", window=self.preview_image_label)

        # Store canvas reference for updating scroll region
        self.preview_image_canvas = image_canvas

        # Bind canvas resize to update scroll region
        def on_canvas_configure(event):
            # Update the scroll region to encompass the image
            image_canvas.configure(scrollregion=image_canvas.bbox("all"))

        image_canvas.bind('<Configure>', on_canvas_configure)

        # Text section
        text_frame = ttk.LabelFrame(main_frame, text=self.ui_lang.get_label("recognized_text_preview", "Recognized Text"))
        text_frame.pack(fill="x", padx=5, pady=5)

        self.preview_text_widget = tk.Text(text_frame, height=8, wrap=tk.WORD)
        style_tk_text_widget(self.preview_text_widget, self.md3_palette)
        text_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.preview_text_widget.yview)
        self.preview_text_widget.configure(yscrollcommand=text_scrollbar.set)

        text_scrollbar.pack(side="right", fill="y")
        self.preview_text_widget.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=5)

        ttk.Button(button_frame, text=self.ui_lang.get_label("refresh_preview", "Refresh Preview"),
                  command=self.refresh_ocr_preview).pack(side="left", padx=5)
        ttk.Button(button_frame, text=self.ui_lang.get_label("close_btn", "Close"),
                  command=self.close_ocr_preview).pack(side="right", padx=5)

        # Set up proper window close protocol
        self.ocr_preview_window.protocol("WM_DELETE_WINDOW", self.close_ocr_preview)

        # Set up window geometry saving on window events
        def on_preview_configure(event):
            if event.widget == self.ocr_preview_window:
                # Save geometry when window is moved or resized
                if hasattr(self, '_preview_geometry_timer'):
                    self.root.after_cancel(self._preview_geometry_timer)
                self._preview_geometry_timer = self.root.after(500, self.save_preview_geometry)

        self.ocr_preview_window.bind('<Configure>', on_preview_configure)

        # Start continuous real-time updates (regardless of translation state)
        self.start_preview_realtime_updates()

        # Initial preview update
        self.refresh_ocr_preview()

    def close_ocr_preview(self):
        """Properly close the OCR Preview window."""
        if self.ocr_preview_window is not None:
            try:
                # Save window geometry before closing
                self.save_preview_geometry()

                # Cancel any pending refresh timer
                if hasattr(self, '_preview_refresh_timer'):
                    self.root.after_cancel(self._preview_refresh_timer)

                # Cancel geometry save timer
                if hasattr(self, '_preview_geometry_timer'):
                    self.root.after_cancel(self._preview_geometry_timer)

                # Stop real-time updates
                self.stop_preview_realtime_updates()

                # Destroy the window
                self.ocr_preview_window.destroy()
            except tk.TclError:
                # Window might already be destroyed
                pass
            finally:
                # Always clear the reference
                self.ocr_preview_window = None

    def save_preview_geometry(self):
        """Save OCR Preview window geometry to config."""
        if self.ocr_preview_window is not None:
            try:
                save_ocr_preview_geometry(self.config, self.ocr_preview_window)
                save_app_config(self.config)
            except Exception as e:
                log_debug(f"Error saving OCR Preview geometry: {e}")

    def start_preview_realtime_updates(self):
        """Start continuous real-time updates for OCR Preview window regardless of translation state."""
        if self.ocr_preview_window is not None:
            try:
                if self.ocr_preview_window.winfo_exists():
                    # Update every 500ms continuously (both when translation is running and stopped)
                    self._preview_realtime_timer = self.root.after(500, self.preview_realtime_update)
                else:
                    self.ocr_preview_window = None
            except tk.TclError:
                self.ocr_preview_window = None

    def stop_preview_realtime_updates(self):
        """Stop real-time updates for OCR Preview window."""
        if hasattr(self, '_preview_realtime_timer'):
            self.root.after_cancel(self._preview_realtime_timer)
            delattr(self, '_preview_realtime_timer')

    def preview_realtime_update(self):
        """Real-time update function for OCR Preview window - works regardless of translation state."""
        if self.ocr_preview_window is not None:
            try:
                if self.ocr_preview_window.winfo_exists():
                    # Refresh preview with current data (works both when translation is on/off)
                    self.refresh_ocr_preview()
                    # Schedule next update
                    self._preview_realtime_timer = self.root.after(500, self.preview_realtime_update)
                else:
                    self.ocr_preview_window = None
            except tk.TclError:
                self.ocr_preview_window = None

    def refresh_ocr_preview(self):
        """Refresh the OCR preview with current settings and captured image."""
        # Check if window still exists
        if self.ocr_preview_window is None:
            return

        try:
            if not self.ocr_preview_window.winfo_exists():
                self.ocr_preview_window = None
                return
        except tk.TclError:
            # Window was destroyed
            self.ocr_preview_window = None
            return

        try:
            # Always try to capture from source area for real-time preview (independent of translation state)
            screenshot_pil = None
            if self.source_overlay and self.source_overlay.winfo_exists():
                try:
                    area = self.source_overlay.get_geometry()
                    if area:
                        x1, y1, x2, y2 = map(int, area)
                        width, height = x2-x1, y2-y1
                        if width > 0 and height > 0:
                            import pyautogui
                            screenshot_pil = pyautogui.screenshot(region=(x1, y1, width, height))
                        else:
                            screenshot_pil = None
                    else:
                        screenshot_pil = None
                except Exception as e:
                    log_debug(f"Error capturing for preview: {e}")
                    screenshot_pil = None

            # Fallback to using last_screenshot only if direct capture failed
            if screenshot_pil is None and hasattr(self, 'last_screenshot') and self.last_screenshot:
                screenshot_pil = self.last_screenshot

            if screenshot_pil:
                from PIL import Image, ImageTk
                current_ocr_model = self.get_ocr_model_setting()

                if current_ocr_model == PADDLEOCR_MODEL_CODE:
                    from worker_threads import get_paddleocr_settings_from_app

                    paddleocr_settings = get_paddleocr_settings_from_app(self)
                    processed_pil = prepare_paddleocr_image(screenshot_pil, paddleocr_settings)
                    ocr_cleaned_text, _lines = recognize_with_paddleocr(
                        screenshot_pil,
                        paddleocr_settings,
                        keep_linebreaks=bool(self.keep_linebreaks_var.get()),
                    )
                else:
                    processed_pil = screenshot_pil.convert("RGB")
                    ocr_cleaned_text = self.ui_lang.get_label(
                        "ocr_preview_local_only",
                        "OCR preview is available for PaddleOCR.",
                    )

                # Convert processed image to PIL for display
                processed_tk = ImageTk.PhotoImage(processed_pil)

                # Update image display in canvas
                self.preview_image_label.configure(image=processed_tk, text="")
                self.preview_image_label.image = processed_tk  # Keep reference

                # Update canvas scroll region to fit the image
                self.preview_image_label.update_idletasks()  # Ensure label has correct size
                image_width = processed_tk.width()
                image_height = processed_tk.height()

                # Adjust canvas height to fit image (with reasonable limits)
                canvas_height = min(image_height, 400)  # Max height of 400 pixels
                self.preview_image_canvas.configure(height=canvas_height)

                # Update the canvas window size and scroll region
                self.preview_image_canvas.itemconfig(self.preview_image_canvas_item, width=image_width, height=image_height)
                self.preview_image_canvas.configure(scrollregion=(0, 0, image_width, image_height))

                # Update text display
                self.preview_text_widget.config(state=tk.NORMAL)
                self.preview_text_widget.delete(1.0, tk.END)
                self.preview_text_widget.insert(tk.END, ocr_cleaned_text if ocr_cleaned_text else self.ui_lang.get_label("no_text_recognized", "No text recognized"))
                self.preview_text_widget.config(state=tk.DISABLED)

            else:
                # No image available
                self.preview_image_label.configure(image="", text=self.ui_lang.get_label("no_image_captured", "No image captured yet"))
                self.preview_image_label.image = None

                # Reset canvas to default size for text display
                self.preview_image_canvas.configure(height=100)  # Small height for text

                # Reset canvas scroll region for text display
                self.preview_image_label.update_idletasks()
                label_width = self.preview_image_label.winfo_reqwidth()
                label_height = self.preview_image_label.winfo_reqheight()

                self.preview_image_canvas.itemconfig(self.preview_image_canvas_item, width=label_width, height=label_height)
                self.preview_image_canvas.configure(scrollregion=(0, 0, label_width, label_height))

                self.preview_text_widget.config(state=tk.NORMAL)
                self.preview_text_widget.delete(1.0, tk.END)
                self.preview_text_widget.insert(tk.END, self.ui_lang.get_label("no_image_for_ocr", "No image available for OCR"))
                self.preview_text_widget.config(state=tk.DISABLED)

        except Exception as e:
            log_debug(f"Error refreshing OCR preview: {e}")
            # Show error in preview
            if hasattr(self, 'preview_text_widget') and self.preview_text_widget.winfo_exists():
                self.preview_text_widget.config(state=tk.NORMAL)
                self.preview_text_widget.delete(1.0, tk.END)
                self.preview_text_widget.insert(tk.END, f"Error: {str(e)}")
                self.preview_text_widget.config(state=tk.DISABLED)

    def load_initial_overlay_areas(self):
        load_areas_from_config_om(self)

    def select_source_area(self):
        select_source_area_om(self)
        self.save_settings()

    def select_target_area(self):
        select_target_area_om(self)
        self.save_settings()

    def create_source_overlay(self):
        create_source_overlay_om(self)

    def create_target_overlay(self):
        create_target_overlay_om(self)  # System recreation, preserve position

    def toggle_source_visibility(self):
        toggle_source_visibility_om(self)
        self.save_settings()

    def toggle_target_visibility(self):
        toggle_target_visibility_om(self)
        self.save_settings()

    def clear_file_caches(self):
        self.cache_manager.clear_file_caches()

    def clear_cache(self):
        """Clear unified translation cache - FIXED VERSION (No pause/resume needed)."""
        try:
            log_debug("Clearing unified translation cache...")

            # Notify MarianMT translator about cache clearing FIRST
            if hasattr(self, 'marian_translator') and self.marian_translator:
                try:
                    if hasattr(self.marian_translator, 'notify_cache_cleared'):
                        self.marian_translator.notify_cache_cleared()
                        log_debug("Notified MarianMT translator about cache clearing.")
                    else:
                        log_debug("MarianMT translator does not have notify_cache_cleared method.")
                except Exception as e_notify:
                    log_debug(f"Error notifying MarianMT about cache clearing: {e_notify}")

            # Clear unified cache (thread-safe, no need to pause translation)
            self.translation_handler.clear_cache()

            # Clear in-memory file cache representations (Level 2 persistence remains)
            self.google_file_cache.clear()
            self.deepl_file_cache.clear()
            log_debug("Cleared in-memory representations of file caches.")

            if hasattr(self, 'ocr_frame_cache'):
                self.ocr_frame_cache.clear()
                log_debug("Cleared OCR frame cache.")

            # Clear queues
            self._clear_queue(self.ocr_queue)
            self._clear_queue(self.translation_queue)
            log_debug("Cleared OCR and translation queues.")

            # Reset text processing state
            self.text_stability_counter = 0
            self.previous_text = ""
            self.last_processed_subtitle = None
            self.last_local_ocr_submitted_text = None
            self.last_local_ocr_submitted_norm = None
            self.last_local_ocr_submitted_scope = None
            self.clear_ocr_stability_gate("cache cleared")
            if hasattr(self, 'active_translation_inflight_keys'):
                self.active_translation_inflight_keys.clear()
            self.translation_cache.clear()
            log_debug("Unified translation cache and related states cleared successfully.")

            # Update status briefly
            original_status_text = self.status_label.cget("text")
            self.status_label.config(text="Status: Cache cleared")
            if self.root.winfo_exists():
                self.root.after(2000, lambda: self.status_label.config(text=original_status_text) if self.status_label.winfo_exists() else None)

        except Exception as e_cc:
            log_debug(f"Error clearing unified cache: {e_cc}")
            if self.root.winfo_exists():
                messagebox.showerror("Error", f"Failed to clear cache: {e_cc}", parent=self.root)
            original_status_text = self.status_label.cget("text")
            self.status_label.config(text="Status: Cache clearing failed")
            self.root.after(2000, lambda: self.status_label.config(text=original_status_text) if self.status_label.winfo_exists() else None)

    def _clear_queue(self, q_to_clear):
        items_cleared_count = 0
        while not q_to_clear.empty():
            try:
                q_to_clear.get_nowait()
                items_cleared_count += 1
            except queue.Empty:
                break
            except Exception as e_cq:
                log_debug(f"Error clearing queue {type(q_to_clear).__name__}: {e_cq}")
                break
        if items_cleared_count > 0:
            log_debug(f"Cleared {items_cleared_count} items from {type(q_to_clear).__name__}.")

    def _reset_translation_scheduler_session_state(self, reason):
        """Invalidate async translation scheduler state at a session boundary."""
        from worker_threads import reset_translation_scheduler_session_state

        reset_translation_scheduler_session_state(self, reason)

    def _reset_gemini_batch_state(self):
        """Reset Gemini OCR batch management state for clean start."""
        self.batch_sequence_counter = 0
        self.last_displayed_batch_sequence = 0
        self.active_ocr_calls = set()
        self.last_processed_subtitle = None
        self.last_local_ocr_submitted_text = None
        self.last_local_ocr_submitted_norm = None
        self.last_local_ocr_submitted_scope = None
        self.clear_ocr_stability_gate("OCR batch state reset")
        self.clear_timeout_timer_start = None
        log_debug("Gemini OCR batch state reset")

    def _root_window_alive(self):
        try:
            if not self.root:
                return False
            if not hasattr(self.root, 'winfo_exists'):
                return True
            return bool(self.root.winfo_exists())
        except Exception:
            return False

    def _stop_translation_for_app_exit(self):
        """Stop worker activity for application exit without scheduling UI callbacks."""
        if not self.is_running:
            log_debug("Process was not running at close time.")
            return

        log_debug("Stopping running OCR/translation process for app exit...")
        self.is_running = False
        self.toggle_in_progress = False

        active_threads_copy = list(getattr(self, 'threads', []) or [])
        try:
            self.threads.clear()
        except Exception:
            self.threads = []

        for thread_obj in active_threads_copy:
            try:
                if thread_obj.is_alive():
                    thread_obj.join(timeout=0.5)
            except Exception as join_error:
                log_debug(f"Error joining thread during app exit: {join_error}")

        if hasattr(self, 'active_ocr_calls'):
            self.active_ocr_calls.clear()
        if hasattr(self, 'active_translation_calls'):
            self.active_translation_calls.clear()

        handler = getattr(self, 'translation_handler', None)
        if handler is not None:
            for method_name in ('request_end_ocr_session', 'request_end_translation_session'):
                method = getattr(handler, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception as session_error:
                        log_debug(f"Error ending session during app exit: {session_error}")

    def _graceful_shutdown_poll(self):
        """
        Non-blocking poll to check if all async API calls have finished.
        This allows the tkinter event loop to process callbacks that decrement pending call counters.
        """
        if getattr(self, '_app_is_closing', False) or getattr(self, '_shutdown_finalized', False):
            log_debug("Graceful shutdown poll ignored because application shutdown is already finalizing.")
            return
        if not self._root_window_alive():
            log_debug("Graceful shutdown poll ignored because root window no longer exists.")
            return

        # Calculate pending calls from all providers
        pending_ocr = 0
        translation_handler = getattr(self, 'translation_handler', None)
        if hasattr(translation_handler, 'ocr_providers'):
            for provider in translation_handler.ocr_providers.values():
                pending_ocr += provider._pending_ocr_calls
        if hasattr(self, 'active_ocr_calls'):
            pending_ocr += len(self.active_ocr_calls)

        pending_translation = 0
        if hasattr(translation_handler, 'providers'):
            for provider in translation_handler.providers.values():
                pending_translation += provider._pending_translation_calls
        if hasattr(self, 'active_translation_calls'):
            pending_translation += len(self.active_translation_calls)

        # Check if timeout is reached or all calls are done
        elapsed = time.monotonic() - self._shutdown_start_time
        if (pending_ocr == 0 and pending_translation == 0) or elapsed > 20.0:
            if elapsed > 20.0:
                log_debug(f"Warning: Shutdown timeout of 20.0s reached. Some API calls may not have completed.")
            else:
                log_debug("All pending API calls have completed.")

            log_debug(f"Graceful shutdown for thread pools completed in {elapsed:.2f}s.")
            self._finalize_shutdown() # Proceed to the final steps
            return

        # If not done, poll again shortly
        log_debug(f"Waiting for pending API calls to complete... OCR: {pending_ocr}, Translation: {pending_translation}")
        if self._root_window_alive() and not getattr(self, '_app_is_closing', False):
            self.root.after(100, self._graceful_shutdown_poll)

    def _finalize_shutdown(self):
        """Contains the final steps of the shutdown process after graceful polling."""
        if getattr(self, '_shutdown_finalized', False):
            log_debug("Finalize shutdown ignored because it already ran.")
            return
        self._shutdown_finalized = True

        # End the sessions HERE, after all pending calls are confirmed to be finished.
        if hasattr(self, 'translation_handler'):
            self.translation_handler.request_end_ocr_session()
            self.translation_handler.request_end_translation_session()

        self._clear_queue(self.ocr_queue)
        self._clear_queue(self.translation_queue)
        self._reset_translation_scheduler_session_state("translation stopped")
        self.clear_ocr_stability_gate("translation stopped")

        if self.translation_text and self.translation_text.winfo_exists():
            try:
                self.translation_text.config(state=tk.NORMAL)
                self.translation_text.delete(1.0, tk.END)
                self.translation_text.config(state=tk.DISABLED)
            except tk.TclError as e_ctt:
                log_debug(f"Error clearing translation text on stop: {e_ctt}")

        if self.source_overlay and self.source_overlay.winfo_exists() and self.source_overlay.winfo_viewable():
            try: self.source_overlay.hide()
            except tk.TclError: log_debug("Error hiding source overlay on stop (likely closed).")

        if self.target_overlay and self.target_overlay.winfo_exists() and self.target_overlay.winfo_viewable():
            try: self.target_overlay.hide()
            except tk.TclError: log_debug("Error hiding target overlay on stop (likely closed).")

        self.start_stop_btn.config(state=tk.NORMAL, style="Primary.TButton")
        status_text_stopped = "Status: " + self.ui_lang.get_label("status_stopped", "Stopped (Press ~ to Start)")
        self.status_label.config(text=status_text_stopped)
        log_debug("Translation process stopped.")

        self.toggle_in_progress = False # Release the lock here

    def toggle_translation(self):
        # Add re-entrancy lock
        if self.toggle_in_progress:
            log_debug("Toggle translation already in progress, ignoring call.")
            return

        self.toggle_in_progress = True

        if self.is_running:
            log_debug("Stopping translation process requested by user.")
            self.is_running = False
            self._shutdown_finalized = False

            # DO NOT request session ends here. This will be done in _finalize_shutdown.
            # Context clearing is now handled automatically after session end logging in llm_provider_base.py

            # Cancel amber pulse immediately when user requests stop
            _dot_canvases = getattr(self, "_pipeline_dot_canvases", None)
            _palette = getattr(self, "_pipeline_pulse_palette", None)
            cancel_pipeline_pulse(self, _dot_canvases, _palette)
            self.start_stop_btn.config(text="Start", state=tk.DISABLED, style="Primary.TButton")
            self.status_label.config(text="Status: Stopping...")
            self.root.update_idletasks()

            active_threads_copy = self.threads[:]
            self.threads.clear()

            thread_stop_start_time = time.monotonic()
            log_debug(f"Waiting for main worker threads to join: {[t.name for t in active_threads_copy if t.is_alive()]}")

            for thread_obj in active_threads_copy:
                if thread_obj.is_alive():
                    try:
                        thread_obj.join(timeout=1.0) # Short timeout for main threads
                    except Exception as join_err_tt:
                        log_debug(f"Error joining thread {thread_obj.name}: {join_err_tt}")

            log_debug(f"Main worker threads joined in {time.monotonic() - thread_stop_start_time:.2f}s.")

            # Use non-blocking poll for graceful shutdown
            log_debug("Starting graceful shutdown poll for API call thread pools...")
            self._shutdown_start_time = time.monotonic()
            self.root.after(0, self._graceful_shutdown_poll)
            # The rest of the shutdown logic is now in _finalize_shutdown()
            # The lock will be released in _finalize_shutdown()

        else:
            try:
                log_debug("Starting translation process requested by user...")
                self.start_stop_btn.config(state=tk.DISABLED)
                self.status_label.config(text="Status: Initializing...")
                self.root.update_idletasks()

                valid_start_flag = True

                if not self.source_overlay or not self._widget_exists_safely(self.source_overlay):
                    messagebox.showerror("Start Error", "Source area overlay missing. Select source area.", parent=self.root)
                    valid_start_flag = False
                if valid_start_flag and (not self.target_overlay or not self._widget_exists_safely(self.target_overlay)):
                    messagebox.showerror("Start Error", "Target area overlay missing. Select target area.", parent=self.root)
                    valid_start_flag = False
                if valid_start_flag and (not self.translation_text or not self._widget_exists_safely(self.translation_text)):
                    messagebox.showerror("Start Error", "Target text display widget missing. Reselect target area.", parent=self.root)
                    valid_start_flag = False

                if valid_start_flag and not self.custom_ai_profiles.get_active_profile("translation"):
                    messagebox.showerror(
                        self.ui_lang.get_label("start_error_title", "Start Error"),
                        self.ui_lang.get_label("start_error_no_translation_profile", "Add and select an AI model profile before starting."),
                        parent=self.root
                    )
                    valid_start_flag = False

                if valid_start_flag and self.get_ocr_model_setting() == 'custom_ai' and not self.custom_ai_profiles.get_active_profile("ocr"):
                    messagebox.showerror(
                        self.ui_lang.get_label("start_error_title", "Start Error"),
                        self.ui_lang.get_label("start_error_no_ocr_profile", "Add and select an AI model profile for OCR before starting, or choose PaddleOCR."),
                        parent=self.root
                    )
                    valid_start_flag = False

                if valid_start_flag:
                     try:
                         self.source_area = self.source_overlay.get_geometry()
                         self.target_area = self.target_overlay.get_geometry()
                         if not self._validate_area_coords(self.source_area, "source"): valid_start_flag = False
                         if valid_start_flag and not self._validate_area_coords(self.target_area, "target"): valid_start_flag = False
                     except (tk.TclError, AttributeError) as e_gog:
                         messagebox.showerror("Start Error", f"Could not get overlay geometry: {e_gog}", parent=self.root)
                         valid_start_flag = False

                if not valid_start_flag:
                    self.start_stop_btn.config(state=tk.NORMAL)
                    status_text_failed = "Status: Start Failed"
                    if self.KEYBOARD_AVAILABLE: status_text_failed += " (Press ~ to Retry)"
                    self.status_label.config(text=status_text_failed)
                    log_debug("Start aborted due to failed pre-start validation checks.")
                    return

                log_debug("Pre-start checks passed. Preparing to start threads...")
                self.text_stability_counter = 0
                self.previous_text = ""
                self.last_image_hash = None
                self.last_screenshot = None
                self.last_processed_image = None

                self._reset_gemini_batch_state()

                try:
                    if self.target_overlay and self.target_overlay.winfo_exists() and not self.target_overlay.winfo_viewable():
                        self.target_overlay.show()
                except tk.TclError:
                    log_debug("Warning: Error ensuring target overlay visibility at start (likely closed).")

                self._clear_queue(self.ocr_queue)
                self._clear_queue(self.translation_queue)
                self._reset_translation_scheduler_session_state(
                    "translation starting"
                )
                self.last_local_ocr_submitted_text = None
                self.last_local_ocr_submitted_norm = None
                self.last_local_ocr_submitted_scope = None
                self.clear_ocr_stability_gate("translation starting")

                self.cache_manager.load_file_caches()

                self._app_is_closing = False
                self._shutdown_finalized = False
                self.is_running = True

                if hasattr(self, 'translation_handler'):
                    if self.is_api_based_ocr_model():
                        self.translation_handler.start_ocr_session()
                    self.translation_handler.start_translation_session()

                self.start_stop_btn.config(text="Stop", state=tk.NORMAL, style="StartRunning.TButton")
                status_text_running = "Status: " + self.ui_lang.get_label("status_running", "Running (Press ~ to Stop)")
                self.status_label.config(text=status_text_running)
                self.root.update_idletasks()
                # Start amber pipeline pulse — tied to self.is_running being True
                _dot_canvases = getattr(self, "_pipeline_dot_canvases", None)
                _palette = getattr(self, "_pipeline_pulse_palette", None)
                if _dot_canvases and _palette:
                    start_pipeline_pulse(self, _dot_canvases, _palette)

                from worker_threads import run_capture_thread, run_ocr_thread, run_translation_thread

                capture_thread_instance = threading.Thread(target=run_capture_thread, args=(self,), name="CaptureThread", daemon=True)
                ocr_thread_instance = threading.Thread(target=run_ocr_thread, args=(self,), name="OCRThread", daemon=True)
                translation_thread_instance = threading.Thread(target=run_translation_thread, args=(self,), name="TranslationThread", daemon=True)

                self.threads = [capture_thread_instance, ocr_thread_instance, translation_thread_instance]
                for t_obj in self.threads:
                    t_obj.start()
                log_debug(f"Threads started: {[t.name for t in self.threads]}")

                # Release lock after successful start
                self.toggle_in_progress = False

            finally:
                # Release lock if start failed before threads were launched
                if not self.is_running:
                    self.toggle_in_progress = False

    def _validate_area_coords(self, area_coordinates, area_type_str):
        min_dimension = 10
        if not area_coordinates or len(area_coordinates) != 4:
            messagebox.showerror("Area Validation Error", f"Invalid {area_type_str} area data: {area_coordinates}.", parent=self.root)
            return False
        try:
            x1_val, y1_val, x2_val, y2_val = map(int, area_coordinates)
            width_val = x2_val - x1_val
            height_val = y2_val - y1_val
            if width_val < min_dimension or height_val < min_dimension:
                messagebox.showerror("Area Validation Error",
                                     f"{area_type_str.capitalize()} area too small ({width_val}x{height_val}). Min {min_dimension}x{min_dimension}.",
                                     parent=self.root)
                return False
            return True
        except (ValueError, TypeError) as e_vac:
            messagebox.showerror("Area Validation Error", f"Invalid coordinates in {area_type_str} area: {area_coordinates}. Error: {e_vac}", parent=self.root)
            return False

    def stop_translation_from_thread(self):
        if self.is_running:
            log_debug("Requesting stop translation from worker thread.")
            if self.root.winfo_exists():
                self.root.after(0, self.toggle_translation)

    def update_translation_model_names(self):
        """Update translation model names with localized strings from CSV files."""
        self.translation_model_names = {
            'custom_ai': self.ui_lang.get_label("translation_model_custom_ai", "Custom AI Translation")
        }
        self.translation_model_values = {v: k for k, v in self.translation_model_names.items()}
        log_debug(f"Updated translation model names: {self.translation_model_names}")

    def get_custom_ai_latency_mode(self):
        """Return the selected Custom AI response mode, normalized to a supported value."""
        var = getattr(self, 'custom_ai_latency_mode_var', None)
        try:
            return normalize_custom_ai_latency_mode(var.get() if var is not None else CUSTOM_AI_LATENCY_MODE_SAFE)
        except Exception:
            return CUSTOM_AI_LATENCY_MODE_SAFE

    def get_custom_ai_ocr_image_format(self):
        """Return the selected Custom AI OCR image file format."""
        var = getattr(self, 'custom_ai_ocr_image_format_var', None)
        try:
            return normalize_api_ocr_image_format(var.get() if var is not None else API_OCR_IMAGE_FORMAT_DEFAULT)
        except Exception:
            return API_OCR_IMAGE_FORMAT_DEFAULT

    def get_custom_ai_ocr_image_mode(self):
        """Return the selected Custom AI OCR image encoding mode."""
        var = getattr(self, 'custom_ai_ocr_image_mode_var', None)
        try:
            return normalize_api_ocr_image_mode(var.get() if var is not None else API_OCR_IMAGE_MODE_DEFAULT)
        except Exception:
            return API_OCR_IMAGE_MODE_DEFAULT

    def get_custom_ai_ocr_image_quality(self):
        """Return the selected Custom AI OCR image quality."""
        var = getattr(self, 'custom_ai_ocr_image_quality_var', None)
        try:
            return normalize_api_ocr_image_quality(var.get() if var is not None else API_OCR_IMAGE_QUALITY_DEFAULT)
        except Exception:
            return API_OCR_IMAGE_QUALITY_DEFAULT

    def get_custom_ai_ocr_image_detail(self):
        """Return the selected Custom AI OCR vision detail mode."""
        var = getattr(self, 'custom_ai_ocr_image_detail_var', None)
        try:
            return normalize_api_ocr_image_detail(var.get() if var is not None else API_OCR_IMAGE_DETAIL_DEFAULT)
        except Exception:
            return API_OCR_IMAGE_DETAIL_DEFAULT

    def get_current_gemini_model_for_translation(self):
        """Get the API name of currently selected Gemini translation model."""
        display_name = self.gemini_translation_model_var.get()
        return self.gemini_models_manager.get_api_name_by_display_name(display_name)

    def get_current_gemini_model_for_ocr(self):
        """Get the API name of currently selected Gemini OCR model."""
        display_name = self.gemini_ocr_model_var.get()
        return self.gemini_models_manager.get_api_name_by_display_name(display_name)

    def get_current_openai_model_for_translation(self):
        """Get the API name of currently selected OpenAI translation model."""
        display_name = self.openai_translation_model_var.get()
        return self.openai_models_manager.get_api_name_by_display_name(display_name)

    def is_openai_model(self, model_name):
        """Check if the given model name is an OpenAI model."""
        return False

    def is_gemini_model(self, model_name):
        """Check if the given model name is a Gemini model."""
        return False

    def get_current_openai_model_for_ocr(self):
        """Get the API name of currently selected OpenAI OCR model."""
        # Read from the new, specific variable
        display_name = self.openai_ocr_model_var.get()
        api_name = self.openai_models_manager.get_api_name_by_display_name(display_name)
        if api_name:
            return api_name
        return 'gpt-4o'  # Default fallback if lookup fails

    def is_api_based_ocr_model(self, model_name=None):
        """Check if the given (or current) OCR model is API-based and needs session management."""
        if model_name is None:
            model_name = self.get_ocr_model_setting()

        return model_name == 'custom_ai'

    def update_ui_language(self):
        """Rebuild visible UI tabs after the UI language changes."""
        try:
            self.start_ui_update()
            # Stop any pulse tied to widgets that are about to be rebuilt.
            _dot_canvases = getattr(self, "_pipeline_dot_canvases", None)
            _palette = getattr(self, "_pipeline_pulse_palette", None)
            cancel_pipeline_pulse(self, _dot_canvases, _palette)
            self.update_translation_model_names()
            selected_index = 0
            try:
                selected_index = self.tab_control.index(self.tab_control.select())
            except Exception:
                selected_index = 0

            for i in range(self.tab_control.index('end') - 1, -1, -1):
                self.tab_control.forget(i)

            self.tab_main = None
            self.tab_settings = None
            self.tab_custom_prompt = None
            self.tab_debug = None

            create_main_tab(self)
            create_settings_tab(self)
            create_custom_prompt_tab(self)
            create_debug_tab(self)

            self.ui_interaction_handler.update_translation_model_ui()
            self.ui_interaction_handler.update_ocr_model_ui()
            self.root.after_idle(lambda: self.ui_interaction_handler.update_ocr_model_ui())

            if self.translation_model_var.get() == 'custom_ai':
                active_profile = self.custom_ai_profiles.get_active_profile("translation")
                display_name = active_profile["name"] if active_profile else self.ui_lang.get_label("custom_ai_no_profiles", "Add an AI model profile")
                self.translation_model_display_var.set(display_name)

            self.ui_interaction_handler.update_all_dropdowns_for_language_change()

            if hasattr(self, 'update_deepl_model_type_for_language'):
                self.update_deepl_model_type_for_language()
            if hasattr(self, 'update_deepl_context_window_for_language'):
                self.update_deepl_context_window_for_language()
            if hasattr(self, 'update_gemini_context_window_for_language'):
                self.update_gemini_context_window_for_language()
            if hasattr(self, 'update_gemini_labels_for_language'):
                self.update_gemini_labels_for_language()

            def on_tab_changed(event):
                current_index = self.tab_control.index(self.tab_control.select())
                if current_index == 0 and hasattr(self, 'main_tab_start_button') and self.main_tab_start_button.winfo_exists():
                    self.main_tab_start_button.focus_set()
                elif current_index == 1 and hasattr(self, 'settings_tab_save_button') and self.settings_tab_save_button.winfo_exists():
                    self.settings_tab_save_button.focus_set()

            self.tab_control.bind("<<NotebookTabChanged>>", on_tab_changed)

            tab_count = self.tab_control.index('end')
            if tab_count > 0:
                self.tab_control.select(min(selected_index, tab_count - 1))

            if self.is_running:
                status_text = "Status: " + self.ui_lang.get_label("status_running", "Running (Press ~ to Stop)")
                self.start_stop_btn.config(
                    text=self.ui_lang.get_label("stop_btn"),
                    style="StartRunning.TButton",
                )
                # Restart pulse animation after tab rebuild
                _dot_canvases = getattr(self, "_pipeline_dot_canvases", None)
                _palette = getattr(self, "_pipeline_pulse_palette", None)
                if _dot_canvases and _palette:
                    start_pipeline_pulse(self, _dot_canvases, _palette)
            else:
                status_text = "Status: " + self.ui_lang.get_label("status_stopped", "Stopped (Press ~ to Start)")
                if not self.KEYBOARD_AVAILABLE:
                    status_text = self.ui_lang.get_label("status_ready", "Status: Ready")
            self.status_label.config(text=status_text)

            if hasattr(self, 'debug_log_toggle_btn') and self.debug_log_toggle_btn.winfo_exists():
                button_key = "toggle_debug_log_disable_btn" if self.debug_logging_enabled_var.get() else "toggle_debug_log_enable_btn"
                self.debug_log_toggle_btn.config(text=self.ui_lang.get_label(button_key))

            log_debug(f"UI language completely rebuilt for: {self.ui_lang.current_lang}")
        except Exception as e:
            log_debug(f"Error updating UI language: {e}")
        finally:
            self.end_ui_update()

    def setup_network_cleanup(self):
        """Setup periodic network connection cleanup to prevent stack corruption."""
        def cleanup_network_connections():
            try:
                # Force client recreation to clear connection pools
                if hasattr(self, 'translation_handler') and hasattr(self.translation_handler, 'gemini_client'):
                    if self.translation_handler.gemini_client is not None:
                        old_client = self.translation_handler.gemini_client

                        # Force client refresh
                        self.translation_handler._force_client_refresh()

                        # Try to close old client connections if possible
                        try:
                            if hasattr(old_client, 'close'):
                                old_client.close()
                            elif hasattr(old_client, '_transport') and hasattr(old_client._transport, 'close'):
                                old_client._transport.close()
                        except Exception as close_error:
                            log_debug(f"Error closing old client: {close_error}")

                        log_debug("Performed periodic network connection cleanup")

                # Also flush DNS cache
                self.flush_dns_cache_if_needed()

            except Exception as e:
                log_debug(f"Error during periodic network cleanup: {e}")

            # Schedule next cleanup in 20 minutes
            if self.is_running:  # Only schedule if application is still running
                self.root.after(1200000, cleanup_network_connections)  # 20 minutes = 1200000ms

        # Start cleanup cycle after 20 minutes of operation
        self.root.after(1200000, cleanup_network_connections)
        log_debug("Scheduled periodic network cleanup every 20 minutes")

    def flush_dns_cache_if_needed(self):
        """Flush system DNS cache if network performance degrades."""
        if not hasattr(self, 'last_dns_flush'):
            self.last_dns_flush = time.time()
            return

        current_time = time.time()
        # Flush DNS every hour during active use
        if current_time - self.last_dns_flush > 3600:  # 1 hour
            try:
                import subprocess
                result = subprocess.run(['ipconfig', '/flushdns'],
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    self.last_dns_flush = current_time
                    log_debug("Successfully flushed DNS cache for network maintenance")
                else:
                    log_debug(f"DNS flush command failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                log_debug("DNS flush command timed out")
            except Exception as e:
                log_debug(f"Could not flush DNS cache: {e}")

    def on_closing(self):
        log_debug("Main window close requested. Initiating shutdown...")
        self._app_is_closing = True
        # Cancel any running amber pulse immediately
        _dot_canvases = getattr(self, "_pipeline_dot_canvases", None)
        _palette = getattr(self, "_pipeline_pulse_palette", None)
        cancel_pipeline_pulse(self, _dot_canvases, _palette)
        if getattr(self, "runtime_metrics_refresh_after_id", None):
            try:
                self.root.after_cancel(self.runtime_metrics_refresh_after_id)
            except Exception:
                pass
            self.runtime_metrics_refresh_after_id = None

        # Close OCR Preview window if open
        if self.ocr_preview_window is not None:
            try:
                log_debug("Closing OCR Preview window...")
                self.close_ocr_preview()
            except Exception as e:
                log_debug(f"Error closing OCR Preview window: {e}")

        self._stop_translation_for_app_exit()

        # # Force end any remaining sessions when application closes
        # if hasattr(self, 'translation_handler'):
        #     try:
        #         self.translation_handler.force_end_sessions_on_app_close()
        #     except Exception as e:
        #         log_debug(f"Error ending sessions on app close: {e}")

        if hasattr(self, 'marian_translator') and self.marian_translator and hasattr(self.marian_translator, 'thread_pool'):
            try:
                log_debug("Shutting down MarianMT thread pool...")
                self.marian_translator.thread_pool.shutdown(wait=True, cancel_futures=True)
                log_debug("MarianMT thread pool shutdown complete.")
            except Exception as e_mtps:
                log_debug(f"Error shutting down MarianMT thread pool: {e_mtps}")

        # Shutdown OCR and translation thread pools
        if hasattr(self, 'ocr_thread_pool'):
            try:
                log_debug("Shutting down OCR thread pool...")
                self.ocr_thread_pool.shutdown(wait=False, cancel_futures=True)
                log_debug("OCR thread pool shutdown complete.")
            except Exception as e_otp:
                log_debug(f"Error shutting down OCR thread pool: {e_otp}")

        if hasattr(self, 'translation_thread_pool'):
            try:
                log_debug("Shutting down translation thread pool...")
                self.translation_thread_pool.shutdown(wait=False, cancel_futures=True)
                log_debug("Translation thread pool shutdown complete.")
            except Exception as e_ttp:
                log_debug(f"Error shutting down translation thread pool: {e_ttp}")
        if hasattr(self, 'translation_handler'):
            try:
                self.translation_handler.close()
            except Exception as e_close:
                log_debug(f"Error closing translation handler: {e_close}")
        try:
            log_debug("Saving final settings before closing...")
            if self._fully_initialized:
                # Save OCR Preview geometry if window is open
                if self.ocr_preview_window is not None:
                    self.save_preview_geometry()
                self.save_settings()
            else:
                # Save OCR Preview geometry even if not fully initialized
                if self.ocr_preview_window is not None:
                    self.save_preview_geometry()
                self.configuration_handler.save_current_window_geometry()
                save_app_config(self.config)
        except Exception as e_ssc:
            log_debug(f"Error saving settings during closing: {e_ssc}")

        if self.KEYBOARD_AVAILABLE:
            try:
                import keyboard
                keyboard.unhook_all()
                log_debug("Unhooked all keyboard shortcuts.")
            except Exception as e_uhk:
                log_debug(f"Error unhooking keyboard shortcuts: {e_uhk}")

        log_debug("Destroying overlay windows if they exist...")
        for overlay_attr_name in ['source_overlay', 'target_overlay']:
            overlay_widget = getattr(self, overlay_attr_name, None)
            if overlay_widget:
                try:
                    # Preserve target overlay position before destroying during shutdown
                    if overlay_attr_name == 'target_overlay':
                        from overlay_manager import _preserve_overlay_position
                        _preserve_overlay_position(self)
                        log_debug("Preserved target overlay position during app shutdown")

                    # Handle tkinter overlays
                    if hasattr(overlay_widget, 'winfo_exists') and overlay_widget.winfo_exists():
                        overlay_widget.destroy()
                    # Handle PySide overlays
                    elif hasattr(overlay_widget, 'close'):
                        overlay_widget.close()

                except Exception as e_dow:
                    log_debug(f"Error destroying {overlay_attr_name}: {e_dow}")
            setattr(self, overlay_attr_name, None)
        self.translation_text = None

        log_debug("Destroying root window...")
        try:
            if self.root and hasattr(self.root, 'winfo_exists') and self.root.winfo_exists():
                self.root.destroy()
            log_debug("Root window destroyed.")
        except Exception as e_drw:
             log_debug(f"Error destroying root window: {e_drw}")
        log_debug("Application shutdown sequence complete.")

    def load_custom_prompt(self):
        """Loads the custom prompt text from file."""
        try:
            if os.path.exists(self.custom_prompt_file):
                with open(self.custom_prompt_file, 'r', encoding='utf-8-sig') as f:
                    self.custom_prompt_text = f.read()
                if not self.custom_prompt_text.strip():
                    self.custom_prompt_text = DEFAULT_CUSTOM_PROMPT
                    with open(self.custom_prompt_file, 'w', encoding='utf-8-sig') as f:
                        f.write(self.custom_prompt_text)
                    log_debug("Initialized empty custom prompt with default text")
                else:
                    log_debug(f"Loaded custom prompt ({len(self.custom_prompt_text)} characters)")
            else:
                self.custom_prompt_text = DEFAULT_CUSTOM_PROMPT
                with open(self.custom_prompt_file, 'w', encoding='utf-8-sig') as f:
                    f.write(self.custom_prompt_text)
                log_debug("Created custom prompt file with default text")
        except Exception as e:
            log_debug(f"Error loading custom prompt: {e}")
            self.custom_prompt_text = DEFAULT_CUSTOM_PROMPT

    def save_custom_prompt(self, text):
        """Saves the custom prompt text to file."""
        try:
            self.custom_prompt_text = text
            with open(self.custom_prompt_file, 'w', encoding='utf-8-sig') as f:
                f.write(text)
            log_debug(f"Saved custom prompt ({len(text)} characters)")
            return True
        except Exception as e:
            log_debug(f"Error saving custom prompt: {e}")
            return False
