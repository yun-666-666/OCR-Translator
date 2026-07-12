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
from modern_ui import apply_white_clean_theme, style_tk_canvas, style_tk_text_widget
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

from app_capture_ocr import AppCaptureOcrMixin
from app_configuration import AppConfigurationMixin
from app_lifecycle import AppLifecycleMixin


class GameChangingTranslator(AppCaptureOcrMixin, AppConfigurationMixin, AppLifecycleMixin):
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
