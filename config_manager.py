# config_manager.py
import configparser
import os
import sys
from ai_optimization import (
    AI_OPTIMIZATION_AUTO,
    LEGACY_AI_SETTING_KEYS,
    migrate_legacy_ai_optimization_settings,
    migrate_legacy_ocr_defaults,
    normalize_ai_optimization_mode,
)
from credential_store import create_default_credential_store
from logger import log_debug
from resource_handler import get_resource_path

PROVIDER_CREDENTIAL_SERVICE = "OCR-Translator-Providers"
PROVIDER_API_KEY_SETTINGS = (
    'google_translate_api_key',
    'deepl_api_key',
    'gemini_api_key',
    'openai_api_key',
)

TRANSLATION_LINE_LAYOUT_COMPACT = 'compact'
TRANSLATION_LINE_LAYOUT_PRESERVE_SOURCE_LINES = 'preserve_source_lines'


def normalize_translation_line_layout(value):
    normalized = str(value or '').strip().lower()
    if normalized == TRANSLATION_LINE_LAYOUT_PRESERVE_SOURCE_LINES:
        return TRANSLATION_LINE_LAYOUT_PRESERVE_SOURCE_LINES
    return TRANSLATION_LINE_LAYOUT_COMPACT

DEFAULT_CONFIG_SETTINGS = {
    'scan_interval': '300',
    'ocr_frame_cache_size': '64',
    'enable_instant_cache_display': 'True',
    'stability_threshold': '0',
    'clear_translation_timeout': '3',
    'ocr_debugging': 'True',
    'source_area_x1': '334',
    'source_area_y1': '856',
    'source_area_x2': '1579',
    'source_area_y2': '1041',
    'source_area_visible': '0', # Changed to string 'False' or 'True' later
    'target_area_x1': '717',
    'target_area_y1': '17',
    'target_area_x2': '1612',
    'target_area_y2': '178',
    'source_area_colour': '#FFFF99',
    'target_area_colour': '#162c43',
    'target_text_colour': '#FFD54F',
    'target_text_outline_colour': '#000000',
    'target_text_outline_width': '2',
    'target_opacity': '0.4',
    'target_font_size': '18',
    'target_font_type': 'Arial',
    'target_font_bold': 'False',
    'main_window_geometry': '619x728+6+23', # This seems to be a complete geometry string
    'main_window_width': '619',   # Keep these for individual component loading
    'main_window_height': '728',
    'main_window_x': '6',
    'main_window_y': '23',
    'translation_model': 'custom_ai', # Default model
    'debug_logging_enabled': 'False',
    'custom_ai_log_content_enabled': 'False',
    'gui_language':'English',
    # OCR Model Selection
    'ocr_model': 'paddleocr',
    'paddleocr_source_dir': 'PaddleOCR-3.7.0',
    'paddleocr_lang': 'en',
    'paddleocr_ocr_version': 'PP-OCRv6',
    'paddleocr_model_size': 'tiny',
    'paddleocr_device': 'cpu',
    'paddleocr_min_score': '0.45',
    'paddleocr_upscale': '1.0',
    'paddleocr_text_det_limit_side_len': '960',
    'paddleocr_text_det_limit_type': 'max',
    'paddleocr_use_textline_orientation': 'False',
    'custom_ai_profiles_file': 'custom_ai_profiles.json',
    'custom_source_lang': 'auto',
    'custom_target_lang': 'en',
    'ai_optimization_mode': AI_OPTIMIZATION_AUTO,
    'custom_ai_submit_interval_ms': '300',
    # OCR Preview window geometry
    'ocr_preview_geometry': '600x800+100+100',
    'ocr_preview_width': '600',
    'ocr_preview_height': '800',
    'ocr_preview_x': '100',
    'ocr_preview_y': '100',
    'keep_linebreaks': 'False',
    'translation_line_layout': TRANSLATION_LINE_LAYOUT_COMPACT,
    'translation_horizontal_centered': 'False',
    'custom_context_window': '5'
}


def _provider_api_key_ref(setting_key):
    return f"provider:{setting_key}:api_key"


def _provider_api_key_ref_setting(setting_key):
    return f"{setting_key}_ref"


def _settings_from_config(config_or_settings):
    if hasattr(config_or_settings, "sections"):
        if 'Settings' not in config_or_settings:
            config_or_settings['Settings'] = {}
        return config_or_settings['Settings']
    return config_or_settings


def _log_provider_credential_issue(action, setting_key, error, fallback=False):
    fallback_text = "; plaintext fallback retained" if fallback else ""
    log_debug(
        "Provider credential "
        f"{action} failed for {setting_key}: {type(error).__name__}{fallback_text}"
    )


def migrate_provider_api_keys_to_credentials(config_or_settings, credential_store=None):
    settings = _settings_from_config(config_or_settings)
    pending_keys = [
        setting_key
        for setting_key in PROVIDER_API_KEY_SETTINGS
        if str(settings.get(setting_key, "") or "")
    ]
    if not pending_keys:
        return False
    store = credential_store or create_default_credential_store(PROVIDER_CREDENTIAL_SERVICE)
    changed = False
    for setting_key in pending_keys:
        ref_setting = _provider_api_key_ref_setting(setting_key)
        plaintext_key = str(settings.get(setting_key, "") or "")
        credential_ref = str(settings.get(ref_setting, "") or "").strip() or _provider_api_key_ref(setting_key)
        try:
            store.set_secret(credential_ref, plaintext_key)
            settings[ref_setting] = credential_ref
            settings[setting_key] = ""
            changed = True
        except Exception as e:
            _log_provider_credential_issue("write", setting_key, e, fallback=True)
    return changed


def get_provider_api_key(config_or_settings, setting_key, credential_store=None):
    settings = _settings_from_config(config_or_settings)
    plaintext_key = str(settings.get(setting_key, "") or "")
    if plaintext_key:
        return plaintext_key
    credential_ref = str(settings.get(_provider_api_key_ref_setting(setting_key), "") or "").strip()
    if not credential_ref:
        return ""
    store = credential_store or create_default_credential_store(PROVIDER_CREDENTIAL_SERVICE)
    try:
        resolved_key = store.get_secret(credential_ref)
    except Exception as e:
        _log_provider_credential_issue("read", setting_key, e)
        return ""
    return str(resolved_key or "")


def load_app_config():
    """Loads configuration from INI file or creates default values."""
    config_path = 'ocr_translator_config.ini'
    config = configparser.ConfigParser()

    dynamic_defaults = DEFAULT_CONFIG_SETTINGS.copy()

    if os.path.exists(config_path):
        try:
            config.read(config_path, encoding='utf-8')
            if 'Settings' not in config:
                log_debug("Config file loaded but missing [Settings] section. Adding.")
                config['Settings'] = {}
        except Exception as e:
            log_debug(f"Error reading config file {config_path}: {e}. Using defaults.")
            config['Settings'] = {}
    else:
         log_debug(f"Config file {config_path} not found. Creating with defaults.")
         config['Settings'] = {}

    settings_changed = False
    config_settings = config['Settings']

    legacy_ai_keys_present = (
        "ai_optimization_mode" not in config_settings
        or "capture_backend" in config_settings
        or any(key in config_settings for key in LEGACY_AI_SETTING_KEYS)
    )
    migrated_ai_mode = migrate_legacy_ai_optimization_settings(config_settings)
    if legacy_ai_keys_present:
        settings_changed = True
        log_debug(
            "Config: Migrated legacy AI response/image settings to "
            f"ai_optimization_mode='{migrated_ai_mode}'"
        )

    if migrate_legacy_ocr_defaults(config_settings):
        settings_changed = True
        log_debug(
            "Config: Migrated former OCR defaults to "
            "stability_threshold='0' and paddleocr_min_score='0.45'"
        )

    if 'translation_line_layout' not in config_settings:
        legacy_keep_linebreaks = config_settings.getboolean(
            'keep_linebreaks', fallback=False
        )
        config_settings['translation_line_layout'] = (
            TRANSLATION_LINE_LAYOUT_PRESERVE_SOURCE_LINES
            if legacy_keep_linebreaks
            else TRANSLATION_LINE_LAYOUT_COMPACT
        )
        settings_changed = True
        log_debug("Config: Migrated legacy keep_linebreaks to translation_line_layout")

    # Obsolete keys check (add 'source_lang', 'target_lang', 'ocr_lang' if you are sure to remove them)
    obsolete_keys = ['api_key', 'gpu_enabled', 'spell_check_enabled', 'word_segmentation_enabled',
                    'spell_check_language', 'subtitle_mode', 'parallel_processing', 'target_text_bg_color',
                    'nllb_beam_size', 'source_lang', 'target_lang', 'ocr_lang', 'gemini_fuzzy_detection',
                    'input_token_cost', 'output_token_cost', 'num_beams',
                    'marian_models_file', 'marian_model', 'marian_source_lang', 'marian_target_lang',
                    'google_file_cache', 'deepl_file_cache', 'gemini_file_cache', 'openai_file_cache',
                    'google_source_lang', 'google_target_lang', 'deepl_source_lang', 'deepl_target_lang',
                    'deepl_model_type', 'deepl_context_window',
                    'gemini_source_lang', 'gemini_target_lang', 'openai_source_lang', 'openai_target_lang',
                    'gemini_model_name', 'gemini_model_temp', 'gemini_translation_model', 'gemini_ocr_model',
                    'openai_translation_model', 'openai_ocr_model',
                    'gemini_context_window', 'openai_context_window',
                    'gemini_api_log_enabled', 'openai_api_log_enabled']  # legacy provider settings stripped
    obsolete_keys.extend([
        'tes' + 'seract_path',
        'image' + '_preprocessing_mode',
        'confidence' + '_threshold',
        'remove' + '_trailing_garbage',
        'adaptive' + '_block_size',
        'adaptive' + '_c',
    ])
    for key in obsolete_keys:
        if key in config_settings:
            del config_settings[key]
            settings_changed = True
            log_debug(f"Config: Removed obsolete '{key}' setting.")


    # Force live product path: Custom AI translation + PaddleOCR/Custom AI OCR.
    current_translation_model = str(config_settings.get('translation_model', 'custom_ai') or 'custom_ai')
    if current_translation_model != 'custom_ai':
        config_settings['translation_model'] = 'custom_ai'
        settings_changed = True
        log_debug(f"Config: Migrated translation_model '{current_translation_model}' to custom_ai")
    current_ocr_model = str(config_settings.get('ocr_model', 'paddleocr') or 'paddleocr')
    if current_ocr_model not in {'paddleocr', 'custom_ai'}:
        config_settings['ocr_model'] = 'paddleocr'
        settings_changed = True
        log_debug(f"Config: Migrated ocr_model '{current_ocr_model}' to paddleocr")

    for key, value in dynamic_defaults.items():
        if key not in config_settings:
            config_settings[key] = value
            settings_changed = True
            log_debug(f"Config: Added missing key '{key}' with default value '{value}'.")

    current_line_layout = config_settings.get(
        'translation_line_layout', TRANSLATION_LINE_LAYOUT_COMPACT
    )
    normalized_line_layout = normalize_translation_line_layout(current_line_layout)
    if normalized_line_layout != current_line_layout:
        config_settings['translation_line_layout'] = normalized_line_layout
        settings_changed = True
        log_debug(
            "Config: Invalid translation line layout "
            f"'{current_line_layout}' changed to '{normalized_line_layout}'"
        )

    legacy_linebreak_value = str(
        normalized_line_layout == TRANSLATION_LINE_LAYOUT_PRESERVE_SOURCE_LINES
    )
    if config_settings.get('keep_linebreaks') != legacy_linebreak_value:
        config_settings['keep_linebreaks'] = legacy_linebreak_value
        settings_changed = True
        log_debug("Config: Synchronized legacy keep_linebreaks with translation layout")

    current_ocr_model = config_settings.get('ocr_model', 'paddleocr')
    if current_ocr_model not in ['paddleocr', 'custom_ai']:
        config_settings['ocr_model'] = 'paddleocr'
        settings_changed = True
        log_debug(f"Config: Invalid OCR model '{current_ocr_model}' changed to 'paddleocr'")

    current_ai_optimization_mode = config_settings.get(
        'ai_optimization_mode', AI_OPTIMIZATION_AUTO
    )
    normalized_ai_optimization_mode = normalize_ai_optimization_mode(
        current_ai_optimization_mode
    )
    if normalized_ai_optimization_mode != current_ai_optimization_mode:
        config_settings['ai_optimization_mode'] = normalized_ai_optimization_mode
        settings_changed = True
        log_debug(
            "Config: Invalid AI optimization mode "
            f"'{current_ai_optimization_mode}' changed to "
            f"'{normalized_ai_optimization_mode}'"
        )

    if migrate_provider_api_keys_to_credentials(config_settings):
        settings_changed = True

    if settings_changed or not os.path.exists(config_path):
         try:
            with open(config_path, 'w', encoding='utf-8') as f:
                config.write(f)
            log_debug("Config file saved/updated with defaults.")
         except Exception as e:
             log_debug(f"Error writing config file {config_path}: {e}")
    return config

def save_app_config(config_object):
    config_path = 'ocr_translator_config.ini'
    try:
        migrate_provider_api_keys_to_credentials(config_object)
        with open(config_path, 'w', encoding='utf-8') as f:
            config_object.write(f)
        log_debug(f"Settings saved successfully to {config_path}")
        return True
    except Exception as file_err:
        log_debug(f"Error writing settings to file: {file_err}")
        return False

def load_main_window_geometry(app_config, root_window, min_size):
    try:
        # Try loading from individual components first, then full geometry string
        width_str = app_config['Settings'].get('main_window_width')
        height_str = app_config['Settings'].get('main_window_height')
        x_str = app_config['Settings'].get('main_window_x')
        y_str = app_config['Settings'].get('main_window_y')

        if all([width_str, height_str, x_str, y_str]):
            width = int(width_str)
            height = int(height_str)
            x = int(x_str)
            y = int(y_str)
        else: # Fallback to full geometry string
            geometry_str = app_config['Settings'].get('main_window_geometry', '600x480+100+100')
            parts = geometry_str.split('+')[0].split('x') + geometry_str.split('+')[1:]
            width, height, x, y = map(int, parts)

        width = max(min_size[0], width)
        height = max(min_size[1], height)
        root_window.geometry(f"{width}x{height}+{x}+{y}")
        log_debug(f"Loaded window geometry: {width}x{height}+{x}+{y}")
    except Exception as e:
        log_debug(f"Error loading window geometry: {e}. Using default 600x480+100+100.")
        root_window.geometry("600x480+100+100")


def save_main_window_geometry(app_config, root_window):
    try:
        if not root_window or not root_window.winfo_exists():
            log_debug("Window does not exist, cannot save geometry.")
            return

        geometry = root_window.geometry() # Full string e.g. "680x927+15+14"
        width = root_window.winfo_width()
        height = root_window.winfo_height()
        x = root_window.winfo_x()
        y = root_window.winfo_y()

        if width > 0 and height > 0:
            app_config['Settings']['main_window_geometry'] = geometry
            app_config['Settings']['main_window_width'] = str(width)
            app_config['Settings']['main_window_height'] = str(height)
            app_config['Settings']['main_window_x'] = str(x)
            app_config['Settings']['main_window_y'] = str(y)
            log_debug(f"Updated geometry in config object: {geometry}")
        else:
             log_debug(f"Skipping geometry save due to invalid dimensions: W={width}, H={height}")
    except Exception as e:
        log_debug(f"Error updating window geometry in config object: {e}")


def load_ocr_preview_geometry(app_config, preview_window, min_size=(400, 500)):
    """Load OCR Preview window geometry from config"""
    try:
        # Try loading from individual components first, then full geometry string
        width_str = app_config['Settings'].get('ocr_preview_width')
        height_str = app_config['Settings'].get('ocr_preview_height')
        x_str = app_config['Settings'].get('ocr_preview_x')
        y_str = app_config['Settings'].get('ocr_preview_y')

        if all([width_str, height_str, x_str, y_str]):
            width = int(width_str)
            height = int(height_str)
            x = int(x_str)
            y = int(y_str)
        else: # Fallback to full geometry string
            geometry_str = app_config['Settings'].get('ocr_preview_geometry', '600x800+100+100')
            parts = geometry_str.split('+')[0].split('x') + geometry_str.split('+')[1:]
            width, height, x, y = map(int, parts)

        width = max(min_size[0], width)
        height = max(min_size[1], height)
        preview_window.geometry(f"{width}x{height}+{x}+{y}")
        log_debug(f"Loaded OCR Preview window geometry: {width}x{height}+{x}+{y}")
    except Exception as e:
        log_debug(f"Error loading OCR Preview window geometry: {e}. Using default 600x800+100+100.")
        preview_window.geometry("600x800+100+100")


def save_ocr_preview_geometry(app_config, preview_window):
    """Save OCR Preview window geometry to config"""
    try:
        if not preview_window or not preview_window.winfo_exists():
            log_debug("OCR Preview window does not exist, cannot save geometry.")
            return

        geometry = preview_window.geometry() # Full string e.g. "680x927+15+14"
        width = preview_window.winfo_width()
        height = preview_window.winfo_height()
        x = preview_window.winfo_x()
        y = preview_window.winfo_y()

        if width > 0 and height > 0:
            app_config['Settings']['ocr_preview_geometry'] = geometry
            app_config['Settings']['ocr_preview_width'] = str(width)
            app_config['Settings']['ocr_preview_height'] = str(height)
            app_config['Settings']['ocr_preview_x'] = str(x)
            app_config['Settings']['ocr_preview_y'] = str(y)
            log_debug(f"Updated OCR Preview geometry in config object: {geometry}")
        else:
             log_debug(f"Skipping OCR Preview geometry save due to invalid dimensions: W={width}, H={height}")
    except Exception as e:
        log_debug(f"Error updating OCR Preview window geometry in config object: {e}")
