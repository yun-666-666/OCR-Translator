import re
import importlib
import os
import shutil
import sys
from functools import lru_cache
import threading
from collections import OrderedDict
from logger import log_debug


def _cv2():
    import cv2
    return cv2


def _np():
    import numpy as np
    return np


def _tesserocr():
    return importlib.import_module('tesserocr')


def _pil_image():
    from PIL import Image
    return Image


class TesseractOcrUnavailableError(RuntimeError):
    """Raised when the local Tesseract API wrapper cannot run OCR."""


def resolve_tessdata_dir_from_tesseract_path(tesseract_path):
    """Resolve the tessdata directory from the existing Tesseract path setting."""
    return _resolve_tessdata_dir_from_tesseract_path_cached(
        os.path.normpath(str(tesseract_path)) if tesseract_path else None,
        os.path.normpath(os.environ.get('TESSDATA_PREFIX', '')) if os.environ.get('TESSDATA_PREFIX') else None,
        os.path.normpath(os.environ.get('LOCALAPPDATA', '')) if os.environ.get('LOCALAPPDATA') else None,
        _get_pyinstaller_runtime_root(),
        _get_tesserocr_package_root(),
        _get_tesseract_executable_from_path(),
    )


def clear_tessdata_dir_cache():
    _resolve_tessdata_dir_from_tesseract_path_cached.cache_clear()


def get_tesseract_languages(tessdata_dir=None):
    """Return the resolved tessdata directory and cached supported language codes."""
    effective_tessdata_dir = tessdata_dir or resolve_tessdata_dir_from_tesseract_path(None)
    normalized_tessdata_dir = os.path.normpath(str(effective_tessdata_dir)) if effective_tessdata_dir else None
    return _get_tesseract_languages_cached(normalized_tessdata_dir)


def clear_tesseract_languages_cache():
    _get_tesseract_languages_cached.cache_clear()


def _get_windows_tesseract_install_dirs():
    if os.name != 'nt':
        return ()

    try:
        import winreg
    except Exception:
        return ()

    roots = []
    seen = set()
    key_paths = [
        (winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\Tesseract-OCR'),
        (winreg.HKEY_CURRENT_USER, r'SOFTWARE\Tesseract-OCR'),
        (winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\WOW6432Node\Tesseract-OCR'),
        (winreg.HKEY_CURRENT_USER, r'SOFTWARE\WOW6432Node\Tesseract-OCR'),
    ]
    value_names = ('InstallDir', '')
    for hive, key_path in key_paths:
        try:
            with winreg.OpenKey(hive, key_path) as key:
                for value_name in value_names:
                    try:
                        value, _value_type = winreg.QueryValueEx(key, value_name)
                    except OSError:
                        continue
                    normalized = os.path.normpath(str(value)) if value else ''
                    if normalized and normalized not in seen:
                        seen.add(normalized)
                        roots.append(normalized)
        except OSError:
            continue

    return tuple(roots)


def _get_pyinstaller_runtime_root():
    if getattr(sys, 'frozen', False):
        runtime_root = getattr(sys, '_MEIPASS', None)
        if runtime_root:
            return os.path.normpath(str(runtime_root))
    return None


def _get_tesserocr_package_root():
    try:
        tesserocr = _tesserocr()
    except Exception:
        return None

    package_file = getattr(tesserocr, '__file__', None)
    if not package_file:
        return None
    return os.path.normpath(os.path.dirname(str(package_file)))


def _get_tesseract_executable_from_path():
    executable = shutil.which('tesseract') or shutil.which('tesseract.exe')
    return os.path.normpath(str(executable)) if executable else None


@lru_cache(maxsize=128)
def _resolve_tessdata_dir_from_tesseract_path_cached(
    tesseract_path,
    tessdata_prefix,
    localappdata,
    pyinstaller_root,
    tesserocr_package_root,
    path_tesseract_exe,
):
    candidates = []

    def _append(path):
        if not path:
            return
        normalized = os.path.normpath(str(path))
        if normalized not in candidates:
            candidates.append(normalized)

    if tesseract_path:
        normalized = tesseract_path
        if os.path.basename(normalized).lower() == 'tessdata':
            _append(normalized)
        elif os.path.isdir(normalized):
            _append(os.path.join(normalized, 'tessdata'))
            _append(normalized)
        else:
            parent_dir = os.path.dirname(normalized)
            if parent_dir:
                _append(os.path.join(parent_dir, 'tessdata'))

    if tessdata_prefix:
        normalized_prefix = tessdata_prefix
        if os.path.basename(normalized_prefix).lower() == 'tessdata':
            _append(normalized_prefix)
        else:
            _append(os.path.join(normalized_prefix, 'tessdata'))
            _append(normalized_prefix)

    for runtime_root in (pyinstaller_root, tesserocr_package_root):
        if runtime_root:
            _append(os.path.join(runtime_root, 'tessdata'))

    if tesserocr_package_root:
        parent_dir = os.path.dirname(tesserocr_package_root)
        if parent_dir:
            _append(os.path.join(parent_dir, 'tessdata'))
            _append(os.path.join(parent_dir, 'tesserocr.libs', 'tessdata'))

    if path_tesseract_exe:
        _append(os.path.join(os.path.dirname(path_tesseract_exe), 'tessdata'))

    for root in _get_windows_tesseract_install_dirs():
        normalized_root = os.path.normpath(str(root))
        basename = os.path.basename(normalized_root).lower()
        if basename == 'tessdata':
            _append(normalized_root)
        elif basename == 'tesseract.exe':
            _append(os.path.join(os.path.dirname(normalized_root), 'tessdata'))
        else:
            _append(os.path.join(normalized_root, 'tessdata'))
            _append(normalized_root)

    common_roots = [
        r'C:\Program Files\Tesseract-OCR',
        r'C:\Program Files (x86)\Tesseract-OCR',
        os.path.join(localappdata, 'Programs', 'Tesseract-OCR') if localappdata else None,
    ]
    for root in common_roots:
        if root:
            _append(os.path.join(root, 'tessdata'))

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    return None


@lru_cache(maxsize=128)
def _get_tesseract_languages_cached(tessdata_dir):
    try:
        tesserocr = _tesserocr()
        resolved_tessdata_dir, available_languages = tesserocr.get_languages(tessdata_dir)
    except Exception as e:
        raise TesseractOcrUnavailableError(
            f"Could not enumerate supported Tesseract languages for tessdata '{tessdata_dir or '<default>'}'"
        ) from e

    normalized_dir = os.path.normpath(str(resolved_tessdata_dir)) if resolved_tessdata_dir else (
        os.path.normpath(str(tessdata_dir)) if tessdata_dir else None
    )
    normalized_languages = tuple(
        sorted({
            str(language).strip().lower()
            for language in (available_languages or [])
            if str(language).strip()
        })
    )
    return normalized_dir, normalized_languages


def _normalize_tesseract_lang_code(lang_code):
    normalized = str(lang_code or '').strip().lower()
    if not normalized or normalized == 'auto':
        return 'eng'
    return normalized


def _split_tesseract_lang_codes(lang_code):
    return tuple(
        part.strip().lower()
        for part in str(lang_code or '').split('+')
        if part.strip()
    )


def get_tesseract_ocr_config(mode='general'):
    """Return structured tesserocr options for the existing OCR modes."""
    mode = (mode or 'general').lower()
    config = {
        'psm': 6,
        'oem': 3,
        'variables': {},
    }
    if mode == 'subtitle':
        config['psm'] = 7
        config['variables']['preserve_interword_spaces'] = '1'
    elif mode == 'gaming':
        config['psm'] = 6
        config['variables']['tessedit_char_whitelist'] = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            ".,!?:;()[]-_\'\"/\\$%&@ "
        )
    elif mode == 'document':
        config['psm'] = 3
    return config


def normalize_tesseract_ocr_config(config_or_string):
    """Accept the legacy config string or the structured tesserocr config."""
    if isinstance(config_or_string, dict):
        variables = config_or_string.get('variables') or {}
        return {
            'psm': int(config_or_string.get('psm', 6)),
            'oem': int(config_or_string.get('oem', 3)),
            'variables': {str(k): str(v) for k, v in variables.items()},
        }

    config_text = str(config_or_string or '')
    parsed = get_tesseract_ocr_config('general')
    psm_match = re.search(r'--psm\s+(\d+)', config_text)
    oem_match = re.search(r'--oem\s+(\d+)', config_text)
    if psm_match:
        parsed['psm'] = int(psm_match.group(1))
    if oem_match:
        parsed['oem'] = int(oem_match.group(1))
    if 'preserve_interword_spaces=1' in config_text:
        parsed['variables']['preserve_interword_spaces'] = '1'
    whitelist_match = re.search(r'tessedit_char_whitelist=(.*)', config_text)
    if whitelist_match:
        parsed['variables']['tessedit_char_whitelist'] = whitelist_match.group(1)
    return parsed


class TesseractOcrEngine:
    """Small reusable wrapper around tesserocr's PyTessBaseAPI."""

    def __init__(self, tessdata_dir=None, tesserocr_module=None):
        self.tessdata_dir = tessdata_dir
        self._tesserocr_module = tesserocr_module
        self._api = None
        self._api_key = None
        self._lock = threading.RLock()

    def close(self):
        with self._lock:
            if self._api is not None:
                for close_method in ('End', 'Close'):
                    method = getattr(self._api, close_method, None)
                    if method is not None:
                        try:
                            method()
                        except Exception as e:
                            log_debug(f"tesserocr API close failed: {e}")
                        break
            self._api = None
            self._api_key = None

    def _module(self):
        if self._tesserocr_module is None:
            try:
                self._tesserocr_module = _tesserocr()
            except ImportError as e:
                raise TesseractOcrUnavailableError(
                    "tesserocr is not installed. Install tesserocr to use local Tesseract OCR."
                ) from e
        return self._tesserocr_module

    def _ensure_api(self, lang_code, ocr_config):
        tesserocr = self._module()
        config = normalize_tesseract_ocr_config(ocr_config)
        effective_lang_code = _normalize_tesseract_lang_code(lang_code)
        variables_key = tuple(sorted(config['variables'].items()))
        api_key = (
            self.tessdata_dir or '',
            effective_lang_code,
            config['psm'],
            config['oem'],
            variables_key,
        )
        if self._api is not None and self._api_key == api_key:
            return self._api, tesserocr, config

        available_tessdata_dir, available_languages = get_tesseract_languages(self.tessdata_dir)
        requested_languages = _split_tesseract_lang_codes(effective_lang_code)
        missing_languages = [language for language in requested_languages if language not in available_languages]
        if missing_languages:
            available_language_preview = ', '.join(available_languages[:12]) if available_languages else '<none>'
            raise TesseractOcrUnavailableError(
                "Requested Tesseract language(s) "
                f"{', '.join(missing_languages)} are not available in tessdata "
                f"'{available_tessdata_dir or self.tessdata_dir or '<default>'}'. "
                f"Available languages: {available_language_preview}"
            )

        self.close()
        kwargs = {
            'lang': effective_lang_code,
            'psm': config['psm'],
            'oem': config['oem'],
        }
        if self.tessdata_dir:
            kwargs['path'] = self.tessdata_dir

        try:
            api = tesserocr.PyTessBaseAPI(**kwargs)
            for name, value in config['variables'].items():
                api.SetVariable(name, value)
        except Exception as e:
            raise TesseractOcrUnavailableError(
                f"Could not initialize tesserocr for language '{effective_lang_code}'"
            ) from e

        self._api = api
        self._api_key = api_key
        log_debug(
            "tesserocr API initialized "
            f"lang={effective_lang_code} psm={config['psm']} tessdata={self.tessdata_dir or '<default>'}"
        )
        return self._api, tesserocr, config

    def _iter_words_with_confidence(self, api, tesserocr, confidence_threshold):
        iterator = api.GetIterator()
        if iterator is None:
            return None

        level = tesserocr.RIL.WORD
        iterate_level = getattr(tesserocr, 'iterate_level', None)
        if iterate_level is not None:
            iterator_source = iterate_level(iterator, level)
        else:
            iterator_source = _manual_tesserocr_iterator(iterator, level)

        filtered_text = []
        for result_item in iterator_source:
            word = result_item.GetUTF8Text(level)
            if not word or not word.strip():
                continue
            confidence = float(result_item.Confidence(level))
            if confidence >= confidence_threshold:
                filtered_text.append(word.strip())
            else:
                log_debug(f"Filtered low-confidence text: '{word.strip()}' ({confidence}%)")
        return filtered_text

    def recognize(self, pil_image, lang_code, ocr_config, confidence_threshold):
        with self._lock:
            api, tesserocr, config = self._ensure_api(lang_code, ocr_config)
            api.SetImage(pil_image)
            recognize = getattr(api, 'Recognize', None)
            if recognize is not None:
                recognize()

            try:
                filtered_words = self._iter_words_with_confidence(api, tesserocr, confidence_threshold)
            except Exception as e:
                log_debug(f"tesserocr iterator OCR failed, using full text fallback: {e}")
                filtered_words = None

            if filtered_words is not None:
                return ' '.join(filtered_words)

            text = api.GetUTF8Text() or ''
            mean_confidence = getattr(api, 'MeanTextConf', lambda: confidence_threshold)()
            if float(mean_confidence) < confidence_threshold:
                log_debug(f"Filtered low-confidence OCR result ({mean_confidence}%)")
                return ''
            return text


def _manual_tesserocr_iterator(iterator, level):
    yield iterator
    while iterator.Next(level):
        yield iterator


_TESSERACT_ENGINE_CACHE = {}
_TESSERACT_ENGINE_CACHE_LOCK = threading.RLock()


def get_tesseract_ocr_engine(tessdata_dir=None):
    """Return a reusable local OCR engine for the given tessdata directory."""
    key = tessdata_dir or '<default>'
    with _TESSERACT_ENGINE_CACHE_LOCK:
        engine = _TESSERACT_ENGINE_CACHE.get(key)
        if engine is None:
            engine = TesseractOcrEngine(tessdata_dir=tessdata_dir)
            _TESSERACT_ENGINE_CACHE[key] = engine
        return engine


def clear_tesseract_ocr_engines():
    with _TESSERACT_ENGINE_CACHE_LOCK:
        engines = list(_TESSERACT_ENGINE_CACHE.values())
        _TESSERACT_ENGINE_CACHE.clear()
    for engine in engines:
        engine.close()


def capture_screen_region(region, backend='auto', mss_factory=None, pyautogui_module=None):
    """Capture a screen region as a PIL RGB image using the fastest available backend."""
    x, y, width, height = map(int, region)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid capture region: {region}")

    selected_backend = (backend or 'auto').lower()
    if selected_backend not in ('auto', 'mss', 'pyautogui'):
        log_debug(f"Unknown capture backend '{backend}', falling back to auto")
        selected_backend = 'auto'

    if selected_backend in ('auto', 'mss'):
        try:
            factory = mss_factory
            if factory is None:
                mss_module = importlib.import_module('mss')
                factory = mss_module.mss

            with factory() as sct:
                monitor = {"left": x, "top": y, "width": width, "height": height}
                shot = sct.grab(monitor)
                image = _pil_image().frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                log_debug(f"CAPTURE: mss captured {width}x{height}")
                return image
        except Exception as e:
            log_debug(f"CAPTURE: mss backend failed ({type(e).__name__}: {e}); falling back to pyautogui")
            if selected_backend == 'mss':
                # Explicit mss still falls back to keep translation running.
                pass

    if pyautogui_module is None:
        pyautogui_module = importlib.import_module('pyautogui')
    image = pyautogui_module.screenshot(region=(x, y, width, height))
    log_debug(f"CAPTURE: pyautogui captured {width}x{height}")
    return image


def build_capture_signature(image_hash, region, backend):
    """Build a capture signature that changes when pixels, region, or backend changes."""
    x, y, width, height = map(int, region)
    return (
        str(image_hash),
        x,
        y,
        width,
        height,
        str(backend or 'auto').lower(),
    )


def build_ocr_frame_cache_key(image_hash, ocr_model, source_lang, preprocessing_mode, region_size, region_origin=None):
    """Build a stable key for OCR results from equivalent frames/settings."""
    width, height = region_size
    if region_origin is None:
        origin_x, origin_y = 0, 0
    else:
        origin_x, origin_y = region_origin
    return (
        str(image_hash),
        str(ocr_model or '').lower(),
        str(source_lang or '').lower(),
        str(preprocessing_mode or '').lower(),
        int(origin_x),
        int(origin_y),
        int(width),
        int(height),
    )


class OCRFrameCache:
    """Small thread-safe LRU cache for repeated subtitle frames."""

    def __init__(self, max_size=64):
        self.max_size = max(0, int(max_size or 0))
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key):
        if self.max_size <= 0:
            return None
        with self._lock:
            if key not in self._cache:
                return None
            value = self._cache.pop(key)
            self._cache[key] = value
            log_debug("OCR CACHE: frame hit")
            return value

    def put(self, key, text):
        if self.max_size <= 0 or text is None or not str(text).strip() or str(text).strip() == "<EMPTY>":
            return
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            self._cache[key] = text
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def resize(self, max_size):
        with self._lock:
            self.max_size = max(0, int(max_size or 0))
            if self.max_size <= 0:
                self._cache.clear()
                return
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()


def preprocess_for_ocr(img, mode='adaptive', block_size=41, c_value=-60): # Parameters are now configurable
    """
    Preprocesses an image for OCR.

    Args:
        img: The input image (NumPy array).
        mode: The type of preprocessing to apply.
              'adaptive': Applies adaptive thresholding (recommended for varying backgrounds).
              'binary': Applies fixed binary thresholding (inverted).
              'binary_inv': Applies fixed binary thresholding.
              'none': Returns the grayscale image without thresholding.
              Defaults to 'adaptive'.
        block_size: Size of the pixel neighborhood used for adaptive thresholding.
                   Must be an odd number. Only used for 'adaptive' mode.
        c_value: Constant subtracted from the mean for adaptive thresholding.
                Can be positive, negative, or zero. Only used for 'adaptive' mode.
    Returns:
        The processed image.
    """
    if img is None or img.size == 0:
        log_debug("Input image to preprocess_for_ocr is empty or None.")
        # Return a small black image or handle as an error appropriately
        return _np().zeros((10, 10), dtype=_np().uint8)

    if len(img.shape) == 3:
        gray = _cv2().cvtColor(img, _cv2().COLOR_BGR2GRAY)
    else:
        gray = img.copy() # Ensure it's a copy if already grayscale

    try:
        if mode == 'adaptive':
            # --- Adaptive Thresholding ---
            # Use the configurable parameters
            blockSize = block_size  # Size of the pixel neighborhood used to calculate the threshold.
                                    # Must be an odd number (e.g., 3, 5, 7, 11, 21).
                                    # Smaller for smaller text/details, larger for larger features.
            C = c_value             # A constant subtracted from the mean or weighted mean.
                                    # Normally, it is positive but may be zero or negative as well.
                                    # Helps fine-tune the threshold.

            # cv2.ADAPTIVE_THRESH_GAUSSIAN_C often gives better results than cv2.ADAPTIVE_THRESH_MEAN_C.
            # cv2.THRESH_BINARY_INV is used because Tesseract generally prefers black text on a white background.
            cv2 = _cv2()
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, blockSize, C)
            log_debug(f"Used adaptive thresholding (blockSize={blockSize}, C={C})")
            # --- End of Adaptive Thresholding ---

        elif mode == 'binary':
            # Original fixed thresholding (inverted: white text becomes black)
            _, processed = _cv2().threshold(gray, 150, 255, _cv2().THRESH_BINARY_INV)
            log_debug(f"Used fixed binary thresholding (INV)")
        elif mode == 'binary_inv':
            # Original fixed thresholding (white text stays white)
            _, processed = _cv2().threshold(gray, 150, 255, _cv2().THRESH_BINARY)
            log_debug(f"Used fixed binary thresholding (standard)")
        elif mode == 'none': # Explicitly handle 'none'
            processed = gray
            log_debug(f"No thresholding applied, using grayscale. Mode: {mode}")
        else: # Fallback for unrecognized modes
            log_debug(f"Unrecognized mode: '{mode}'. Defaulting to grayscale.")
            processed = gray
            
        return processed

    except Exception as e:
        log_debug(f"Preprocessing error (mode: {mode}): {e}, returning grayscale image")
        # In case of any error during processing, return the grayscale image
        return gray

def scale_for_ocr(img):
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    min_dim = 300
    if h < min_dim or w < min_dim:
        scale_factor = max(min_dim / h, min_dim / w)
        scaled = _cv2().resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=_cv2().INTER_CUBIC)
        return scaled
    return img

def get_tesseract_model_params(mode='general'):
    config = get_tesseract_ocr_config(mode)
    parts = [f"--psm {config['psm']}", f"--oem {config['oem']}"]
    for name, value in config['variables'].items():
        parts.extend(['-c', f'{name}={value}'])
    return ' '.join(parts)

def ocr_region_with_confidence(
    img,
    region,
    lang_code,
    custom_config,
    confidence_threshold,
    ocr_engine=None,
    tessdata_dir=None,
):
    x, y, w, h = region
    if len(img.shape) == 3:
        roi = img[y:y+h, x:x+w]
    else:
        roi = img[y:y+h, x:x+w]
    
    scaled_roi = scale_for_ocr(roi)
    
    try:
        pil_roi = _pil_image().fromarray(scaled_roi)
        engine = ocr_engine or get_tesseract_ocr_engine(tessdata_dir=tessdata_dir)
        return engine.recognize(
            pil_roi,
            lang_code,
            normalize_tesseract_ocr_config(custom_config),
            confidence_threshold,
        )
    except TesseractOcrUnavailableError:
        raise
    except Exception as e:
        log_debug(f"OCR error in region {region}: {e}")
        return ""

def post_process_ocr_text_general(text, lang='auto'):
    if not text: return text
    cleaned = text.strip()
    ocr_errors = {
        '\u201E': '"', '\u2019': "'", '\u2014': '-', '\u2013': '-',
    }
    if lang.startswith('fra') or lang.startswith('fr'):
        cleaned = cleaned.replace('||', 'Il')
    
    # English-specific OCR fixes
    if lang.startswith('eng') or lang.startswith('en'):
        # Special case for | character (commonly at start of sentences)
        cleaned = re.sub(r'^\|\s', 'I ', cleaned)  # | at start followed by space
        cleaned = re.sub(r'\s\|\s', ' I ', cleaned)  # | surrounded by spaces
        
        # Other fixes using word boundaries
        english_ocr_fixes = {
            '{': '(', '}': ')', '\\/': 'V',
        }
        for error, correction in english_ocr_fixes.items():
            cleaned = re.sub(r'\b' + re.escape(error) + r'\b', correction, cleaned)
    
    for error, correction in ocr_errors.items():
        cleaned = re.sub(r'\b' + re.escape(error) + r'\b', correction, cleaned)
    
    # Preserve newlines, only collapse multiple spaces/tabs
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    
    return cleaned

def remove_text_after_last_punctuation_mark(text):
    if not text: return text
    pattern = r'[.!?]|\.{3}|…'
    matches = list(re.finditer(pattern, text))
    if not matches: return text
    last_match = matches[-1]
    end_pos = last_match.end()
    if last_match.group() == ".":
        if end_pos + 2 <= len(text) and text[end_pos:end_pos+2] == "..":
            end_pos += 2
    return text[:end_pos]

def post_process_ocr_for_game_subtitle(text):
    if not text: return text
    cleaned = text.strip()
    name_match = re.search(r'^([A-Za-z\s]+):', cleaned)
    if name_match:
        character_name = name_match.group(1).strip()
        character_name = ' '.join(word.capitalize() for word in character_name.split())
        cleaned = cleaned.replace(name_match.group(0), f"{character_name}:")
    substitutions = {
        # "l-": "I-", "ledi": "Jedi", "jedl": "Jedi",  # Commented out - too aggressive
        # "RepubIic": "Republic", "repubIic": "republic",  # Commented out - too aggressive
    }
    for error, correction in substitutions.items():
        cleaned = cleaned.replace(error, correction)
    cleaned = re.sub(r'(\w+:)(\w)', r'\1 \2', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'^[\|\[\]\{\}<>\s\.,;:_\-=+\'\"]{1,5}', '', cleaned)
    cleaned = re.sub(r'[\|\[\]\{\}<>\s\.,;:_\-=+\'\"]{1,5}$', '', cleaned)
    return cleaned
