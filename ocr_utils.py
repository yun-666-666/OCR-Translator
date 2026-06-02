import re
import asyncio
import importlib
import io
import threading
from collections import OrderedDict
from logger import log_debug


def _cv2():
    import cv2
    return cv2


def _np():
    import numpy as np
    return np


def _pytesseract():
    import pytesseract
    return pytesseract


def _pil_image():
    from PIL import Image
    return Image


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

    def clear(self):
        with self._lock:
            self._cache.clear()


def _import_windows_ocr_modules(importer=importlib.import_module):
    prefixes = ('winsdk', 'winrt')
    last_error = None
    for prefix in prefixes:
        try:
            return {
                'ocr': importer(f'{prefix}.windows.media.ocr'),
                'globalization': importer(f'{prefix}.windows.globalization'),
                'imaging': importer(f'{prefix}.windows.graphics.imaging'),
                'streams': importer(f'{prefix}.windows.storage.streams'),
            }
        except ImportError as e:
            last_error = e
    raise last_error or ImportError("WinRT OCR modules are not available")


def is_windows_ocr_available(importer=importlib.import_module):
    """Return True when Python WinRT bindings for Windows OCR can be imported."""
    try:
        modules = _import_windows_ocr_modules(importer)
        if not all(modules.values()):
            return False
        try:
            engine = modules['ocr'].OcrEngine.try_create_from_user_profile_languages()
            return engine is not None
        except Exception as e:
            log_debug(f"Windows OCR availability check failed: {e}")
            return False
    except Exception:
        return False


def _to_windows_language_code(lang_code):
    lang = (lang_code or '').lower().replace('_', '-')
    mapping = {
        'eng': 'en',
        'jpn': 'ja',
        'jpn-vert': 'ja',
        'kor': 'ko',
        'chi-sim': 'zh-CN',
        'chi_sim': 'zh-CN',
        'chi-tra': 'zh-TW',
        'chi_tra': 'zh-TW',
        'deu': 'de',
        'fra': 'fr',
        'spa': 'es',
        'ita': 'it',
        'pol': 'pl',
        'por': 'pt',
        'rus': 'ru',
        'ukr': 'uk',
        'ces': 'cs',
        'cze': 'cs',
        'nld': 'nl',
        'swe': 'sv',
        'dan': 'da',
        'fin': 'fi',
        'nor': 'no',
    }
    return mapping.get(lang, lang if len(lang) in (2, 5) else 'en')


async def _recognize_windows_ocr_async(pil_image, lang_code):
    modules = _import_windows_ocr_modules()
    ocr_module = modules['ocr']
    globalization = modules['globalization']
    imaging = modules['imaging']
    streams = modules['streams']

    rgb_image = pil_image.convert('RGB')
    buffer = io.BytesIO()
    rgb_image.save(buffer, format='BMP')
    image_bytes = buffer.getvalue()

    stream = streams.InMemoryRandomAccessStream()
    writer = streams.DataWriter(stream)
    writer.write_bytes(image_bytes)
    await writer.store_async()
    await writer.flush_async()
    writer.detach_stream()
    stream.seek(0)

    decoder = await imaging.BitmapDecoder.create_async(stream)
    bitmap = await decoder.get_software_bitmap_async()

    engine = None
    try:
        language = globalization.Language(_to_windows_language_code(lang_code))
        engine = ocr_module.OcrEngine.try_create_from_language(language)
    except Exception as e:
        log_debug(f"Windows OCR language initialization failed for {lang_code}: {e}")

    if engine is None:
        engine = ocr_module.OcrEngine.try_create_from_user_profile_languages()
    if engine is None:
        raise RuntimeError("Windows OCR engine is not available for the requested language")

    result = await engine.recognize_async(bitmap)
    lines = []
    for line in getattr(result, 'lines', []) or []:
        line_text = getattr(line, 'text', '')
        if line_text:
            lines.append(line_text)
    return "\n".join(lines).strip() or "<EMPTY>"


def recognize_windows_ocr(pil_image, lang_code):
    """Best-effort Windows OCR wrapper. Raises if WinRT OCR cannot run."""
    return asyncio.run(_recognize_windows_ocr_async(pil_image, lang_code))

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
    if mode == 'subtitle':
        return f'--psm 7 --oem 3 -c preserve_interword_spaces=1'
    elif mode == 'gaming':
        return f'--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;()[]-_\'"/\\$%&@ '
    elif mode == 'document':
        return f'--psm 3 --oem 3'
    else:
        return f'--psm 6 --oem 3'

def ocr_region_with_confidence(img, region, lang_code, custom_config, confidence_threshold):
    x, y, w, h = region
    if len(img.shape) == 3:
        roi = img[y:y+h, x:x+w]
    else:
        roi = img[y:y+h, x:x+w]
    
    scaled_roi = scale_for_ocr(roi)
    
    try:
        pytesseract = _pytesseract()
        data = pytesseract.image_to_data(
            scaled_roi,
            lang=lang_code,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
        filtered_text = []
        for i in range(len(data['text'])):
            if not data['text'][i].strip():
                continue
            if float(data['conf'][i]) >= confidence_threshold:
                filtered_text.append(data['text'][i])
            else:
                log_debug(f"Filtered low-confidence text: '{data['text'][i]}' ({data['conf'][i]}%)")
        return ' '.join(filtered_text)
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
