import re
import importlib
import io
import threading
from collections import OrderedDict
from dataclasses import dataclass
from logger import log_debug, log_debug_coalesced


def _pil_image():
    from PIL import Image
    return Image


API_OCR_IMAGE_MODE_LOSSLESS_WEBP = 'lossless_webp'
API_OCR_IMAGE_MODE_BALANCED_WEBP = 'balanced_webp'
API_OCR_IMAGE_MODE_SMALL_GRAYSCALE_WEBP = 'small_grayscale_webp'
API_OCR_IMAGE_MODE_DEFAULT = API_OCR_IMAGE_MODE_BALANCED_WEBP
API_OCR_IMAGE_MODES = (
    API_OCR_IMAGE_MODE_LOSSLESS_WEBP,
    API_OCR_IMAGE_MODE_BALANCED_WEBP,
    API_OCR_IMAGE_MODE_SMALL_GRAYSCALE_WEBP,
)
API_OCR_IMAGE_QUALITY_DEFAULT = 85
API_OCR_IMAGE_DETAILS = ('auto', 'low', 'high')
API_OCR_IMAGE_DETAIL_DEFAULT = 'auto'
API_OCR_IMAGE_FORMAT_WEBP = 'webp'
API_OCR_IMAGE_FORMAT_PNG = 'png'
API_OCR_IMAGE_FORMAT_JPEG = 'jpeg'
API_OCR_IMAGE_FORMAT_DEFAULT = API_OCR_IMAGE_FORMAT_WEBP
API_OCR_IMAGE_FORMATS = (
    API_OCR_IMAGE_FORMAT_WEBP,
    API_OCR_IMAGE_FORMAT_PNG,
    API_OCR_IMAGE_FORMAT_JPEG,
)
API_OCR_IMAGE_MIME_TYPES = {
    API_OCR_IMAGE_FORMAT_WEBP: 'image/webp',
    API_OCR_IMAGE_FORMAT_PNG: 'image/png',
    API_OCR_IMAGE_FORMAT_JPEG: 'image/jpeg',
}


@dataclass(frozen=True)
class EncodedApiOcrImage:
    data: bytes
    mime_type: str
    image_format: str
    image_detail: str = API_OCR_IMAGE_DETAIL_DEFAULT
    policy_reason: str = ""


def _normalize_capture_region(region):
    x, y, width, height = map(int, region)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid capture region: {region}")
    return x, y, width, height


def _validate_capture_image(image, backend):
    size = getattr(image, 'size', None)
    if not size or len(size) != 2:
        raise ValueError(f"{backend} returned an invalid image")
    width, height = map(int, size)
    if width <= 0 or height <= 0:
        raise ValueError(f"{backend} returned an empty image")
    return image


def _set_capture_metadata(image, backend, fallback_reason=None):
    try:
        image._gct_capture_backend = backend
        image._gct_capture_fallback_reason = fallback_reason
    except Exception:
        pass
    return image


def _capture_with_mss(region, mss_factory=None):
    x, y, width, height = _normalize_capture_region(region)
    factory = mss_factory
    if factory is None:
        mss_module = importlib.import_module('mss')
        factory = mss_module.mss

    with factory() as sct:
        monitor = {"left": x, "top": y, "width": width, "height": height}
        shot = sct.grab(monitor)
        return _pil_image().frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


def capture_screen_region(region, mss_factory=None):
    """Capture a screen region as a PIL RGB image using the MSS backend."""
    x, y, width, height = _normalize_capture_region(region)
    try:
        image = _capture_with_mss(
            (x, y, width, height),
            mss_factory=mss_factory,
        )
        image = _validate_capture_image(image, "mss")
        _set_capture_metadata(image, "mss")
        log_debug_coalesced(
            ("capture-success", "mss", width, height),
            f"CAPTURE: mss captured {width}x{height}",
            interval_seconds=5.0,
        )
        return image
    except Exception as e:
        log_debug(
            "CAPTURE: mss backend failed "
            f"({type(e).__name__}: {e})"
        )
        raise RuntimeError(
            f"MSS capture failed for region {width}x{height}"
        ) from e


def normalize_api_ocr_image_mode(mode):
    normalized = str(mode or API_OCR_IMAGE_MODE_DEFAULT).strip().lower()
    if normalized in API_OCR_IMAGE_MODES:
        return normalized
    return API_OCR_IMAGE_MODE_DEFAULT


def normalize_api_ocr_image_quality(quality):
    try:
        normalized = int(quality)
    except (TypeError, ValueError):
        normalized = API_OCR_IMAGE_QUALITY_DEFAULT
    return max(1, min(100, normalized))


def normalize_api_ocr_image_detail(detail):
    normalized = str(detail or API_OCR_IMAGE_DETAIL_DEFAULT).strip().lower()
    if normalized in API_OCR_IMAGE_DETAILS:
        return normalized
    return API_OCR_IMAGE_DETAIL_DEFAULT


def normalize_api_ocr_image_format(image_format):
    normalized = str(image_format or API_OCR_IMAGE_FORMAT_DEFAULT).strip().lower()
    if normalized == 'jpg':
        normalized = API_OCR_IMAGE_FORMAT_JPEG
    if normalized in API_OCR_IMAGE_FORMATS:
        return normalized
    return API_OCR_IMAGE_FORMAT_DEFAULT


def _flatten_transparency_for_api_ocr(pil_image):
    Image = _pil_image()
    if pil_image.mode in ('RGBA', 'LA'):
        rgb_img = Image.new('RGB', pil_image.size, (255, 255, 255))
        if pil_image.mode == 'RGBA':
            rgb_img.paste(pil_image, mask=pil_image.split()[-1])
        else:
            rgb_img.paste(pil_image)
        return rgb_img
    if pil_image.mode == 'P' and getattr(pil_image, 'info', {}).get('transparency') is not None:
        return _flatten_transparency_for_api_ocr(pil_image.convert('RGBA'))
    return pil_image


def _prepare_image_for_api_ocr(pil_image, mode):
    image = _flatten_transparency_for_api_ocr(pil_image)
    if mode == API_OCR_IMAGE_MODE_SMALL_GRAYSCALE_WEBP:
        if image.mode != 'L':
            image = image.convert('L')
        return image
    if image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')
    return image


def encode_image_for_api_ocr_payload(
    pil_image,
    mode=API_OCR_IMAGE_MODE_DEFAULT,
    quality=API_OCR_IMAGE_QUALITY_DEFAULT,
    image_format=API_OCR_IMAGE_FORMAT_DEFAULT,
):
    """Encode a PIL image for Custom AI OCR API upload with MIME metadata."""
    if pil_image is None:
        raise ValueError("Cannot encode an empty OCR image")

    normalized_mode = normalize_api_ocr_image_mode(mode)
    normalized_quality = normalize_api_ocr_image_quality(quality)
    normalized_format = normalize_api_ocr_image_format(image_format)
    image = _prepare_image_for_api_ocr(pil_image, normalized_mode)
    buffer = io.BytesIO()

    if normalized_format == API_OCR_IMAGE_FORMAT_WEBP and normalized_mode == API_OCR_IMAGE_MODE_LOSSLESS_WEBP:
        image.save(
            buffer,
            format='WebP',
            lossless=True,
            method=0,
            exact=True,
        )
    elif normalized_format == API_OCR_IMAGE_FORMAT_WEBP:
        image.save(
            buffer,
            format='WebP',
            quality=normalized_quality,
            method=3,
        )
    elif normalized_format == API_OCR_IMAGE_FORMAT_PNG:
        image.save(
            buffer,
            format='PNG',
            optimize=True,
        )
    else:
        image.save(
            buffer,
            format='JPEG',
            quality=normalized_quality,
            optimize=True,
        )

    image_bytes = buffer.getvalue()
    if not image_bytes:
        raise ValueError("Encoded OCR image payload is empty")
    return EncodedApiOcrImage(
        data=image_bytes,
        mime_type=API_OCR_IMAGE_MIME_TYPES[normalized_format],
        image_format=normalized_format,
    )


def encode_image_for_api_ocr(
    pil_image,
    mode=API_OCR_IMAGE_MODE_DEFAULT,
    quality=API_OCR_IMAGE_QUALITY_DEFAULT,
):
    """Encode a PIL image for Custom AI OCR API upload as bytes."""
    return encode_image_for_api_ocr_payload(
        pil_image,
        mode=mode,
        quality=quality,
        image_format=API_OCR_IMAGE_FORMAT_WEBP,
    ).data


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
            log_debug_coalesced(
                "ocr-frame-cache-hit",
                "OCR CACHE: frame hit",
                interval_seconds=5.0,
            )
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
