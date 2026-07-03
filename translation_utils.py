import re
from logger import log_debug

TRANSLATION_ERROR_PREFIXES = (
    "err:",
    "translation error:",
    "custom ai translation error:",
    "ai model profile for translation is missing",
    "error: unknown translation model",
    "no translation for model:",
    "google api error:",
    "google api key missing:",
    "google translate api key missing",
    "google translate api client not initialized",
    "requests library not available for google translate",
    "google translate api returned unexpected result:",
    "google translate api request error:",
    "google translate api error:",
    "google client init error:",
    "deepl api libraries not available",
    "deepl api key missing",
    "deepl client init error:",
    "deepl api client not initialized",
    "deepl api returned empty or invalid result",
    "deepl api fallback returned empty or invalid result",
    "deepl api error:",
    "marianmt error:",
    "marianmt not initialized",
    "marianmt language pair not determined:",
    "marianmt translator not initialized",
    "marianmt translation error:",
)


def is_translation_error_result(value):
    if not isinstance(value, str):
        return True
    normalized = value.lstrip().casefold()
    return normalized.startswith(TRANSLATION_ERROR_PREFIXES)


# Note: This function is retained for backward compatibility but should be replaced 
# with the LanguageManager class for new code. Translation handlers should use
# language_manager.get_tesseract_code() instead.
def get_lang_code_for_translation_api(lang_code):
    """Converts various language codes to standard ISO 639-1 for translation APIs."""
    code_map = {
        'eng': 'en', 'pol': 'pl', 'fra': 'fr', 'deu': 'de', 'spa': 'es',
        'ita': 'it', 'jpn': 'ja', 'kor': 'ko', 'chi_sim': 'zh-cn',
        'chi_tra': 'zh-tw', 'rus': 'ru', 'por': 'pt', 'nld': 'nl',
        'ara': 'ar', 'hin': 'hi', 'ukr': 'uk', 'ces': 'cs', 'dan': 'da',
        'fin': 'fi', 'swe': 'sv', 'nor': 'no', 'auto': 'auto'
    }
    if isinstance(lang_code, str):
        lang_code_lower = lang_code.lower()
        result = code_map.get(lang_code_lower, lang_code_lower)
        # log_debug(f"Converting language code '{lang_code}' to API code '{result}'") # Can be verbose
        return result
    return lang_code # Should be a string

def post_process_translation_text(text):
    if not text: return text
    
    # Fix spacing before punctuation marks
    text = re.sub(r'\s+\?', '?', text)
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    # PRESERVE dialog line breaks while cleaning up excessive spaces
    # Split by newlines, clean spaces within each line, then rejoin
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Clean up multiple spaces within each line, but preserve the line structure
        cleaned_line = re.sub(r'[ \t]{2,}', ' ', line)  # Only target spaces and tabs, not newlines
        cleaned_lines.append(cleaned_line)
    
    # Rejoin with preserved newlines
    text = '\n'.join(cleaned_lines)
    
    return text
