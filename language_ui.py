import os
import csv
from logger import log_debug
from resource_handler import get_resource_path


LANGUAGE_DISPLAY_NAMES = {
    "eng": "English",
    "zh": "\u4e2d\u6587",
}

SUPPORTED_GUI_LANGUAGE_CODES = ("eng", "zh")

LEGACY_DISPLAY_NAME_ALIASES = {
    "\u6d93\ue15f\u6783": "\u4e2d\u6587",
    "Polish": "English",
    "polish": "English",
    "Polski": "English",
    "polski": "English",
    "POL": "English",
    "pol": "English",
}


def normalize_gui_language_display_name(display_name):
    """Return a supported GUI language display name, falling back to English."""
    normalized = LEGACY_DISPLAY_NAME_ALIASES.get(display_name, display_name)
    if normalized in LANGUAGE_DISPLAY_NAMES.values():
        return normalized
    return LANGUAGE_DISPLAY_NAMES["eng"]


def normalize_gui_language_code(lang_code):
    """Return a supported GUI language code, falling back to English."""
    if lang_code in SUPPORTED_GUI_LANGUAGE_CODES:
        return lang_code
    return "eng"


class UILanguageManager:
    def __init__(self, resources_dir="resources"):
        self.resources_dir = resources_dir
        self.default_lang = "eng"
        self.current_lang = self.default_lang
        self.labels = {}
        self.available_languages = self._scan_available_languages()
        self.load_language(self.default_lang)

    def _scan_available_languages(self):
        """Scan the resources directory for supported GUI language files."""
        found_codes = set()
        try:
            resources_path = get_resource_path(self.resources_dir)

            if os.path.exists(resources_path) and os.path.isdir(resources_path):
                filenames = os.listdir(resources_path)
            else:
                parent_dir = os.path.dirname(resources_path)
                filenames = os.listdir(parent_dir) if os.path.exists(parent_dir) and os.path.isdir(parent_dir) else []

            for filename in filenames:
                if filename.startswith("gui_") and filename.endswith(".csv"):
                    lang_code = filename[4:-4]
                    if lang_code in SUPPORTED_GUI_LANGUAGE_CODES:
                        found_codes.add(lang_code)

            languages = {
                code: LANGUAGE_DISPLAY_NAMES[code]
                for code in SUPPORTED_GUI_LANGUAGE_CODES
                if code in found_codes
            }
            if not languages:
                log_debug(f"No supported GUI language files found in resources directory: {resources_path}")
                languages = {"eng": LANGUAGE_DISPLAY_NAMES["eng"]}
            return languages
        except Exception as e:
            log_debug(f"Error scanning for GUI language files: {e}")
            return {"eng": LANGUAGE_DISPLAY_NAMES["eng"]}

    def get_available_languages(self):
        """Return a dictionary of available GUI languages {code: display_name}."""
        return self.available_languages

    def get_language_list(self):
        """Return GUI language display names for the dropdown."""
        return list(self.available_languages.values())

    def get_language_code_from_name(self, display_name):
        """Convert a display name back to a supported language code."""
        normalized_display_name = self.normalize_display_name(display_name)
        for code, name in self.available_languages.items():
            if name == normalized_display_name:
                return code
        return self.default_lang

    def normalize_display_name(self, display_name):
        """Normalize legacy or removed language names to supported display text."""
        return normalize_gui_language_display_name(display_name)

    def load_language(self, lang_code):
        """Load labels from the specified language CSV file."""
        lang_code = normalize_gui_language_code(lang_code or self.default_lang)

        self.current_lang = lang_code
        self.labels = {}

        try:
            lang_file_path = os.path.join(self.resources_dir, f"gui_{lang_code}.csv")
            lang_file = get_resource_path(lang_file_path)

            if os.path.exists(lang_file):
                with open(lang_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            key, value = row[0], row[1]
                            self.labels[key] = value
                log_debug(f"Loaded {len(self.labels)} UI labels from {lang_file}")

                sample_labels = list(self.labels.items())[:5]
                log_debug(f"Sample labels: {sample_labels}")
            else:
                log_debug(f"Language file not found: {lang_file}")
                if lang_code != self.default_lang:
                    log_debug(f"Falling back to default language: {self.default_lang}")
                    self.load_language(self.default_lang)
        except Exception as e:
            log_debug(f"Error loading language file: {e}")
            if lang_code != self.default_lang:
                log_debug(f"Falling back to default language: {self.default_lang}")
                self.load_language(self.default_lang)

    def get_label(self, key, default=None):
        """Get a UI label by key with fallback to default text."""
        if key in self.labels:
            return self.labels[key]

        if default:
            return default

        return key
