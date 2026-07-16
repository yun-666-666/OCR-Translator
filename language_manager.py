# language_manager.py
import os
import csv
from logger import log_debug
from resource_handler import get_resource_path
from constants import RTL_LANGUAGES


class LanguageManager:
    """Manages language lists and code mappings for Custom AI translation."""

    def __init__(self):
        self.custom_source_languages = []
        self.custom_target_languages = []
        self.custom_source_names = []
        self.custom_target_names = []

        self.language_display_names = []
        self.generic_lang_codes = {}

        self.load_language_lists()
        self.load_language_display_names()
        self.load_generic_lang_codes()

    def load_language_lists(self):
        try:
            source_path = get_resource_path("resources/google_trans_source.csv")
            target_path = get_resource_path("resources/google_trans_target.csv")
            self.custom_source_languages = self._load_language_csv(source_path, include_auto=True)
            self.custom_target_languages = self._load_language_csv(target_path, include_auto=False)
            self.custom_source_names = [name for name, _ in self.custom_source_languages]
            self.custom_target_names = [name for name, _ in self.custom_target_languages]
            log_debug(
                "Loaded Custom AI language lists: "
                f"Src({len(self.custom_source_names)}), Tgt({len(self.custom_target_names)})"
            )
        except Exception as e:
            log_debug(f"Error loading language lists: {e}")
            self.custom_source_languages = [("Auto", "auto"), ("English", "en"), ("Chinese (Simplified)", "zh-CN")]
            self.custom_target_languages = [("English", "en"), ("Chinese (Simplified)", "zh-CN")]
            self.custom_source_names = [name for name, _ in self.custom_source_languages]
            self.custom_target_names = [name for name, _ in self.custom_target_languages]

    def _load_language_csv(self, file_path, include_auto=False):
        languages = []
        if include_auto:
            languages.append(("Auto", "auto"))
        if not os.path.exists(file_path):
            log_debug(f"Language CSV not found: {file_path}")
            return languages
        with open(file_path, "r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                name = row[0].strip()
                code = row[1].strip()
                if not name or not code:
                    continue
                if name.lower() == "language" or code.lower() == "code":
                    continue
                languages.append((name, code))
        return languages

    def load_language_display_names(self):
        try:
            file_path = get_resource_path("resources/language_display_names.csv")
            self.language_display_names = []
            if not os.path.exists(file_path):
                log_debug(f"language_display_names.csv not found: {file_path}")
                return
            with open(file_path, "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    provider = (row.get("provider") or row.get("Provider") or "").strip().lower()
                    if provider and provider not in {"google", "custom_ai", "custom", "generic", ""}:
                        # Keep generic/google-style names; skip provider-specific legacy rows.
                        if provider in {"deepl", "gemini", "openai", "marianmt", "marian"}:
                            continue
                    english_name = (row.get("english_name") or row.get("English") or row.get("name") or "").strip()
                    polish_name = (row.get("polish_name") or row.get("Polish") or english_name).strip()
                    chinese_name = (row.get("chinese_name") or row.get("Chinese") or english_name).strip()
                    code = (row.get("code") or row.get("api_code") or "").strip()
                    if not english_name:
                        continue
                    self.language_display_names.append(
                        {
                            "provider": provider or "generic",
                            "english_name": english_name,
                            "polish_name": polish_name or english_name,
                            "chinese_name": chinese_name or english_name,
                            "code": code,
                        }
                    )
            log_debug(f"Loaded {len(self.language_display_names)} language display names")
        except Exception as e:
            log_debug(f"Error loading language display names: {e}")
            self.language_display_names = []

    def load_generic_lang_codes(self):
        try:
            file_path = get_resource_path("resources/lang_codes.csv")
            self.generic_lang_codes = {}
            if not os.path.exists(file_path):
                log_debug(f"lang_codes.csv not found: {file_path}")
                return
            with open(file_path, "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    language = (row.get("Language") or row.get("language") or "").strip()
                    code = (row.get("2-letter code") or row.get("code") or "").strip()
                    if language and code:
                        self.generic_lang_codes[language.lower()] = code
            log_debug(f"Loaded {len(self.generic_lang_codes)} generic language codes")
        except Exception as e:
            log_debug(f"Error loading generic language codes: {e}")
            self.generic_lang_codes = {}

    def get_language_lists(self, provider_name, lang_direction="source"):
        provider = str(provider_name or "").lower()
        if provider != "custom_ai":
            return []
        return (
            self.custom_source_languages
            if lang_direction == "source"
            else self.custom_target_languages
        )

    def get_language_names(self, provider_name, lang_direction="source"):
        languages = self.get_language_lists(provider_name, lang_direction)
        return [name for name, _ in languages]

    def get_code_from_name(self, display_name, provider_name=None, lang_direction="source"):
        if not display_name:
            return None
        languages = self.get_language_lists(provider_name, lang_direction)
        for name, code in languages:
            if name == display_name:
                return code
        return None

    def get_name_from_code(self, api_code, provider_name=None, lang_direction="source"):
        if not api_code:
            return None
        languages = self.get_language_lists(provider_name, lang_direction)
        for name, code in languages:
            if code == api_code:
                return name
        return None

    def get_localized_language_name(self, english_name, ui_language="english", provider_name=None):
        if not english_name:
            return english_name
        ui = str(ui_language or "english").lower()
        for entry in self.language_display_names:
            if entry["english_name"] == english_name:
                if ui in {"polish", "polski", "pol"}:
                    return entry.get("polish_name") or english_name
                if ui in {"chinese", "zh", "zh-cn", "zh_cn", "chs"}:
                    return entry.get("chinese_name") or english_name
                return entry.get("english_name") or english_name
        return english_name

    def get_code_from_localized_name(self, display_name, provider_name=None, ui_language="english"):
        if not display_name:
            return None
        # Direct match against active language list first.
        code = self.get_code_from_name(display_name, provider_name, "source")
        if code:
            return code
        code = self.get_code_from_name(display_name, provider_name, "target")
        if code:
            return code

        ui = str(ui_language or "english").lower()
        for entry in self.language_display_names:
            candidates = {
                entry.get("english_name"),
                entry.get("polish_name"),
                entry.get("chinese_name"),
            }
            if display_name in candidates:
                if entry.get("code"):
                    return entry["code"]
                # Fall back to matching english name against the language CSV.
                return (
                    self.get_code_from_name(entry.get("english_name"), provider_name, "source")
                    or self.get_code_from_name(entry.get("english_name"), provider_name, "target")
                )
        # Generic code table fallback.
        return self.generic_lang_codes.get(str(display_name).lower())

    def get_localized_names_for_provider(self, provider_name, lang_direction="source", ui_language="english"):
        languages = self.get_language_lists(provider_name, lang_direction)
        names = []
        for english_name, _code in languages:
            names.append(self.get_localized_language_name(english_name, ui_language, provider_name))
        return names

    def _polish_sort_key(self, value):
        polish_order = {
            "a": "01", "ą": "02", "b": "03", "c": "04", "ć": "05", "d": "06",
            "e": "07", "ę": "08", "f": "09", "g": "10", "h": "11", "i": "12",
            "j": "13", "k": "14", "l": "15", "ł": "16", "m": "17", "n": "18",
            "ń": "19", "o": "20", "ó": "21", "p": "22", "q": "23", "r": "24",
            "s": "25", "ś": "26", "t": "27", "u": "28", "v": "29", "w": "30",
            "x": "31", "y": "32", "z": "33", "ź": "34", "ż": "35",
        }
        result = []
        for char in str(value or "").lower():
            if char in polish_order:
                result.append(polish_order[char])
            elif "a" <= char <= "z":
                result.append(f"00{ord(char):02d}")
            else:
                result.append(f"99{ord(char):04d}")
        return "".join(result)

    def sort_polish_names(self, names_list):
        try:
            return sorted(names_list, key=self._polish_sort_key)
        except Exception as e:
            log_debug(f"Error in Polish sorting, falling back to default: {e}")
            return sorted(names_list)

    def is_rtl_language(self, language_code):
        if not language_code:
            return False
        normalized_code = language_code.lower().strip()
        if normalized_code in {"auto", "unknown"}:
            return False
        if normalized_code.startswith(("ar", "fa", "he")):
            return True
        return normalized_code in RTL_LANGUAGES

    def get_text_direction(self, language_code):
        return "rtl" if self.is_rtl_language(language_code) else "ltr"
