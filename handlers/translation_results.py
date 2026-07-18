"""Provider result formatting, logs, and archived legacy provider adapters."""

import sys
from datetime import datetime

from logger import (
    CUSTOM_AI_OCR_SHORT_LOG_FILENAME,
    CUSTOM_AI_TRANSLATION_SHORT_LOG_FILENAME,
    is_debug_logging_enabled,
    summarize_text_for_log,
)
from translation_utils import is_translation_error_result


def _log_debug(message):
    facade = sys.modules.get("handlers.translation_handler")
    if facade is not None:
        return facade.log_debug(message)


def _append_rotating_text(*args, **kwargs):
    facade = sys.modules.get("handlers.translation_handler")
    if facade is not None:
        return facade.append_rotating_text(*args, **kwargs)


class TranslationResultsMixin:
    def _initialize_deepl_log_file(self):
        """Archived DeepL log helper: no disk writes on the live path."""
        return None

    def _is_deepl_logging_enabled(self):
        """Archived DeepL logging toggle: always disabled."""
        return False

    def _log_deepl_translation_call(
        self,
        original_text,
        source_lang,
        target_lang,
        context_string,
        context_size,
        translated_text,
        model_type,
        call_start_time,
        call_duration,
    ):
        """Archived DeepL call logger: no side effects."""
        return None

    # === UNIFIED TRANSLATE METHOD ===
    def _log_custom_short_call(self, call_type, profile, result_text, usage, duration):
        try:
            # Metrics/latency bookkeeping always runs, even when disk logging is off.
            usage_missing = self._custom_usage_is_missing(usage)
            cost = self._custom_usage_optional_number(usage, "cost_usd")
            prompt_tokens = self._custom_usage_optional_number(
                usage,
                "input_tokens",
                "prompt_tokens",
            )
            completion_tokens = self._custom_usage_optional_number(
                usage,
                "output_tokens",
                "completion_tokens",
            )
            reasoning_tokens_line = ""
            if isinstance(usage, dict) and "reasoning_tokens" in usage:
                reasoning_tokens = self._custom_usage_optional_number(
                    usage,
                    "reasoning_tokens",
                )
                reasoning_tokens_line = (
                    f"Reasoning Tokens: {self._format_custom_usage_count(reasoning_tokens)}\n"
                )
            cached_prompt_tokens = self._custom_usage_optional_number(
                usage,
                "cached_input_tokens",
                "cached_prompt_tokens",
            )
            cached_input_ratio = self._record_custom_prompt_cache_usage(
                call_type,
                usage,
                profile=profile,
            )
            if call_type == "translation":
                self._record_custom_ai_latency_observation(
                    duration,
                    success=True,
                    profile=profile,
                )

            if not is_debug_logging_enabled():
                return

            with self._custom_log_state_lock:
                log_executor = self._custom_log_executor
                if log_executor is None:
                    _log_debug("Custom AI short log skipped after handler close")
                    return

                log_file = (
                    CUSTOM_AI_OCR_SHORT_LOG_FILENAME
                    if call_type == "ocr"
                    else CUSTOM_AI_TRANSLATION_SHORT_LOG_FILENAME
                )
                session_header = ""
                if call_type not in self._custom_session_started:
                    session_header = (
                        f"\nSESSION 1 STARTED "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
                    )
                header = (
                    "========= OCR CALL ==========="
                    if call_type == "ocr"
                    else "===== TRANSLATION CALL ======="
                )
                if self._is_custom_ai_log_content_enabled():
                    result_section = (
                        "Result:\n--------------------\n"
                        f"{result_text}\n"
                        "--------------------\n\n"
                    )
                else:
                    result_section = (
                        f"Result: {summarize_text_for_log(result_text)}\n\n"
                    )
                if usage_missing:
                    usage_status_line = "Usage: usage_missing\n"
                    cached_ratio_text = "n/a"
                else:
                    usage_status_line = ""
                    if cached_prompt_tokens is None and prompt_tokens is None:
                        cached_ratio_text = "n/a"
                    else:
                        cached_ratio_text = f"{cached_input_ratio:.2f}"
                block = (
                    f"{session_header}"
                    f"{header}\n"
                    f"Provider: {profile.get('name')}\n"
                    f"Model: {profile.get('model')}\n"
                    f"Duration: {duration:.3f}s\n"
                    f"{usage_status_line}"
                    f"Input Tokens: {self._format_custom_usage_count(prompt_tokens)}\n"
                    f"Cached Input Tokens: {self._format_custom_usage_count(cached_prompt_tokens)}\n"
                    f"cached_input_ratio={cached_ratio_text}\n"
                    f"Output Tokens: {self._format_custom_usage_count(completion_tokens)}\n"
                    f"{reasoning_tokens_line}"
                    f"Cost: {self._format_custom_usage_cost(cost)}\n"
                    f"{result_section}"
                )
                log_executor.submit(
                    self._write_custom_short_log,
                    log_file,
                    block,
                )
                self._custom_session_started.add(call_type)
        except Exception as e:
            _log_debug(f"Custom AI short log scheduling failed: {e}")

    def _legacy_translate_disabled(self):
        """Keep the removed legacy route inert for old callers."""
        return None

    # === ARCHIVED LEGACY PROVIDER ADAPTERS (inert) ===
    def _google_translate(self, text_to_translate_gt, source_lang_gt, target_lang_gt):
        """Archived Google Translate adapter: no key reads, no network calls."""
        return "Legacy Google Translate path is archived"

    def _deepl_translate(self, text_to_translate_dl, source_lang_dl, target_lang_dl):
        """Archived DeepL adapter: no client init, no key reads, no network calls."""
        return "Legacy DeepL path is archived"

    def get_deepl_usage(self):
        """Archived DeepL usage probe: always returns None without side effects."""
        return None

    # === UTILITY METHODS (UNCHANGED) ===
    def _is_error_message(self, text):
        return is_translation_error_result(text)

    def is_placeholder_text(self, text_content):
        if not text_content:
            return True
        text_lower = text_content.lower().strip()
        placeholders = [
            "source text will appear here",
            "translation will appear here",
            "translation...",
            "ocr source",
            "source text",
            "loading...",
            "translating...",
            "",
            "translation",
            "...",
            "translation error:",
        ]
        return text_lower in placeholders or text_lower.startswith("translation error:")

    def clear_cache(self):
        """Clear the unified translation cache."""
        self.unified_cache.clear_all()
        _log_debug("Cleared unified translation cache")

    def calculate_text_similarity(self, text1_sim, text2_sim):
        if not text1_sim or not text2_sim:
            return 0.0
        if len(text1_sim) < 10 or len(text2_sim) < 10:
            return 1.0 if text1_sim == text2_sim else 0.0

        words1_set = set(text1_sim.lower().split())
        words2_set = set(text2_sim.lower().split())
        intersection_len = len(words1_set.intersection(words2_set))
        union_len = len(words1_set.union(words2_set))
        return intersection_len / union_len if union_len > 0 else 0.0
