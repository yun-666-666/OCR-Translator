# handlers/translation_handler.py
import re
import os
import gc
import sys
import time
import html
import hashlib
import traceback
import threading
import concurrent.futures
from collections import OrderedDict
from datetime import datetime, timedelta
from urllib.parse import urlsplit, urlunsplit

from logger import (
    CUSTOM_AI_SHORT_LOG_BACKUP_COUNT,
    CUSTOM_AI_SHORT_LOG_MAX_BYTES,
    append_rotating_text,
    log_debug,
    log_debug_coalesced,
    summarize_text_for_log,
)
from unified_translation_cache import UnifiedTranslationCache
from translation_utils import is_translation_error_result
from custom_ai import (
    CustomAILatencyModeAdvisor,
    CustomAIProvider,
    CUSTOM_AI_LATENCY_MODE_ADAPTIVE,
    CUSTOM_AI_LATENCY_MODE_RACE,
    CUSTOM_AI_LATENCY_MODE_SAFE,
    CUSTOM_AI_LATENCY_MODE_STREAM,
    normalize_custom_ai_structured_output_mode,
    normalize_custom_ai_latency_mode,
    normalize_custom_ai_wire_api,
)

REQUESTS_AVAILABLE = False
CUSTOM_CONTEXT_MAX_CHAR_BUDGET = 2400
CUSTOM_CONTEXT_MIN_CHAR_BUDGET = 600
CUSTOM_CONTEXT_SOURCE_PENALTY_CAP = 1800
CUSTOM_PROMPT_CACHE_EMA_ALPHA = 0.25
CUSTOM_PROMPT_CACHE_MIN_SAMPLES = 3
CUSTOM_PROMPT_CACHE_LOW_RATIO = 0.20
CUSTOM_PROMPT_CACHE_WEAK_RATIO = 0.45
CUSTOM_PROMPT_CACHE_LONG_INPUT_TOKENS = 3500
CUSTOM_PROMPT_CACHE_VERY_LONG_INPUT_TOKENS = 6000
CUSTOM_AI_ROUTE_STATE_MAX_ENTRIES = 32
_CUSTOM_AI_PROFILE_UNSET = object()


from handlers.translation_context import TranslationContextMixin
from handlers.translation_requests import TranslationRequestsMixin
from handlers.translation_results import TranslationResultsMixin


class TranslationHandler(TranslationContextMixin, TranslationRequestsMixin, TranslationResultsMixin):
    def __init__(self, app):
        self.app = app
        self.unified_cache = UnifiedTranslationCache(
            max_size=1000,
            persistence_path=getattr(app, "custom_ai_translation_cache_file", None),
        )
        
        self.custom_ai_provider = CustomAIProvider()
        self.providers = {}
        self.ocr_providers = {}
        self._custom_context_lock = threading.RLock()
        self.custom_context_window = []
        self._custom_context_generation = 0
        self._custom_context_order_by_source = {}
        self._custom_context_fallback_order = 0
        self._custom_log_state_lock = threading.Lock()
        self._custom_log_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="CustomAIShortLog",
        )
        self._custom_session_started = set()
        self._custom_route_state_lock = threading.RLock()
        self._custom_latency_advisors = OrderedDict()
        self._custom_prompt_cache_metrics = OrderedDict()
        # Retain the legacy attribute as the sparse-profile compatibility
        # advisor. Real configured routes use their own bounded advisor.
        self._custom_latency_advisor = self._get_custom_latency_advisor()
        self._custom_race_state_lock = threading.Lock()
        self._custom_race_inflight_profiles = set()
        
        # Legacy DeepL-specific context storage retained only to keep old callbacks harmless.
        self.deepl_context_window = []  # List of source texts only
        self.deepl_current_source_lang = None
        self.deepl_current_target_lang = None
        
        log_debug("Translation handler initialized with custom AI provider and unified cache")

    def _is_custom_ai_log_content_enabled(self):
        """Return True only when the user explicitly opts into short-log bodies."""
        var = getattr(self.app, "custom_ai_log_content_enabled_var", None)
        if var is None:
            return False
        try:
            return bool(var.get())
        except Exception:
            return False

    def _write_custom_short_log(self, log_file, block):
        try:
            append_rotating_text(
                log_file,
                block,
                max_bytes=CUSTOM_AI_SHORT_LOG_MAX_BYTES,
                backup_count=CUSTOM_AI_SHORT_LOG_BACKUP_COUNT,
            )
        except Exception as e:
            log_debug(f"Custom AI short log write failed: {e}")

    def _format_dialog_text(self, text):
        """Format dialog text by adding line breaks before dashes that follow sentence-ending punctuation.
        
        This pre-processing ensures that dialog like:
        "- How are you? - Fine. - Great."
        
        becomes:
        "- How are you?
        - Fine.
        - Great."
        
        Args:
            text (str): The translation text to format
            
        Returns:
            str: The formatted text with proper dialog line breaks
        """
        if not text or not isinstance(text, str):
            return text
        
        # Check if the text starts with any dash (more robust - no space required)
        dash_check = (text.startswith("-") or text.startswith("–") or text.startswith("—"))
        if not dash_check:
            return text
        
        # Apply the formatting transformations
        formatted_text = text
        
        # New rule: Handle quoted dialogue format
        dialogue_patterns = ['"-', '" "', '- "', '" - "']
        has_dialogue_quotes = formatted_text.count('"') >= 4
        has_dialogue_pattern = any(pattern in formatted_text for pattern in dialogue_patterns)

        if has_dialogue_quotes and has_dialogue_pattern:
            # Check if there are occurrences of '"-'
            if '"-' in formatted_text:
                # Replace '"-' with '-'
                formatted_text = formatted_text.replace('"-', '-')
            # Check if there are occurrences of '- "' (dash + space + quote)
            elif '- "' in formatted_text:
                # Replace '- "' with '-'
                formatted_text = formatted_text.replace('- "', '-')
            else:
                # Replace odd occurrences of '"' with '-'
                result = []
                quote_count = 0
                for char in formatted_text:
                    if char == '"':
                        quote_count += 1
                        if quote_count % 2 == 1:  # Odd occurrence (1st, 3rd, 5th, etc.)
                            result.append('-')
                        else:  # Even occurrence (2nd, 4th, 6th, etc.)
                            result.append('"')
                    else:
                        result.append(char)
                formatted_text = ''.join(result)
            
            # Remove all remaining quotes
            formatted_text = formatted_text.replace('"', '')

        # Replace ". -" with ".\n-" (period + space + hyphen)
        formatted_text = formatted_text.replace(". -", ".\n-")
        formatted_text = formatted_text.replace(". –", ".\n–")
        formatted_text = formatted_text.replace(". —", ".\n—")
        
        # Replace "? -" with "?\n-" (question mark + space + hyphen)
        formatted_text = formatted_text.replace("? -", "?\n-")
        formatted_text = formatted_text.replace("? –", "?\n–")
        formatted_text = formatted_text.replace("? —", "?\n—")
        
        # Replace "! -" with "!\n-" (exclamation mark + space + hyphen)
        formatted_text = formatted_text.replace("! -", "!\n-")
        formatted_text = formatted_text.replace("! –", "?\n–")
        formatted_text = formatted_text.replace("! —", "!\n—")
        
        if formatted_text != text:
            log_debug_coalesced(
                "translation-dialog-format-applied",
                "Dialog formatting applied "
                f"input {summarize_text_for_log(text)} "
                f"output {summarize_text_for_log(formatted_text)}",
                interval_seconds=5.0,
            )
        
        return formatted_text
