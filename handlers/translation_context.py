"""Session and bounded translation-context responsibilities."""

import sys

CUSTOM_CONTEXT_MAX_CHAR_BUDGET = 2400
CUSTOM_CONTEXT_MIN_CHAR_BUDGET = 600
CUSTOM_CONTEXT_SOURCE_PENALTY_CAP = 1800
CUSTOM_PROMPT_CACHE_EMA_ALPHA = 0.25
CUSTOM_PROMPT_CACHE_MIN_SAMPLES = 3
CUSTOM_PROMPT_CACHE_LOW_RATIO = 0.20
CUSTOM_PROMPT_CACHE_WEAK_RATIO = 0.45
CUSTOM_PROMPT_CACHE_LONG_INPUT_TOKENS = 3500
CUSTOM_PROMPT_CACHE_VERY_LONG_INPUT_TOKENS = 6000


def _log_debug(message):
    facade = sys.modules.get("handlers.translation_handler")
    if facade is not None:
        return facade.log_debug(message)


class TranslationContextMixin:
    def start_translation_session(self):
        provider = self._get_active_llm_provider()
        if provider:
            provider.start_translation_session()
        
        self._clear_active_context()

    def request_end_translation_session(self):        
        provider = self._get_active_llm_provider()
        if provider:
            result = provider.request_end_translation_session()
        else:
            result = True
        
        self._clear_active_context()
        
        return result

    # === CONTEXT MANAGEMENT ===
    def _clear_active_context(self):
        """Clear context window for the currently active LLM provider. Called when language, model, or settings change."""
        with self._custom_context_lock:
            self.custom_context_window = []
            self._custom_context_generation += 1
            self._custom_context_order_by_source.clear()
            self._custom_context_fallback_order = 0
        _log_debug("Custom AI context cleared")
        provider = self._get_active_llm_provider()
        if provider:
            provider._clear_context()
            _log_debug(f"{provider.provider_name.title()} context cleared via active provider")
        else:
            _log_debug("No active LLM provider found for context clearing")

    # === OCR SESSION MANAGEMENT ===
    def start_ocr_session(self):
        """Start OCR session for the active OCR provider."""
        provider = self._get_active_ocr_provider()
        if provider:
            provider.start_ocr_session()

    def request_end_ocr_session(self):
        """Request to end OCR session for the active OCR provider."""
        provider = self._get_active_ocr_provider()
        if provider:
            return provider.request_end_ocr_session()
        return True

    def force_end_sessions_on_app_close(self):
        # Context clearing is now handled automatically in base class after session end logging        
        # End translation sessions
        for provider in self.providers.values():
            try:
                provider.end_translation_session(force=True)
            except Exception as e:
                _log_debug(f"Error force ending {provider.provider_name} session: {e}")
        
        # End OCR sessions
        for provider in self.ocr_providers.values():
            try:
                provider.end_session(force=True)
            except Exception as e:
                _log_debug(f"Error force ending {provider.provider_name} OCR session: {e}")
        
        self._clear_active_context()

    def close(self):
        log_executor = None
        try:
            with self._custom_log_state_lock:
                log_executor = self._custom_log_executor
                self._custom_log_executor = None
            if log_executor is not None:
                log_executor.shutdown(wait=True, cancel_futures=False)
        except Exception as e:
            _log_debug(f"Error flushing Custom AI short log: {e}")
        try:
            if hasattr(self.unified_cache, "close"):
                self.unified_cache.close()
        except Exception as e:
            _log_debug(f"Error closing unified translation cache: {e}")
        try:
            if hasattr(self.custom_ai_provider, "close"):
                self.custom_ai_provider.close()
        except Exception as e:
            _log_debug(f"Error closing Custom AI provider: {e}")

    # === DEEPL CONTEXT MANAGEMENT ===
    def _clear_deepl_context(self):
        """Clear DeepL context window (called on language change or session end)."""
        try:
            self.deepl_context_window = []
            self.deepl_current_source_lang = None
            self.deepl_current_target_lang = None
            _log_debug("DeepL context cleared")
        except Exception as e:
            _log_debug(f"Error clearing DeepL context: {e}")
    
    def _build_deepl_context(self, context_size):
        """Build DeepL context string from previous source texts only.
        
        Args:
            context_size: Number of previous subtitles to include (0-3)
            
        Returns:
            Context string or None if no context available
        """
        # Validate context size
        if not isinstance(context_size, int):
            _log_debug(f"Invalid DeepL context size type: {type(context_size)}, using 0")
            return None
        
        if context_size < 0 or context_size > 3:
            _log_debug(f"Invalid DeepL context size: {context_size}, clamping to 0-3")
            context_size = max(0, min(3, context_size))
        
        if context_size == 0 or not self.deepl_context_window:
            return None
        
        # Get last N source texts
        context_texts = self.deepl_context_window[-context_size:]
        
        # Join subtitles with linebreaks
        context_string = "\n".join(context_texts)
        
        return context_string
    
    def _update_deepl_context(self, source_text):
        """Update DeepL context window with new source text.
        
        Args:
            source_text: New source subtitle to add to context
        """
        # Check for duplicate (same as last subtitle)
        if self.deepl_context_window and self.deepl_context_window[-1] == source_text:
            _log_debug(f"Skipping DeepL context update - duplicate source text")
            return
        
        # Add new source text
        self.deepl_context_window.append(source_text)
        
        # Keep only last 5 texts (more than max context setting for flexibility)
        self.deepl_context_window = self.deepl_context_window[-5:]
        _log_debug(f"DeepL context updated. Window size: {len(self.deepl_context_window)}")
    
    # === DEEPL LOGGING SYSTEM ===
    
    def _custom_context_entry_source(self, entry):
        if isinstance(entry, (tuple, list)) and entry:
            return str(entry[0])
        return str(entry)

    def _sync_custom_context_order_locked(self):
        current_sources = []
        for entry in self.custom_context_window:
            source = self._custom_context_entry_source(entry)
            current_sources.append(source)
            if source in self._custom_context_order_by_source:
                continue
            self._custom_context_fallback_order += 1
            self._custom_context_order_by_source[source] = (
                self._custom_context_fallback_order
            )
        current_source_set = set(current_sources)
        self._custom_context_order_by_source = {
            source: order
            for source, order in self._custom_context_order_by_source.items()
            if source in current_source_set
        }
        if self._custom_context_order_by_source:
            self._custom_context_fallback_order = max(
                self._custom_context_fallback_order,
                max(self._custom_context_order_by_source.values()),
            )

    def _next_custom_context_order_locked(self, translation_sequence):
        try:
            sequence_order = int(translation_sequence)
        except (TypeError, ValueError):
            sequence_order = None
        if sequence_order is not None:
            self._custom_context_fallback_order = max(
                self._custom_context_fallback_order,
                sequence_order,
            )
            return sequence_order
        self._custom_context_fallback_order += 1
        return self._custom_context_fallback_order

    def _update_custom_context(
        self,
        source_text,
        translated_text=None,
        translation_sequence=None,
        context_generation=None,
    ):
        context_size = self._get_custom_context_window_size()
        source_key = str(source_text)
        with self._custom_context_lock:
            if (
                context_generation is not None
                and context_generation != self._custom_context_generation
            ):
                _log_debug(
                    "Ignored stale Custom AI context update "
                    f"generation={context_generation} "
                    f"current={self._custom_context_generation}"
                )
                return
            if context_size == 0:
                self.custom_context_window = []
                self._custom_context_order_by_source.clear()
                self._custom_context_fallback_order = 0
                return

            self._sync_custom_context_order_locked()
            context_order = self._next_custom_context_order_locked(
                translation_sequence
            )
            existing_order = self._custom_context_order_by_source.get(
                source_key
            )
            if (
                existing_order is not None
                and translation_sequence is not None
                and context_order < existing_order
            ):
                return

            if translated_text is None:
                context_entry = source_text
            else:
                context_entry = (source_text, translated_text)

            refreshed_window = [
                entry
                for entry in self.custom_context_window
                if self._custom_context_entry_source(entry) != source_key
            ]
            refreshed_window.append(context_entry)
            self._custom_context_order_by_source[source_key] = context_order
            refreshed_window.sort(
                key=lambda entry: self._custom_context_order_by_source[
                    self._custom_context_entry_source(entry)
                ]
            )
            self.custom_context_window = refreshed_window[-context_size:]
            retained_sources = {
                self._custom_context_entry_source(entry)
                for entry in self.custom_context_window
            }
            self._custom_context_order_by_source = {
                source: order
                for source, order in self._custom_context_order_by_source.items()
                if source in retained_sources
            }

    def _get_custom_context_for_request(
        self,
        current_source=None,
        profile=None,
    ):
        context_size = self._get_custom_context_window_size()
        if context_size == 0:
            return []

        context_budget = self._get_custom_context_char_budget(
            current_source,
            profile=profile,
        )
        with self._custom_context_lock:
            context_window = list(self.custom_context_window)
        selected = []
        selected_chars = 0
        for entry in reversed(context_window):
            if (
                current_source is not None
                and isinstance(entry, (tuple, list))
                and entry
                and entry[0] == current_source
            ):
                continue
            if current_source is not None and entry == current_source:
                continue

            entry_chars = self._get_custom_context_entry_char_count(entry)
            remaining_chars = context_budget - selected_chars
            if entry_chars > remaining_chars:
                if not selected and remaining_chars > 0:
                    selected.append(
                        self._truncate_custom_context_entry(
                            entry,
                            remaining_chars,
                        )
                    )
                break

            selected.append(entry)
            selected_chars += entry_chars
            if len(selected) >= context_size:
                break
        return list(reversed(selected))

    def _get_custom_context_char_budget(
        self,
        current_source=None,
        profile=None,
    ):
        source_text = str(current_source or "")
        source_penalty = min(
            CUSTOM_CONTEXT_SOURCE_PENALTY_CAP,
            len(source_text),
        )
        base_budget = max(
            CUSTOM_CONTEXT_MIN_CHAR_BUDGET,
            CUSTOM_CONTEXT_MAX_CHAR_BUDGET - source_penalty,
        )
        budget_factor = self._get_custom_prompt_cache_budget_factor(
            profile=profile,
        )
        return max(
            CUSTOM_CONTEXT_MIN_CHAR_BUDGET,
            int(base_budget * budget_factor),
        )

    def _get_custom_prompt_cache_budget_factor(self, profile=None):
        with self._custom_route_state_lock:
            metrics = self._get_custom_prompt_cache_metrics(profile=profile)
            sample_count = metrics["sample_count"]
            cached_ratio_ema = metrics["cached_input_ratio_ema"]
            input_tokens_ema = metrics["input_tokens_ema"]

        if sample_count < CUSTOM_PROMPT_CACHE_MIN_SAMPLES:
            return 1.0

        factor = 1.0
        if input_tokens_ema is not None:
            if input_tokens_ema >= CUSTOM_PROMPT_CACHE_VERY_LONG_INPUT_TOKENS:
                factor = min(factor, 0.5)
            elif input_tokens_ema >= CUSTOM_PROMPT_CACHE_LONG_INPUT_TOKENS:
                factor = min(factor, 0.75)
        if cached_ratio_ema is not None:
            if cached_ratio_ema < CUSTOM_PROMPT_CACHE_LOW_RATIO:
                factor = min(factor, 0.5)
            elif cached_ratio_ema < CUSTOM_PROMPT_CACHE_WEAK_RATIO:
                factor = min(factor, 0.75)
        return factor

    def _get_custom_context_entry_char_count(self, entry):
        if isinstance(entry, (tuple, list)):
            return sum(len(str(part or "")) for part in entry[:2])
        return len(str(entry or ""))

    def _truncate_custom_context_entry(self, entry, max_chars):
        max_chars = max(0, int(max_chars))
        if isinstance(entry, (tuple, list)) and len(entry) >= 2:
            source = str(entry[0] or "")
            translation = str(entry[1] or "")
            source_budget = min(len(source), max_chars // 2)
            translation_budget = max_chars - source_budget
            if len(translation) < translation_budget:
                source_budget = min(
                    len(source),
                    max_chars - len(translation),
                )
                translation_budget = max_chars - source_budget
            return (
                source[:source_budget],
                translation[:translation_budget],
            )
        return str(entry or "")[:max_chars]

    def _get_custom_context_window_size(self):
        var = getattr(self.app, 'custom_context_window_var', None)
        try:
            value = int(var.get()) if var is not None else 5
        except (TypeError, ValueError):
            value = 5
        context_size = max(0, min(10, value))
        optimization_getter = getattr(
            self.app,
            "get_ai_optimization_mode",
            None,
        )
        if callable(optimization_getter):
            try:
                if str(optimization_getter() or "").strip().lower() == "speed":
                    return min(context_size, 1)
            except Exception:
                pass
        return context_size

    def _custom_usage_number(self, usage, *keys):
        if not isinstance(usage, dict):
            return 0.0
        for key in keys:
            value = usage.get(key)
            if value is None:
                continue
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                continue
        return 0.0

    def _custom_usage_cached_input_ratio(self, usage):
        prompt_tokens = self._custom_usage_number(
            usage,
            "input_tokens",
            "prompt_tokens",
        )
        cached_tokens = self._custom_usage_number(
            usage,
            "cached_input_tokens",
            "cached_prompt_tokens",
        )
        ratio_value = None
        if isinstance(usage, dict):
            try:
                ratio_value = float(usage.get("cached_input_ratio"))
            except (TypeError, ValueError):
                ratio_value = None
        if ratio_value is None:
            ratio_value = cached_tokens / prompt_tokens if prompt_tokens > 0 else 0.0
        return max(0.0, min(1.0, ratio_value)), prompt_tokens

    def _record_custom_prompt_cache_usage(
        self,
        call_type,
        usage,
        profile=None,
    ):
        cached_ratio, prompt_tokens = self._custom_usage_cached_input_ratio(usage)
        if call_type != "translation" or prompt_tokens <= 0:
            return cached_ratio

        alpha = CUSTOM_PROMPT_CACHE_EMA_ALPHA
        with self._custom_route_state_lock:
            metrics = self._get_custom_prompt_cache_metrics(profile=profile)
            if metrics["sample_count"] == 0:
                metrics["cached_input_ratio_ema"] = cached_ratio
                metrics["input_tokens_ema"] = prompt_tokens
            else:
                metrics["cached_input_ratio_ema"] = (
                    (1.0 - alpha) * metrics["cached_input_ratio_ema"]
                    + alpha * cached_ratio
                )
                metrics["input_tokens_ema"] = (
                    (1.0 - alpha) * metrics["input_tokens_ema"]
                    + alpha * prompt_tokens
                )
            metrics["sample_count"] += 1
            cached_ratio_ema = metrics["cached_input_ratio_ema"]
            input_tokens_ema = metrics["input_tokens_ema"]

        self._set_runtime_metric_gauge(
            "custom_ai_cached_input_ratio",
            cached_ratio,
        )
        self._set_runtime_metric_gauge(
            "custom_ai_cached_input_ratio_ema",
            cached_ratio_ema,
        )
        self._set_runtime_metric_gauge(
            "custom_ai_input_tokens_ema",
            input_tokens_ema,
        )
        return cached_ratio

    def _record_custom_ai_latency_observation(
        self,
        duration,
        success=True,
        profile=None,
    ):
        try:
            with self._custom_route_state_lock:
                advisor = self._get_custom_latency_advisor(profile=profile)
                observer = getattr(advisor, "observe_request", None)
                if not callable(observer):
                    return
                observer(duration, success=success)
        except Exception as observe_error:
            _log_debug(
                "Custom AI adaptive latency observation failed: "
                f"{type(observe_error).__name__} - {observe_error}"
            )

