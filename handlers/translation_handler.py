# handlers/translation_handler.py
import re
import os
import gc
import sys
import time
import html
import traceback
import threading
import concurrent.futures
from datetime import datetime, timedelta

from logger import append_rotating_text, log_debug
from unified_translation_cache import UnifiedTranslationCache
from translation_utils import is_translation_error_result
from custom_ai import (
    CustomAIProvider,
    CUSTOM_AI_LATENCY_MODE_RACE,
    CUSTOM_AI_LATENCY_MODE_SAFE,
    CUSTOM_AI_LATENCY_MODE_STREAM,
    normalize_custom_ai_latency_mode,
    normalize_custom_ai_wire_api,
)

REQUESTS_AVAILABLE = False
CUSTOM_CONTEXT_CHAR_BUDGET = 4000


class TranslationHandler:
    def __init__(self, app):
        self.app = app
        self.unified_cache = UnifiedTranslationCache(
            max_size=1000,
            persistence_path=getattr(app, "custom_ai_translation_cache_file", None),
        )
        
        self.custom_ai_provider = CustomAIProvider()
        self.providers = {}
        self.ocr_providers = {}
        self.custom_context_window = []
        self._custom_log_state_lock = threading.Lock()
        self._custom_log_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="CustomAIShortLog",
        )
        self._custom_session_started = set()
        
        # Legacy DeepL-specific context storage retained only to keep old callbacks harmless.
        self.deepl_context_window = []  # List of source texts only
        self.deepl_current_source_lang = None
        self.deepl_current_target_lang = None
        
        log_debug("Translation handler initialized with custom AI provider and unified cache")

    def _get_active_llm_provider(self):
        """Get the currently active LLM provider based on selected translation model."""
        return None

    def _get_active_ocr_provider(self):
        """Get the currently active OCR provider based on selected OCR model."""
        return None

    def perform_ocr(self, image_data, source_lang):
        """Main public method for performing OCR. Delegates to the currently selected API provider."""
        profile = self.app.custom_ai_profiles.get_active_profile("ocr")
        if not profile:
            log_debug("No active custom AI model profile configured for OCR")
            return "<e>: AI model profile for OCR is missing"
        try:
            result, usage, duration = self.custom_ai_provider.recognize(
                profile,
                image_data,
                source_lang,
                keep_linebreaks=self.app.keep_linebreaks_var.get(),
                latency_mode=self._get_custom_ai_latency_mode(),
            )
            self._log_custom_short_call("ocr", profile, result, usage, duration)
            return result
        except Exception as e:
            error_text = str(e)
            if hasattr(self.custom_ai_provider, "_sanitize_error"):
                error_text = self.custom_ai_provider._sanitize_error(error_text, profile.get("api_key", ""))
            log_debug(f"Custom AI OCR error: {type(e).__name__} - {error_text}")
            return f"<e>: Custom AI OCR error: {type(e).__name__} - {error_text}"

    # === LLM SESSION MANAGEMENT ===
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
        self.custom_context_window = []
        log_debug("Custom AI context cleared")
        provider = self._get_active_llm_provider()
        if provider:
            provider._clear_context()
            log_debug(f"{provider.provider_name.title()} context cleared via active provider")
        else:
            log_debug("No active LLM provider found for context clearing")

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
                log_debug(f"Error force ending {provider.provider_name} session: {e}")
        
        # End OCR sessions
        for provider in self.ocr_providers.values():
            try:
                provider.end_session(force=True)
            except Exception as e:
                log_debug(f"Error force ending {provider.provider_name} OCR session: {e}")
        
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
            log_debug(f"Error flushing Custom AI short log: {e}")
        try:
            if hasattr(self.unified_cache, "close"):
                self.unified_cache.close()
        except Exception as e:
            log_debug(f"Error closing unified translation cache: {e}")
        try:
            if hasattr(self.custom_ai_provider, "close"):
                self.custom_ai_provider.close()
        except Exception as e:
            log_debug(f"Error closing Custom AI provider: {e}")

    # === DEEPL CONTEXT MANAGEMENT ===
    def _clear_deepl_context(self):
        """Clear DeepL context window (called on language change or session end)."""
        try:
            self.deepl_context_window = []
            self.deepl_current_source_lang = None
            self.deepl_current_target_lang = None
            log_debug("DeepL context cleared")
        except Exception as e:
            log_debug(f"Error clearing DeepL context: {e}")
    
    def _build_deepl_context(self, context_size):
        """Build DeepL context string from previous source texts only.
        
        Args:
            context_size: Number of previous subtitles to include (0-3)
            
        Returns:
            Context string or None if no context available
        """
        # Validate context size
        if not isinstance(context_size, int):
            log_debug(f"Invalid DeepL context size type: {type(context_size)}, using 0")
            return None
        
        if context_size < 0 or context_size > 3:
            log_debug(f"Invalid DeepL context size: {context_size}, clamping to 0-3")
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
            log_debug(f"Skipping DeepL context update - duplicate source text")
            return
        
        # Add new source text
        self.deepl_context_window.append(source_text)
        
        # Keep only last 5 texts (more than max context setting for flexibility)
        self.deepl_context_window = self.deepl_context_window[-5:]
        log_debug(f"DeepL context updated. Window size: {len(self.deepl_context_window)}")
    
    # === DEEPL LOGGING SYSTEM ===
    
    def _initialize_deepl_log_file(self):
        """Initialize DeepL translation log file with header if it doesn't exist."""
        try:
            if not os.path.exists(self.deepl_log_file):
                with open(self.deepl_log_file, 'w', encoding='utf-8-sig') as f:
                    f.write("=== DEEPL TRANSLATION API CALL LOG ===\n")
                    f.write(f"Log initialized: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 50 + "\n\n")
                log_debug(f"DeepL translation log file initialized: {self.deepl_log_file}")
        except Exception as e:
            log_debug(f"Error initializing DeepL log file: {e}")
    
    def _is_deepl_logging_enabled(self):
        """Check if DeepL API logging is enabled in settings."""
        # For now, always return True. You can add a setting later if needed.
        return True
    
    def _log_deepl_translation_call(self, original_text, source_lang, target_lang, 
                                   context_string, context_size, translated_text, model_type, 
                                   call_start_time, call_duration):
        """Log DeepL translation API call with context information."""
        if not self._is_deepl_logging_enabled():
            return
        
        try:
            with self.deepl_log_lock:
                # Format timestamps
                start_timestamp = call_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                end_time = call_start_time + timedelta(seconds=call_duration)
                end_timestamp = end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                
                # Convert model type to display name
                model_display = "Next-gen" if model_type == "quality_optimized" else "Classic"
                
                # Build the log entry
                log_entry = f"""=== DEEPL TRANSLATION API CALL ===
Timestamp: {start_timestamp}
Language Pair: {source_lang} -> {target_lang}
Original Text: {original_text}

MESSAGE SENT TO DEEPL:
"""
                
                # Add context section if context was used
                if context_string:
                    subtitle_count = min(context_size, len(self.deepl_context_window))
                    count_text = f" ({subtitle_count} subtitle{'s' if subtitle_count != 1 else ''})" if context_size > 0 and subtitle_count > 0 else ""
                    log_entry += f"""
CONTEXT{count_text}:
{context_string}
"""
                # Add text to translate
                log_entry += f"""
TEXT TO TRANSLATE:
{original_text}

RESPONSE RECEIVED:
Model: {model_display}
Timestamp: {end_timestamp}
Call Duration: {call_duration:.3f} seconds

---BEGIN RESPONSE---
{translated_text}
---END RESPONSE---

========================================

"""
                
                # Write to log file
                with open(self.deepl_log_file, 'a', encoding='utf-8-sig') as f:
                    f.write(log_entry)
                
                log_debug(f"DeepL translation call logged: {source_lang}->{target_lang}, Duration={call_duration:.3f}s")
        
        except Exception as e:
            log_debug(f"Error logging DeepL translation call: {e}")

    # === UNIFIED TRANSLATE METHOD ===
    def translate_text_with_timeout(
        self,
        text_content,
        timeout_seconds=10.0,
        ocr_batch_number=None,
        stream_callback=None,
        translation_sequence=None,
    ):
        # Translation already runs inside the background translation worker pool.
        # Avoid spawning an extra daemon thread here, otherwise the outer worker
        # may release in-flight/concurrency state before the real HTTP request ends.
        try:
            return self.translate_text(
                text_content,
                ocr_batch_number,
                stream_callback=stream_callback,
                translation_sequence=translation_sequence,
            )
        except Exception as e:
            log_debug(f"Translation exception: {e}")
            return f"Translation error: {str(e)}"

    def translate_text(self, text_content_main, ocr_batch_number=None, stream_callback=None, translation_sequence=None):
        cleaned_text_main = text_content_main.strip() if text_content_main else ""
        if not cleaned_text_main or self.is_placeholder_text(cleaned_text_main):
            return None

        translation_start_monotonic = time.monotonic()
        selected_model = self.app.translation_model_var.get()
        log_debug(f"Translate request for \"{cleaned_text_main}\" using {selected_model}")

        if selected_model != 'custom_ai':
            log_debug(f"Legacy translation model '{selected_model}' is disabled; using custom_ai route")
            selected_model = 'custom_ai'

        return self._custom_ai_translate(
            cleaned_text_main,
            translation_start_monotonic,
            stream_callback=stream_callback,
            translation_sequence=translation_sequence,
        )

    def get_cached_translation_for_display(self, text_content):
        """Return a display-ready cached translation without making a provider call."""
        cleaned_text = text_content.strip() if text_content else ""
        if not cleaned_text or self.is_placeholder_text(cleaned_text):
            return None

        selected_model = self.app.translation_model_var.get()
        if selected_model != 'custom_ai':
            selected_model = 'custom_ai'

        if selected_model == 'custom_ai':
            return self._get_custom_ai_cached_translation(cleaned_text)

        return None

    def _get_custom_ai_cache_profile_and_params(self, current_source=None):
        profile = self.app.custom_ai_profiles.get_active_profile("translation")
        if not profile:
            return None, None, None, None

        source_lang = getattr(self.app, 'custom_source_lang', None) or self.app.source_lang_var.get()
        target_lang = getattr(self.app, 'custom_target_lang', None) or self.app.target_lang_var.get()
        cache_params = self._cache_params_for_profile(
            profile,
            current_source=current_source,
        )
        return profile, source_lang, target_lang, cache_params

    def _get_custom_ai_cached_translation(self, cleaned_text):
        profile, source_lang, target_lang, cache_params = self._get_custom_ai_cache_profile_and_params(
            current_source=cleaned_text,
        )
        if not profile:
            return None

        cached_result = self.unified_cache.get(cleaned_text, source_lang, target_lang, "custom_ai", **cache_params)
        if not cached_result:
            return None

        self._update_custom_context(cleaned_text, cached_result)
        return self._format_dialog_text(cached_result)

    def get_inflight_translation_key(self, text_content):
        cleaned_text = text_content.strip() if text_content else ""
        if not cleaned_text or self.is_placeholder_text(cleaned_text):
            return None

        profile, source_lang, target_lang, cache_params = self._get_custom_ai_cache_profile_and_params(
            current_source=cleaned_text,
        )
        if not profile:
            return ("custom_ai", cleaned_text, "missing_profile")

        return (
            "custom_ai",
            cleaned_text,
            source_lang,
            target_lang,
            cache_params.get("profile_id", ""),
            cache_params.get("base_url", ""),
            cache_params.get("model", ""),
            cache_params.get("wire_api", "chat_completions"),
            cache_params.get("reasoning_effort", ""),
            cache_params.get("custom_prompt", ""),
            cache_params.get("keep_linebreaks", False),
            cache_params.get("context", ()),
        )

    def _get_custom_ai_latency_mode(self):
        if hasattr(self.app, 'get_custom_ai_latency_mode'):
            return self.app.get_custom_ai_latency_mode()
        var = getattr(self.app, 'custom_ai_latency_mode_var', None)
        try:
            return normalize_custom_ai_latency_mode(var.get() if var is not None else CUSTOM_AI_LATENCY_MODE_SAFE)
        except Exception:
            return CUSTOM_AI_LATENCY_MODE_SAFE

    def get_translation_concurrency_limit(self):
        selected_model = self.app.translation_model_var.get()
        if selected_model != 'custom_ai':
            return max(1, int(getattr(self.app, "max_concurrent_translation_calls", 6)))
        return 1

    def get_translation_submit_interval_seconds(self, text_content=None):
        interval_var = getattr(self.app, "custom_ai_submit_interval_ms_var", None)
        try:
            if interval_var is not None:
                interval_ms = int(interval_var.get())
                base_interval = max(0, min(5000, interval_ms)) / 1000.0
            else:
                base_interval = max(
                    0.0,
                    float(getattr(self.app, "min_translation_interval", 0.3)),
                )
        except (TypeError, ValueError):
            base_interval = 0.3
        selected_model = self.app.translation_model_var.get()
        if selected_model != 'custom_ai':
            return max(0.0, base_interval)

        cleaned_text = str(text_content or "").replace("<br>", "\n").strip()
        if not cleaned_text:
            return max(base_interval, 1.0)

        # Large OCR payloads consume more tokens and are more likely to hit relay limits,
        # so pace long-form submissions more conservatively than subtitle-sized text.
        scaled_interval = base_interval
        if len(cleaned_text) > 80:
            scaled_interval += min(3.0, (len(cleaned_text) - 80) / 180.0)

        return scaled_interval

    def get_translation_provider_cooldown_seconds(self):
        selected_model = self.app.translation_model_var.get()
        if selected_model != 'custom_ai':
            return 0.0

        profile = self.app.custom_ai_profiles.get_active_profile("translation")
        if not profile:
            return 0.0

        cooldown_getter = getattr(self.custom_ai_provider, "get_cooldown_remaining", None)
        if not callable(cooldown_getter):
            return 0.0

        try:
            profiles = [profile]
            if self._get_custom_ai_latency_mode() == CUSTOM_AI_LATENCY_MODE_RACE:
                profiles = self._get_custom_ai_race_profiles(profile)
            remaining_values = [
                max(0.0, float(cooldown_getter(candidate)))
                for candidate in profiles
            ]
            return min(remaining_values) if remaining_values else 0.0
        except Exception as cooldown_error:
            log_debug(
                "Custom AI cooldown check failed: "
                f"{type(cooldown_error).__name__} - {cooldown_error}"
            )
            return 0.0

    def _custom_ai_translate(
        self,
        cleaned_text_main,
        translation_start_monotonic,
        stream_callback=None,
        translation_sequence=None,
    ):
        profile, source_lang, target_lang, cache_params = self._get_custom_ai_cache_profile_and_params(
            current_source=cleaned_text_main,
        )
        if not profile:
            return "AI model profile for translation is missing."

        cached_result = self._get_custom_ai_cached_translation(cleaned_text_main)
        if cached_result:
            return cached_result

        latency_mode = self._get_custom_ai_latency_mode()
        context = self._get_custom_context_for_request(
            current_source=cleaned_text_main,
        )
        keep_linebreaks = self.app.keep_linebreaks_var.get()

        try:
            if latency_mode == CUSTOM_AI_LATENCY_MODE_RACE:
                (
                    translated_api_text,
                    usage,
                    duration,
                    cache_params,
                    winning_profile,
                ) = self._custom_ai_translate_race(
                    profile,
                    cleaned_text_main,
                    source_lang,
                    target_lang,
                    context,
                    keep_linebreaks,
                )
            else:
                translated_api_text, usage, duration = self.custom_ai_provider.translate(
                    profile,
                    cleaned_text_main,
                    source_lang,
                    target_lang,
                    custom_prompt=getattr(self.app, 'custom_prompt_text', ''),
                    context=context,
                    keep_linebreaks=keep_linebreaks,
                    latency_mode=latency_mode,
                    stream_callback=stream_callback if latency_mode == CUSTOM_AI_LATENCY_MODE_STREAM else None,
                )
                winning_profile = profile
            self._log_custom_short_call("translation", winning_profile, translated_api_text, usage, duration)
        except Exception as e:
            log_debug(f"Custom AI translation error: {type(e).__name__} - {e}")
            return f"Custom AI translation error: {type(e).__name__} - {e}"

        if translated_api_text and not self._is_error_message(translated_api_text):
            cache_targets = [cache_params]
            if latency_mode == CUSTOM_AI_LATENCY_MODE_RACE:
                active_cache_params = self._cache_params_for_profile(
                    profile,
                    current_source=cleaned_text_main,
                )
                if active_cache_params != cache_params:
                    cache_targets.append(active_cache_params)
            for target_cache_params in cache_targets:
                self.unified_cache.store(
                    cleaned_text_main,
                    source_lang,
                    target_lang,
                    "custom_ai",
                    translated_api_text,
                    **target_cache_params,
                )
            self._update_custom_context(cleaned_text_main, translated_api_text)

        log_debug(f"Custom AI translation \"{cleaned_text_main}\" -> \"{str(translated_api_text)}\" took {time.monotonic() - translation_start_monotonic:.3f}s")
        return self._format_dialog_text(translated_api_text)

    def _custom_ai_translate_race(self, active_profile, text, source_lang, target_lang, context, keep_linebreaks):
        candidates = self._get_custom_ai_race_profiles(active_profile)
        if len(candidates) <= 1:
            candidate = candidates[0] if candidates else active_profile
            translated, usage, duration = self.custom_ai_provider.translate(
                candidate,
                text,
                source_lang,
                target_lang,
                custom_prompt=getattr(self.app, 'custom_prompt_text', ''),
                context=context,
                keep_linebreaks=keep_linebreaks,
                latency_mode=CUSTOM_AI_LATENCY_MODE_RACE,
            )
            return (
                translated,
                usage,
                duration,
                self._cache_params_for_profile(
                    candidate,
                    current_source=text,
                ),
                candidate,
            )

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(candidates),
            thread_name_prefix="CustomAIRace",
        )
        future_to_profile = {
            executor.submit(
                self.custom_ai_provider.translate,
                candidate,
                text,
                source_lang,
                target_lang,
                custom_prompt=getattr(self.app, 'custom_prompt_text', ''),
                context=context,
                keep_linebreaks=keep_linebreaks,
                latency_mode=CUSTOM_AI_LATENCY_MODE_RACE,
            ): candidate
            for candidate in candidates
        }
        shutdown_started = False
        errors = []
        try:
            for future in concurrent.futures.as_completed(future_to_profile):
                candidate = future_to_profile[future]
                try:
                    translated, usage, duration = future.result()
                except Exception as e:
                    errors.append(f"{candidate.get('name', 'Custom AI')}: {e}")
                    continue
                log_debug(
                    "LATENCY: custom_ai race winner "
                    f"profile={candidate.get('name', 'Custom AI')} "
                    f"duration={duration:.3f}s candidates={len(candidates)}"
                )
                executor.shutdown(wait=False, cancel_futures=True)
                shutdown_started = True
                return (
                    translated,
                    usage,
                    duration,
                    self._cache_params_for_profile(
                        candidate,
                        current_source=text,
                    ),
                    candidate,
                )
        finally:
            if not shutdown_started:
                executor.shutdown(wait=False, cancel_futures=True)

        raise ValueError("All Custom AI race endpoints failed. Tried: " + "; ".join(errors))

    def _get_custom_ai_race_profiles(self, active_profile):
        active_signature = self._custom_ai_race_signature(active_profile)
        candidates = []
        seen = set()

        def add_candidate(profile):
            if not isinstance(profile, dict):
                return
            if self._custom_ai_race_signature(profile) != active_signature:
                return
            identity = profile.get("id") or (
                profile.get("base_url", ""),
                profile.get("model", ""),
                normalize_custom_ai_wire_api(profile.get("wire_api")),
                str(
                    profile.get("reasoning_effort")
                    or profile.get("model_reasoning_effort")
                    or ""
                ).strip().lower(),
            )
            if identity in seen:
                return
            seen.add(identity)
            candidates.append(profile)

        add_candidate(active_profile)
        try:
            profiles = self.app.custom_ai_profiles.list_profiles("translation", enabled_only=True)
        except TypeError:
            profiles = self.app.custom_ai_profiles.list_profiles(enabled_only=True)
        except Exception as e:
            log_debug(f"Custom AI race profile list failed: {e}")
            profiles = []
        for profile in profiles:
            add_candidate(profile)

        cooldown_getter = getattr(
            self.custom_ai_provider,
            "get_cooldown_remaining",
            None,
        )
        if not callable(cooldown_getter):
            return candidates

        healthy_candidates = []
        for candidate in candidates:
            try:
                remaining = max(0.0, float(cooldown_getter(candidate)))
            except Exception as cooldown_error:
                log_debug(
                    "Custom AI race cooldown check failed: "
                    f"{type(cooldown_error).__name__} - {cooldown_error}"
                )
                remaining = 0.0
            if remaining <= 0.0:
                healthy_candidates.append(candidate)
        return healthy_candidates or candidates

    def _custom_ai_race_signature(self, profile):
        profile = profile if isinstance(profile, dict) else {}
        return (
            str(profile.get("model") or "").strip(),
            normalize_custom_ai_wire_api(profile.get("wire_api")),
            str(
                profile.get("reasoning_effort")
                or profile.get("model_reasoning_effort")
                or ""
            ).strip().lower(),
        )

    def _cache_params_for_profile(self, profile, current_source=None):
        keep_linebreaks_var = getattr(self.app, "keep_linebreaks_var", None)
        try:
            keep_linebreaks = bool(keep_linebreaks_var.get()) if keep_linebreaks_var is not None else False
        except Exception:
            keep_linebreaks = False

        return {
            "profile_id": profile.get("id", ""),
            "base_url": str(profile.get("base_url", "")).strip().rstrip("/"),
            "model": profile.get("model", ""),
            "wire_api": str(
                profile.get("wire_api") or "chat_completions"
            ).strip().lower(),
            "reasoning_effort": str(
                profile.get("reasoning_effort")
                or profile.get("model_reasoning_effort")
                or ""
            ).strip().lower(),
            "custom_prompt": getattr(self.app, "custom_prompt_text", ""),
            "keep_linebreaks": keep_linebreaks,
            "context": tuple(
                self._get_custom_context_for_request(
                    current_source=current_source,
                )
            ),
        }

    def _update_custom_context(self, source_text, translated_text=None):
        context_size = self._get_custom_context_window_size()
        if context_size == 0:
            self.custom_context_window = []
            return

        if translated_text is None:
            context_entry = source_text
        else:
            context_entry = (source_text, translated_text)

        refreshed_window = []
        for entry in self.custom_context_window:
            if isinstance(entry, (tuple, list)) and entry:
                if entry[0] == source_text:
                    continue
            elif entry == source_text:
                continue
            refreshed_window.append(entry)

        refreshed_window.append(context_entry)
        self.custom_context_window = refreshed_window[-context_size:]

    def _get_custom_context_for_request(self, current_source=None):
        context_size = self._get_custom_context_window_size()
        if context_size == 0:
            return []

        selected = []
        selected_chars = 0
        for entry in reversed(self.custom_context_window):
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
            remaining_chars = CUSTOM_CONTEXT_CHAR_BUDGET - selected_chars
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
        return max(0, min(10, value))

    def _write_custom_short_log(self, log_file, block):
        try:
            append_rotating_text(
                log_file,
                block,
                max_bytes=2 * 1024 * 1024,
                backup_count=2,
            )
        except Exception as e:
            log_debug(f"Custom AI short log write failed: {e}")

    def _log_custom_short_call(self, call_type, profile, result_text, usage, duration):
        try:
            with self._custom_log_state_lock:
                log_executor = self._custom_log_executor
                if log_executor is None:
                    log_debug("Custom AI short log skipped after handler close")
                    return

                log_file = (
                    "CustomAI_OCR_Short_Log.txt"
                    if call_type == "ocr"
                    else "CustomAI_Translation_Short_Log.txt"
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
                cost = 0.0
                prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
                completion_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
                cached_prompt_tokens = usage.get("cached_prompt_tokens", 0) if isinstance(usage, dict) else 0
                block = (
                    f"{session_header}"
                    f"{header}\n"
                    f"Provider: {profile.get('name')}\n"
                    f"Model: {profile.get('model')}\n"
                    f"Duration: {duration:.3f}s\n"
                    f"Input Tokens: {prompt_tokens}\n"
                    f"Cached Input Tokens: {cached_prompt_tokens}\n"
                    f"Output Tokens: {completion_tokens}\n"
                    f"Cost: ${cost:.8f}\n"
                    f"Result:\n--------------------\n{result_text}\n--------------------\n\n"
                )
                log_executor.submit(
                    self._write_custom_short_log,
                    log_file,
                    block,
                )
                self._custom_session_started.add(call_type)
        except Exception as e:
            log_debug(f"Custom AI short log scheduling failed: {e}")

    def _legacy_translate_disabled(self):
        """Legacy provider code is kept below for reference but is no longer reached."""
        return None
        
        source_lang, target_lang, extra_params = None, None, {}
        
        # Setup provider-specific parameters
        if selected_model == 'marianmt':
            source_lang, target_lang = self.app.marian_source_lang, self.app.marian_target_lang
            extra_params = {"beam_size": self.app.num_beams_var.get()}
        elif selected_model == 'google_api':
            source_lang, target_lang = self.app.google_source_lang, self.app.google_target_lang
        elif selected_model == 'deepl_api':
            if not self.app.DEEPL_API_AVAILABLE: return "DeepL API libraries not available."
            if not self.app.deepl_api_key_var.get().strip(): return "DeepL API key missing."
            if self.app.deepl_api_client is None:
                try:
                    import deepl
                    self.app.deepl_api_client = deepl.Translator(self.app.deepl_api_key_var.get().strip())
                except Exception as e:
                    return f"DeepL Client init error: {e}"
            source_lang, target_lang = self.app.deepl_source_lang, self.app.deepl_target_lang
            extra_params = {"model_type": self.app.deepl_model_type_var.get()}
        elif selected_model == 'gemini_api':
            source_lang, target_lang = self.app.gemini_source_lang, self.app.gemini_target_lang
            provider = self.providers['gemini']
            extra_params = {"context_window": provider._get_context_window_size()}
        elif self.app.is_openai_model(selected_model):
            source_lang, target_lang = self.app.openai_source_lang, self.app.openai_target_lang
            provider = self.providers['openai']
            extra_params = {"context_window": provider._get_context_window_size()}
        else:
            return f"Error: Unknown translation model '{selected_model}'"

        # 1. Check Unified Cache (In-Memory LRU)
        cached_result = self.unified_cache.get(cleaned_text_main, source_lang, target_lang, selected_model, **extra_params)
        if cached_result:
            log_debug(f"Translation \"{cleaned_text_main}\" -> \"{cached_result}\" from unified cache")
            
            # Check if file cache is enabled and save LRU result to file cache if not already there
            file_cache_enabled = False
            cache_key_for_file = None
            
            if selected_model == 'gemini_api' and self.app.gemini_file_cache_var.get():
                file_cache_enabled = True
                cache_key_for_file = f"gemini:{source_lang}:{target_lang}:{cleaned_text_main}"
            elif selected_model == 'google_api' and self.app.google_file_cache_var.get():
                file_cache_enabled = True
                cache_key_for_file = f"google:{source_lang}:{target_lang}:{cleaned_text_main}"
            elif selected_model == 'deepl_api' and self.app.deepl_file_cache_var.get():
                file_cache_enabled = True
                model_type = extra_params.get('model_type', 'latency_optimized')
                cache_key_for_file = f"deepl:{source_lang}:{target_lang}:{model_type}:{cleaned_text_main}"
            elif self.app.is_openai_model(selected_model) and self.app.openai_file_cache_var.get():
                file_cache_enabled = True
                cache_key_for_file = f"openai:{source_lang}:{target_lang}:{cleaned_text_main}"
            
            # If file cache is enabled, check if translation exists in file cache
            if file_cache_enabled and cache_key_for_file:
                provider_name = selected_model.replace('_api', '') if '_api' in selected_model else selected_model
                file_cache_result = self.app.cache_manager.check_file_cache(provider_name, cache_key_for_file)
                
                # If not in file cache, save the LRU result to file cache
                if not file_cache_result:
                    log_debug(f"LRU cache hit but file cache miss. Saving to {provider_name} file cache.")
                    self.app.cache_manager.save_to_file_cache(provider_name, cache_key_for_file, cached_result)
                else:
                    log_debug(f"Translation found in both LRU cache and {provider_name} file cache.")
            
            provider = self._get_active_llm_provider()
            if provider and not self._is_error_message(cached_result):
                provider._update_sliding_window(cleaned_text_main, cached_result)
            return self._format_dialog_text(cached_result)

        # 2. Check File Cache
        file_cache_hit = None
        if selected_model == 'gemini_api' and self.app.gemini_file_cache_var.get():
            key = f"gemini:{source_lang}:{target_lang}:{cleaned_text_main}"
            file_cache_hit = self.app.cache_manager.check_file_cache('gemini', key)
        elif selected_model == 'google_api' and self.app.google_file_cache_var.get():
            key = f"google:{source_lang}:{target_lang}:{cleaned_text_main}"
            file_cache_hit = self.app.cache_manager.check_file_cache('google', key)
        elif selected_model == 'deepl_api' and self.app.deepl_file_cache_var.get():
            model_type = extra_params.get('model_type', 'latency_optimized')
            key = f"deepl:{source_lang}:{target_lang}:{model_type}:{cleaned_text_main}"
            file_cache_hit = self.app.cache_manager.check_file_cache('deepl', key)
        elif self.app.is_openai_model(selected_model) and self.app.openai_file_cache_var.get():
            key = f"openai:{source_lang}:{target_lang}:{cleaned_text_main}"
            file_cache_hit = self.app.cache_manager.check_file_cache('openai', key)
        
        if file_cache_hit:
            log_debug(f"Found \"{cleaned_text_main}\" in {selected_model} file cache.")
            self.unified_cache.store(cleaned_text_main, source_lang, target_lang, selected_model, file_cache_hit, **extra_params)
            
            provider = self._get_active_llm_provider()
            if provider and not self._is_error_message(file_cache_hit):
                provider._update_sliding_window(cleaned_text_main, file_cache_hit)
                
            return self._format_dialog_text(file_cache_hit)

        # 3. All Caches Miss - Perform API Call
        log_debug(f"All caches MISS for \"{cleaned_text_main}\". Calling API.")
        translated_api_text = None
        
        if selected_model == 'marianmt':
            translated_api_text = self._marian_translate(cleaned_text_main, source_lang, target_lang, extra_params['beam_size'])
        elif selected_model == 'google_api':
            translated_api_text = self._google_translate(cleaned_text_main, source_lang, target_lang)
        elif selected_model == 'deepl_api':
            translated_api_text = self._deepl_translate(cleaned_text_main, source_lang, target_lang)
        elif selected_model == 'gemini_api':
            provider = self.providers['gemini']
            translated_api_text = provider.translate(cleaned_text_main, source_lang, target_lang, ocr_batch_number)
        elif self.app.is_openai_model(selected_model):
            provider = self.providers['openai']
            translated_api_text = provider.translate(cleaned_text_main, source_lang, target_lang, ocr_batch_number)
        
        # 4. Store successful translation
        if translated_api_text and not self._is_error_message(translated_api_text):
            if selected_model == 'gemini_api' and self.app.gemini_file_cache_var.get():
                cache_key_to_save = f"gemini:{source_lang}:{target_lang}:{cleaned_text_main}"
                self.app.cache_manager.save_to_file_cache('gemini', cache_key_to_save, translated_api_text)
            elif selected_model == 'google_api' and self.app.google_file_cache_var.get():
                cache_key_to_save = f"google:{source_lang}:{target_lang}:{cleaned_text_main}"
                self.app.cache_manager.save_to_file_cache('google', cache_key_to_save, translated_api_text)
            elif selected_model == 'deepl_api' and self.app.deepl_file_cache_var.get():
                model_type = extra_params.get('model_type', 'latency_optimized')
                cache_key_to_save = f"deepl:{source_lang}:{target_lang}:{model_type}:{cleaned_text_main}"
                self.app.cache_manager.save_to_file_cache('deepl', cache_key_to_save, translated_api_text)
            elif self.app.is_openai_model(selected_model) and self.app.openai_file_cache_var.get():
                cache_key_to_save = f"openai:{source_lang}:{target_lang}:{cleaned_text_main}"
                self.app.cache_manager.save_to_file_cache('openai', cache_key_to_save, translated_api_text)

            self.unified_cache.store(cleaned_text_main, source_lang, target_lang, selected_model, translated_api_text, **extra_params)
        
        log_debug(f"Translation \"{cleaned_text_main}\" -> \"{str(translated_api_text)}\" took {time.monotonic() - translation_start_monotonic:.3f}s")
        return self._format_dialog_text(translated_api_text)


    # === NON-LLM PROVIDER METHODS (UNCHANGED) ===
    def _google_translate(self, text_to_translate_gt, source_lang_gt, target_lang_gt):
        log_debug(f"Google Translate API call for: {text_to_translate_gt}")
        api_key_google = self.app.google_api_key_var.get().strip()
        if not api_key_google: return "Google Translate API key missing"
        if not REQUESTS_AVAILABLE: return "Requests library not available for Google Translate"
        try:
            url = "https://translation.googleapis.com/language/translate/v2"
            params = {'key': api_key_google, 'q': text_to_translate_gt, 'target': target_lang_gt, 'format': 'text'}
            if source_lang_gt and source_lang_gt.lower() != 'auto': params['source'] = source_lang_gt
            response = requests.post(url, data=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            if result and 'data' in result and 'translations' in result['data']:
                return html.unescape(result['data']['translations'][0]['translatedText'])
            return f"Google Translate API returned unexpected result: {result}"
        except requests.exceptions.RequestException as e_req:
            return f"Google Translate API request error: {str(e_req)}"
        except Exception as e_cgt:
            return f"Google Translate API error: {type(e_cgt).__name__} - {str(e_cgt)}"

    def _deepl_translate(self, text_to_translate_dl, source_lang_dl, target_lang_dl):
        """Translate using DeepL API with context support and logging."""
        
        # Check for language change and clear context if needed
        if (self.deepl_current_source_lang is not None and 
            self.deepl_current_target_lang is not None and
            (self.deepl_current_source_lang != source_lang_dl or 
             self.deepl_current_target_lang != target_lang_dl)):
            log_debug(f"Language pair changed from {self.deepl_current_source_lang}->{self.deepl_current_target_lang} "
                     f"to {source_lang_dl}->{target_lang_dl}, clearing DeepL context")
            self._clear_deepl_context()
        
        # Track current language pair
        self.deepl_current_source_lang = source_lang_dl
        self.deepl_current_target_lang = target_lang_dl
        
        # Get context window size from settings
        context_size = getattr(self.app, 'deepl_context_window_var', None)
        context_size = context_size.get() if context_size else 0
        
        # Build context string (source language only)
        context_string = self._build_deepl_context(context_size)
        
        # Add custom prompt prefix if available
        custom_prompt = getattr(self.app, 'custom_prompt_text', '').strip()
        if custom_prompt:
            if context_string:
                context_string = f"[{custom_prompt}]\n{context_string}"
            else:
                context_string = f"[{custom_prompt}]\n"
        
        model_type = self.app.deepl_model_type_var.get()
        log_debug(f"DeepL API call for: {text_to_translate_dl} using model_type={model_type}")
        
        if context_string:
            log_debug(f"DeepL context ({len(context_string)} chars, {context_size} subtitles): {context_string[:100]}...")
        
        if not self.app.deepl_api_client:
            return "DeepL API client not initialized"
        
        try:
            deepl_source_param = source_lang_dl if source_lang_dl and source_lang_dl.lower() != 'auto' else None
            
            # Check if source or target language is a beta language
            is_beta_translation = False
            if source_lang_dl and source_lang_dl.upper() in DEEPL_BETA_LANGUAGES:
                is_beta_translation = True
            if target_lang_dl and target_lang_dl.upper() in DEEPL_BETA_LANGUAGES:
                is_beta_translation = True
            
            # Log beta language detection
            if is_beta_translation:
                log_debug(f"Beta language detected - source: {source_lang_dl}, target: {target_lang_dl}")
            
            # Prepare translation parameters
            translate_params = {
                'text': text_to_translate_dl,
                'target_lang': target_lang_dl,
                'model_type': model_type
            }
            
            # Add source language if not auto-detect
            if deepl_source_param:
                translate_params['source_lang'] = deepl_source_param
            
            # Enable beta languages flag if using beta languages
            # Must be passed via extra_body_parameters dictionary (note: plural)
            if is_beta_translation:
                translate_params['extra_body_parameters'] = {'enable_beta_languages': True}
                log_debug("enable_beta_languages=True added to API call via extra_body_parameters")
            
            # Add context if available (IMPORTANT: context not counted for billing!)
            if context_string and deepl_source_param:  # Only send context with explicit source language
                translate_params['context'] = context_string
            
            # Record start time for logging
            call_start_time = datetime.now()
            
            try:
                result_dl = self.app.deepl_api_client.translate_text(**translate_params)
                
                # Calculate call duration
                call_duration = (datetime.now() - call_start_time).total_seconds()
                
                if result_dl and hasattr(result_dl, 'text') and result_dl.text:
                    translated_text = result_dl.text
                    
                    # Log the API call
                    self._log_deepl_translation_call(
                        original_text=text_to_translate_dl,
                        source_lang=source_lang_dl,
                        target_lang=target_lang_dl,
                        context_string=context_string,
                        context_size=context_size,
                        translated_text=translated_text,
                        model_type=model_type,
                        call_start_time=call_start_time,
                        call_duration=call_duration
                    )
                    
                    # Update context window with source text (not translation!)
                    self._update_deepl_context(text_to_translate_dl)
                    return translated_text
                return "DeepL API returned empty or invalid result"
            except Exception as quality_error:
                if (model_type == "quality_optimized" and 
                    ("language pair" in str(quality_error).lower() or 
                     "not supported" in str(quality_error).lower() or 
                     "unsupported" in str(quality_error).lower())):
                    log_debug(f"DeepL quality_optimized failed, falling back to latency_optimized: {quality_error}")
                    
                    # Prepare fallback parameters
                    fallback_params = {
                        'text': text_to_translate_dl,
                        'target_lang': target_lang_dl,
                        'model_type': "latency_optimized"
                    }
                    
                    # Add source language if available
                    if deepl_source_param:
                        fallback_params['source_lang'] = deepl_source_param
                    
                    # Enable beta languages flag if using beta languages
                    # Note: Beta languages cannot use latency_optimized, so this fallback might fail
                    # Must be passed via extra_body_parameters dictionary (note: plural)
                    if is_beta_translation:
                        fallback_params['extra_body_parameters'] = {'enable_beta_languages': True}
                        log_debug("Warning: Beta languages require quality_optimized model, fallback may fail")
                    
                    # Include context in fallback attempt as well
                    if context_string and deepl_source_param:
                        fallback_params['context'] = context_string
                    
                    # Record start time for fallback call
                    fallback_start_time = datetime.now()
                    
                    result_dl_fallback = self.app.deepl_api_client.translate_text(**fallback_params)
                    
                    # Calculate fallback call duration
                    fallback_duration = (datetime.now() - fallback_start_time).total_seconds()
                    
                    if result_dl_fallback and hasattr(result_dl_fallback, 'text') and result_dl_fallback.text:
                        translated_text = result_dl_fallback.text
                        
                        # Log the fallback API call (with latency_optimized model)
                        self._log_deepl_translation_call(
                            original_text=text_to_translate_dl,
                            source_lang=source_lang_dl,
                            target_lang=target_lang_dl,
                            context_string=context_string,
                            context_size=context_size,
                            translated_text=translated_text,
                            model_type="latency_optimized",
                            call_start_time=fallback_start_time,
                            call_duration=fallback_duration
                        )
                        
                        # Update context window with source text (not translation!)
                        self._update_deepl_context(text_to_translate_dl)
                        return translated_text
                    return "DeepL API fallback returned empty or invalid result"
                else:
                    raise quality_error
        except Exception as e_cdl:
            log_debug(f"DeepL API error: {type(e_cdl).__name__} - {str(e_cdl)}")
            return f"DeepL API error: {type(e_cdl).__name__} - {str(e_cdl)}"

    def get_deepl_usage(self):
        """Get DeepL API usage statistics from the usage endpoint."""
        if not self.app.DEEPL_API_AVAILABLE:
            log_debug("DeepL API libraries not available for usage check")
            return None
            
        api_key = self.app.deepl_api_key_var.get().strip()
        if not api_key:
            log_debug("DeepL API key missing for usage check")
            return None
        
        if not REQUESTS_AVAILABLE:
            log_debug("Requests library not available for DeepL usage check")
            return None
        
        try:
            url = "https://api-free.deepl.com/v2/usage"
            headers = {
                "Authorization": f"DeepL-Auth-Key {api_key}",
                "User-Agent": "OCR-Translator/1.1.0"
            }
            
            log_debug("Checking DeepL API usage...")
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                usage_data = response.json()
                log_debug(f"DeepL usage retrieved: {usage_data}")
                return usage_data
            elif response.status_code == 403:
                log_debug("DeepL API: Invalid API key or unauthorized access")
                return None
            elif response.status_code == 456:
                log_debug("DeepL API: Quota exceeded")
                return None
            else:
                log_debug(f"DeepL API error: HTTP {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            log_debug(f"DeepL usage API network error: {e}")
            return None
        except Exception as e:
            log_debug(f"DeepL usage API error: {e}")
            return None

    def _marian_translate(self, text_to_translate_mm, source_lang_mm, target_lang_mm, beam_value_mm):
        log_debug(f"MarianMT translation call for: {text_to_translate_mm} (beam={beam_value_mm})")
        if self.app.marian_translator is None: return "MarianMT translator not initialized"
        text_to_translate_cleaned = re.sub(r'\s+', ' ', text_to_translate_mm).strip()
        if not text_to_translate_cleaned: return ""
        try:
            self.app.marian_translator.num_beams = beam_value_mm
            result_mm = self.app.marian_translator.translate(text_to_translate_cleaned, source_lang_mm, target_lang_mm)
            return result_mm
        except Exception as e_cmm:
            return f"MarianMT translation error: {type(e_cmm).__name__} - {str(e_cmm)}"

    # === UTILITY METHODS (UNCHANGED) ===
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
        # DEBUG: Always log when this function is called
        log_debug(f"DIALOG_FORMAT_DEBUG: _format_dialog_text called with: {repr(text)}")
        
        if not text or not isinstance(text, str):
            log_debug(f"DIALOG_FORMAT_DEBUG: Text is None or not string, returning: {repr(text)}")
            return text
        
        # Check if the text starts with any dash (more robust - no space required)
        dash_check = (text.startswith("-") or text.startswith("–") or text.startswith("—"))
        log_debug(f"DIALOG_FORMAT_DEBUG: Text starts with dash: {dash_check}")
        
        if not dash_check:
            log_debug(f"DIALOG_FORMAT_DEBUG: Text doesn't start with dash, returning unchanged")
            return text
        
        log_debug(f"DIALOG_FORMAT_DEBUG: Text starts with dash, proceeding with formatting")
        
        # Apply the formatting transformations
        formatted_text = text
        
        # Check for patterns before applying
        patterns_found = []
        patterns_to_check = [". -", ". –", ". —", "? -", "? –", "? —", "! -", "! –", "! —"]
        for pattern in patterns_to_check:
            if pattern in text:
                patterns_found.append(pattern)
        
        log_debug(f"DIALOG_FORMAT_DEBUG: Patterns found: {patterns_found}")
        
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
            log_debug(f"DIALOG_FORMAT_DEBUG: Dialog formatting applied!")
            log_debug(f"DIALOG_FORMAT_DEBUG: Original: {repr(text)}")
            log_debug(f"DIALOG_FORMAT_DEBUG: Formatted: {repr(formatted_text)}")
        else:
            log_debug(f"DIALOG_FORMAT_DEBUG: No changes made to text")
        
        return formatted_text
    
    def _is_error_message(self, text):
        return is_translation_error_result(text)
    
    def is_placeholder_text(self, text_content):
        if not text_content: return True
        text_lower = text_content.lower().strip()
        placeholders = ["source text will appear here", "translation will appear here", "translation...", "ocr source", "source text", "loading...", "translating...", "", "translation", "...", "translation error:"]
        return text_lower in placeholders or text_lower.startswith("translation error:")

    def clear_cache(self):
        """Clear the unified translation cache."""
        self.unified_cache.clear_all()
        log_debug("Cleared unified translation cache")

    def update_marian_active_model(self, model_name_uam, source_lang_uam=None, target_lang_uam=None):
        if self.app.marian_translator is None:
            self.initialize_marian_translator()
            if self.app.marian_translator is None:
                log_debug("Cannot update MarianMT model - translator not initialized and init failed.")
                return False
        
        try:
            final_source_lang = source_lang_uam if source_lang_uam else self.app.marian_source_lang
            final_target_lang = target_lang_uam if target_lang_uam else self.app.marian_target_lang

            if not final_source_lang or not final_target_lang:
                log_debug(f"Cannot update MarianMT model '{model_name_uam}': source/target language not determined.")
                return False
            
            log_debug(f"Attempting to make MarianMT model active: {model_name_uam} for {final_source_lang}->{final_target_lang}")

            if hasattr(self.app.marian_translator, '_unload_current_model'):
                self.app.marian_translator._unload_current_model()
            
            if hasattr(self.app.marian_translator, 'direct_pairs'):
                self.app.marian_translator.direct_pairs[(final_source_lang, final_target_lang)] = model_name_uam
            
            self.unified_cache.clear_provider('marianmt')

            if hasattr(self.app.marian_translator, '_try_load_direct_model'):
                load_success = self.app.marian_translator._try_load_direct_model(final_source_lang, final_target_lang)
                if load_success:
                    log_debug(f"Successfully loaded MarianMT model for {final_source_lang}->{final_target_lang}")
                    return True
                else:
                    log_debug(f"Failed to load MarianMT model for {final_source_lang}->{final_target_lang}")
                    return False
            return False
        except Exception as e_umam:
            log_debug(f"Error updating MarianMT active model: {e_umam}")
            return False

    def update_marian_beam_value(self):
        if self.app.marian_translator is not None:
            try:
                beam_value_clamped = max(1, min(50, self.app.num_beams_var.get()))
                if beam_value_clamped != self.app.num_beams_var.get():
                    self.app.num_beams_var.set(beam_value_clamped)
                self.app.marian_translator.num_beams = beam_value_clamped
                log_debug(f"Updated MarianMT beam search value in translator to: {beam_value_clamped}")
            except Exception as e_umbv:
                log_debug(f"Error updating MarianMT beam value: {e_umbv}")

    def calculate_text_similarity(self, text1_sim, text2_sim):
        if not text1_sim or not text2_sim: return 0.0
        if len(text1_sim) < 10 or len(text2_sim) < 10: 
            return 1.0 if text1_sim == text2_sim else 0.0
        
        words1_set = set(text1_sim.lower().split())
        words2_set = set(text2_sim.lower().split())
        intersection_len = len(words1_set.intersection(words2_set))
        union_len = len(words1_set.union(words2_set))
        return intersection_len / union_len if union_len > 0 else 0.0

    def initialize_marian_translator(self):
        if self.app.marian_translator is not None: return
        if not hasattr(self.app, 'MARIANMT_AVAILABLE') or not self.app.MARIANMT_AVAILABLE: return
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir_name = "marian_models_cache"
            if getattr(sys, 'frozen', False):
                 executable_dir = os.path.dirname(sys.executable)
                 cache_dir = os.path.join(executable_dir, "_internal", cache_dir_name)
            else:
                 cache_dir = os.path.join(base_dir, cache_dir_name)
            os.makedirs(cache_dir, exist_ok=True)
            current_beam_value = self.app.num_beams_var.get()
            self.app.marian_translator = MarianMTTranslator(cache_dir=cache_dir, num_beams=current_beam_value)
            log_debug(f"MarianMT translator initialized (cache: {cache_dir}, beams: {current_beam_value})")
        except Exception as e:
            log_debug(f"Error initializing MarianMT translator: {e}\n{traceback.format_exc()}")
