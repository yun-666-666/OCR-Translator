"""Custom AI routing, request snapshots, caching, and concurrency."""

import concurrent.futures
import hashlib
import math
import sys
import time
from urllib.parse import urlsplit, urlunsplit

from custom_ai import (
    CustomAILatencyModeAdvisor,
    CUSTOM_AI_LATENCY_MODE_ADAPTIVE,
    CUSTOM_AI_LATENCY_MODE_RACE,
    CUSTOM_AI_LATENCY_MODE_SAFE,
    CUSTOM_AI_LATENCY_MODE_STREAM,
    CUSTOM_AI_STRUCTURED_OUTPUT_AUTO,
    CUSTOM_AI_STRUCTURED_OUTPUT_CONTRACT_JSON_SCHEMA,
    CUSTOM_AI_STRUCTURED_OUTPUT_CONTRACT_TEXT,
    normalize_custom_ai_latency_mode,
    normalize_custom_ai_structured_output_mode,
    normalize_custom_ai_wire_api,
)
from ai_optimization import looks_like_unsupported_image_format_error
from logger import summarize_text_for_log

CUSTOM_AI_ROUTE_STATE_MAX_ENTRIES = 32
_CUSTOM_AI_PROFILE_UNSET = object()


def _log_debug(message):
    facade = sys.modules.get("handlers.translation_handler")
    if facade is not None:
        return facade.log_debug(message)


class TranslationRequestsMixin:
    def _set_runtime_metric_gauge(self, name, value):
        metrics = getattr(self.app, "runtime_metrics", None)
        setter = getattr(metrics, "set_gauge", None)
        if callable(setter):
            try:
                setter(name, value)
            except Exception:
                pass

    def _set_runtime_metric_label(self, name, value):
        metrics = getattr(self.app, "runtime_metrics", None)
        setter = getattr(metrics, "set_label", None)
        if callable(setter):
            try:
                setter(name, value)
            except Exception:
                pass

    def _increment_runtime_metric(self, name, amount=1):
        metrics = getattr(self.app, "runtime_metrics", None)
        incrementer = getattr(metrics, "increment", None)
        if callable(incrementer):
            try:
                incrementer(name, amount)
            except Exception:
                pass

    def _custom_ai_route_state_key(self, profile):
        """Return a non-secret identity for adaptive state ownership."""
        if not isinstance(profile, dict) or not str(
            profile.get("base_url") or ""
        ).strip():
            return ("custom-ai-default-route",)
        try:
            raw_parts = urlsplit(str(profile.get("base_url") or "").strip())
            host = str(raw_parts.hostname or "").lower()
            if not raw_parts.scheme or not host:
                return ("custom-ai-default-route",)
            if ":" in host:
                host = f"[{host}]"
            port = raw_parts.port
            default_port = (
                (raw_parts.scheme.lower() == "https" and port == 443)
                or (raw_parts.scheme.lower() == "http" and port == 80)
            )
            port_suffix = (
                f":{port}"
                if port is not None and not default_port
                else ""
            )
            safe_profile = dict(profile)
            safe_profile["base_url"] = urlunsplit(
                (
                    raw_parts.scheme.lower(),
                    f"{host}{port_suffix}",
                    raw_parts.path,
                    "",
                    "",
                )
            )
            endpoint = self._canonical_custom_ai_profile_endpoint(
                safe_profile
            )
            endpoint_parts = urlsplit(endpoint)
            endpoint = urlunsplit(
                (
                    endpoint_parts.scheme,
                    endpoint_parts.netloc,
                    endpoint_parts.path,
                    "",
                    "",
                )
            )
            route_options_fingerprint = hashlib.sha256(
                repr(
                    (
                        raw_parts.username or "",
                        raw_parts.password or "",
                        raw_parts.query or "",
                        raw_parts.fragment or "",
                    )
                ).encode("utf-8")
            ).hexdigest()[:16]
            credential_scope = self.custom_ai_provider._credential_scope_key(
                profile
            )
            wire_api = normalize_custom_ai_wire_api(
                profile.get("wire_api")
            )
            model = str(profile.get("model") or "").strip()
            return (
                endpoint,
                route_options_fingerprint,
                credential_scope,
                wire_api,
                model,
            )
        except Exception:
            return ("custom-ai-default-route",)

    def _get_bounded_custom_ai_route_state(self, state_map, route_key, factory):
        with self._custom_route_state_lock:
            state = state_map.pop(route_key, None)
            if state is None:
                state = factory()
            state_map[route_key] = state
            while len(state_map) > CUSTOM_AI_ROUTE_STATE_MAX_ENTRIES:
                state_map.popitem(last=False)
            return state

    def _get_custom_latency_advisor(self, profile=None, route_key=None):
        route_key = route_key or self._custom_ai_route_state_key(profile)
        return self._get_bounded_custom_ai_route_state(
            self._custom_latency_advisors,
            route_key,
            CustomAILatencyModeAdvisor,
        )

    def _get_custom_prompt_cache_metrics(self, profile=None, route_key=None):
        route_key = route_key or self._custom_ai_route_state_key(profile)
        return self._get_bounded_custom_ai_route_state(
            self._custom_prompt_cache_metrics,
            route_key,
            lambda: {
                "sample_count": 0,
                "cached_input_ratio_ema": None,
                "input_tokens_ema": None,
            },
        )

    def _note_custom_ai_race_winner(self, profile):
        profile = profile if isinstance(profile, dict) else {}
        self._set_runtime_metric_label(
            "race_winner",
            f"custom_ai / {profile.get('name', 'Custom AI')}",
        )

    def _get_active_llm_provider(self):
        """Get the currently active LLM provider based on selected translation model."""
        return None

    def _get_active_ocr_provider(self):
        """Get the currently active OCR provider based on selected OCR model."""
        return None

    def _sanitize_custom_ai_profile_error(self, error, profile):
        error_text = str(error)
        sanitizer = getattr(
            self.custom_ai_provider,
            "_sanitize_error",
            None,
        )
        if callable(sanitizer):
            try:
                error_text = sanitizer(
                    error_text,
                    (
                        profile.get("api_key", "")
                        if isinstance(profile, dict)
                        else ""
                    ),
                )
            except Exception:
                pass
        return error_text

    def _classify_custom_ai_profile_failure(self, error_text):
        """Return (cooldown_seconds, exponential_backoff) for a profile failure.

        Cooldown duration and exponential-backoff decisions share this single
        classification source so the two cannot drift. Bare substrings such as
        "gateway" or "connection" are intentionally not treated as transient.
        """
        lowered = str(error_text or "").strip().lower()
        permanent_markers = (
            "http 401",
            "http 402",
            "http 403",
            "insufficient_balance",
            "insufficient balance",
            "precharge",
            "预扣费",
            "余额不足",
        )
        if any(marker in lowered for marker in permanent_markers):
            return 300.0, False

        deterministic_markers = (
            "http 404",
            "invalid model",
            "model not found",
            "invalid endpoint",
        )
        if any(marker in lowered for marker in deterministic_markers):
            return 300.0, False

        # HTTP 429 and explicit rate-limit responses already have a dedicated
        # transport-scoped cooldown, including Retry-After handling.  Do not
        # add a second profile-unavailable cooldown on top of it: that used to
        # turn xAI's 15-second rate-limit wait into a 30-second subtitle queue.
        rate_limit_markers = (
            "http 429",
            "rate limit",
            "too many requests",
            "retry-after",
            "retry after",
            "quota exceeded",
        )
        if any(marker in lowered for marker in rate_limit_markers):
            return 0.0, False

        empty_markers = (
            "did not contain message content",
            "returned an invalid translation",
            "empty response",
            "empty content",
        )
        if any(marker in lowered for marker in empty_markers):
            return 10.0, False

        connection_reset_markers = (
            "ssl",
            "unexpected_eof",
            "eof occurred",
            "connection refused",
            "connection reset",
            "connection aborted",
            "remote end closed",
            "failed to establish a new connection",
        )
        if any(marker in lowered for marker in connection_reset_markers):
            # A dead keep-alive/TLS connection is frequently healthy again on
            # a fresh session.  Probe quickly, then back off exponentially if
            # the failure persists instead of freezing subtitle delivery.
            return 2.0, True

        timeout_markers = (
            "read timed out",
            "read timeout",
            "connect timeout",
            "connection timed out",
            "name or service not known",
            "nodename nor servname",
            "temporary failure in name resolution",
            "getaddrinfo failed",
        )
        if any(marker in lowered for marker in timeout_markers):
            return 5.0, True

        strict_transient_markers = (
            "http 502",
            "http 503",
            "http 504",
            "bad gateway",
        )
        if any(marker in lowered for marker in strict_transient_markers):
            return 15.0, True

        soft_transient_markers = (
            "http 500",
            "non-json",
            "non json",
        )
        if any(marker in lowered for marker in soft_transient_markers):
            return 15.0, False

        return 30.0, False

    def _custom_ai_profile_failure_cooldown_seconds(self, error_text):
        return self._classify_custom_ai_profile_failure(error_text)[0]

    def _custom_ai_profile_failure_uses_exponential_backoff(self, error_text):
        return self._classify_custom_ai_profile_failure(error_text)[1]

    def _custom_ai_profile_failure_is_rate_limited(self, error_text):
        cooldown_seconds, _ = self._classify_custom_ai_profile_failure(
            error_text
        )
        return cooldown_seconds == 0.0

    def _begin_custom_ai_profile_request(self, profile, request_kind):
        begin = getattr(
            self.custom_ai_provider,
            "begin_profile_request",
            None,
        )
        if not callable(begin):
            return None
        try:
            return begin(profile, request_kind=request_kind)
        except Exception as sequence_error:
            _log_debug(
                "Custom AI profile request sequencing failed: "
                f"{type(sequence_error).__name__}"
            )
            return None

    def _mark_custom_ai_profile_failure(
        self,
        profile,
        error_text,
        request_kind="translation",
        request_sequence=None,
    ):
        if self._custom_ai_profile_failure_is_rate_limited(error_text):
            _log_debug(
                "LATENCY: retained Custom AI rate-limit cooldown without "
                "additional profile-unavailable backoff "
                f"request_kind={request_kind or 'shared'}"
            )
            return
        marker = getattr(
            self.custom_ai_provider,
            "mark_profile_unavailable",
            None,
        )
        if callable(marker):
            marker_kwargs = {
                "seconds": self._custom_ai_profile_failure_cooldown_seconds(
                    error_text
                ),
                "request_kind": request_kind,
                "request_sequence": request_sequence,
            }
            if self._custom_ai_profile_failure_uses_exponential_backoff(error_text):
                marker_kwargs["exponential_backoff"] = True
            marker(profile, error_text, **marker_kwargs)

    def perform_ocr(
        self,
        image_data,
        source_lang,
        image_mime_type="image/webp",
        image_detail="auto",
        image_format="webp",
        keep_linebreaks=None,
    ):
        """Main public method for performing OCR. Delegates to the currently selected API provider."""
        profile = self.app.custom_ai_profiles.get_active_profile("ocr")
        if not profile:
            _log_debug("No active custom AI model profile configured for OCR")
            return "<e>: AI model profile for OCR is missing"
        # Mirror translation speed policy: force reasoning none for OCR requests
        # under speed mode so remote token spend matches the user-facing policy.
        profile = self._translation_request_profile(profile)
        request_sequence = self._begin_custom_ai_profile_request(
            profile,
            "ocr",
        )
        try:
            latency_mode = self._get_custom_ai_latency_mode()
            if latency_mode == CUSTOM_AI_LATENCY_MODE_ADAPTIVE:
                latency_mode = CUSTOM_AI_LATENCY_MODE_SAFE
            if keep_linebreaks is None:
                keep_linebreaks_var = getattr(self.app, "keep_linebreaks_var", None)
                try:
                    keep_linebreaks = (
                        bool(keep_linebreaks_var.get())
                        if keep_linebreaks_var is not None
                        else False
                    )
                except Exception:
                    keep_linebreaks = False
            else:
                keep_linebreaks = bool(keep_linebreaks)
            result, usage, duration = self.custom_ai_provider.recognize(
                profile,
                image_data,
                source_lang,
                keep_linebreaks=keep_linebreaks,
                latency_mode=latency_mode,
                image_detail=image_detail,
                image_mime_type=image_mime_type,
            )
            available = getattr(
                self.custom_ai_provider,
                "mark_profile_available",
                None,
            )
            if callable(available):
                available(
                    profile,
                    request_kind="ocr",
                    request_sequence=request_sequence,
                )
            self._log_custom_short_call("ocr", profile, result, usage, duration)
            return result
        except Exception as e:
            error_text = self._sanitize_custom_ai_profile_error(e, profile)
            capability_memory = getattr(
                self.app,
                "ai_ocr_image_capability_memory",
                None,
            )
            if (
                capability_memory is not None
                and looks_like_unsupported_image_format_error(
                    error_text,
                    image_format,
                )
            ):
                capability_memory.mark_format_unsupported(
                    profile,
                    image_format,
                )
                _log_debug(
                    "Remembered unsupported Custom AI OCR image format "
                    f"format={image_format}"
                )
            self._mark_custom_ai_profile_failure(
                profile,
                error_text,
                request_kind="ocr",
                request_sequence=request_sequence,
            )
            _log_debug(f"Custom AI OCR error: {type(e).__name__} - {error_text}")
            return f"<e>: Custom AI OCR error: {type(e).__name__} - {error_text}"

    # === LLM SESSION MANAGEMENT ===
    def translate_text_with_timeout(
        self,
        text_content,
        timeout_seconds=10.0,
        ocr_batch_number=None,
        stream_callback=None,
        translation_sequence=None,
        latency_mode=None,
        request_snapshot=None,
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
                latency_mode=latency_mode,
                timeout_seconds=timeout_seconds,
                request_snapshot=request_snapshot,
            )
        except Exception as e:
            _log_debug(f"Translation exception: {e}")
            return f"Translation error: {str(e)}"

    def translate_text(self, text_content_main, ocr_batch_number=None, stream_callback=None, translation_sequence=None, latency_mode=None, timeout_seconds=None, request_snapshot=None):
        cleaned_text_main = text_content_main.strip() if text_content_main else ""
        if not cleaned_text_main or self.is_placeholder_text(cleaned_text_main):
            return None

        translation_start_monotonic = time.monotonic()
        selected_model = self.app.translation_model_var.get()
        _log_debug(
            "Translate request "
            f"{summarize_text_for_log(cleaned_text_main)} "
            f"using {selected_model}"
        )

        if selected_model != 'custom_ai':
            _log_debug(f"Legacy translation model '{selected_model}' is disabled; using custom_ai route")
            selected_model = 'custom_ai'

        return self._custom_ai_translate(
            cleaned_text_main,
            translation_start_monotonic,
            stream_callback=stream_callback,
            translation_sequence=translation_sequence,
            latency_mode=latency_mode,
            timeout_seconds=timeout_seconds,
            request_snapshot=request_snapshot,
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
            translation_sequence = None
            try:
                translation_sequence = (
                    int(self.app.translation_sequence_counter) + 1
                )
            except (AttributeError, TypeError, ValueError):
                pass
            return self._get_custom_ai_cached_translation(
                cleaned_text,
                translation_sequence=translation_sequence,
            )

        return None

    def _get_custom_ai_cache_profile_and_params(
        self,
        current_source=None,
        latency_mode=None,
        profile=_CUSTOM_AI_PROFILE_UNSET,
        force_no_reasoning=None,
    ):
        if profile is _CUSTOM_AI_PROFILE_UNSET:
            profile = self.app.custom_ai_profiles.get_active_profile(
                "translation"
            )
        if not profile:
            return None, None, None, None
        profile = self._translation_request_profile(
            profile,
            force_no_reasoning=force_no_reasoning,
        )

        source_lang = getattr(self.app, 'custom_source_lang', None) or self.app.source_lang_var.get()
        target_lang = getattr(self.app, 'custom_target_lang', None) or self.app.target_lang_var.get()
        cache_params = self._cache_params_for_profile(
            profile,
            current_source=current_source,
            latency_mode=latency_mode,
        )
        return profile, source_lang, target_lang, cache_params

    def _speed_translation_policy_enabled(self):
        optimization_getter = getattr(
            self.app,
            "get_ai_optimization_mode",
            None,
        )
        try:
            optimization_mode = (
                optimization_getter()
                if callable(optimization_getter)
                else ""
            )
        except Exception:
            optimization_mode = ""
        return str(optimization_mode or "").strip().lower() == "speed"

    def _translation_request_profile(
        self,
        profile,
        force_no_reasoning=None,
    ):
        request_profile = dict(profile) if isinstance(profile, dict) else {}
        if force_no_reasoning is None:
            force_no_reasoning = self._speed_translation_policy_enabled()
        if force_no_reasoning:
            request_profile["reasoning_effort"] = "none"
        return request_profile

    def _get_custom_ai_cached_translation(
        self,
        cleaned_text,
        translation_sequence=None,
        context_generation=None,
        profile=None,
        source_lang=None,
        target_lang=None,
        cache_params=None,
    ):
        if context_generation is None:
            with self._custom_context_lock:
                context_generation = self._custom_context_generation
        if cache_params is None:
            (
                profile,
                source_lang,
                target_lang,
                cache_params,
            ) = self._get_custom_ai_cache_profile_and_params(
                current_source=cleaned_text,
            )
        if not profile:
            return None

        cached_result = self.unified_cache.get(
            cleaned_text,
            source_lang,
            target_lang,
            "custom_ai",
            **cache_params,
        )
        if not cached_result:
            # H2-B: peer-contract secondary lookup for display-equivalent
            # text/json_schema transport flips under AUTO only.
            for peer_params in self._custom_ai_display_cache_param_variants(
                cache_params,
                profile=profile,
            )[1:]:
                cached_result = self.unified_cache.get(
                    cleaned_text,
                    source_lang,
                    target_lang,
                    "custom_ai",
                    **peer_params,
                )
                if cached_result:
                    self._increment_runtime_metric(
                        "translation_cache_contract_peer_hit"
                    )
                    break
        if not cached_result:
            return None

        self._update_custom_context(
            cleaned_text,
            cached_result,
            translation_sequence=translation_sequence,
            context_generation=context_generation,
        )
        return self._format_dialog_text(cached_result)

    def get_inflight_translation_key(self, text_content):
        snapshot = self.get_custom_ai_translation_request_snapshot(
            text_content,
            commit=False,
        )
        if snapshot:
            return snapshot.get("inflight_key")
        return None

    def get_custom_ai_translation_request_snapshot(
        self,
        text_content,
        commit=False,
    ):
        cleaned_text = text_content.strip() if text_content else ""
        if not cleaned_text or self.is_placeholder_text(cleaned_text):
            return None

        configured_latency_mode = self._get_custom_ai_latency_mode()
        force_no_reasoning = self._speed_translation_policy_enabled()
        try:
            active_profile = self.app.custom_ai_profiles.get_active_profile(
                "translation"
            )
        except Exception:
            active_profile = None
        active_profile = dict(active_profile) if active_profile else None
        decision = self._resolve_custom_ai_latency_mode_for_request(
            current_source=cleaned_text,
            configured_mode=configured_latency_mode,
            commit=commit,
            profile=active_profile,
        )
        latency_mode = decision.mode
        with self._custom_route_state_lock:
            timeout_decision = self._get_custom_latency_advisor(
                profile=active_profile
            ).resolve_request_timeout(
                10.0,
                latency_mode=latency_mode,
            )
        profile, source_lang, target_lang, cache_params = self._get_custom_ai_cache_profile_and_params(
            current_source=cleaned_text,
            latency_mode=latency_mode,
            profile=active_profile,
            force_no_reasoning=force_no_reasoning,
        )
        if not profile:
            inflight_key = ("custom_ai", cleaned_text, "missing_profile")
        else:
            inflight_key = self._build_custom_ai_inflight_key(
                cleaned_text,
                source_lang,
                target_lang,
                cache_params,
                latency_mode,
            )

        with self._custom_context_lock:
            context_generation = self._custom_context_generation
        timeout_seconds = float(timeout_decision.seconds)
        deadline_monotonic = time.monotonic() + timeout_seconds
        return {
            "inflight_key": inflight_key,
            "latency_mode": latency_mode,
            "configured_latency_mode": configured_latency_mode,
            "force_no_reasoning": force_no_reasoning,
            "reason": decision.reason,
            "p90_seconds": decision.p90_seconds,
            "sample_count": decision.sample_count,
            "timeout_seconds": timeout_seconds,
            "timeout_reason": timeout_decision.reason,
            "timeout_p90_seconds": timeout_decision.p90_seconds,
            "timeout_sample_count": timeout_decision.sample_count,
            "route_p90_seconds": timeout_decision.p90_seconds,
            "route_sample_count": timeout_decision.sample_count,
            "deadline_monotonic": deadline_monotonic,
            "profile": dict(profile) if profile else None,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "cache_params": dict(cache_params) if cache_params else None,
            "context_generation": context_generation,
        }

    def commit_custom_ai_latency_mode_snapshot(self, snapshot):
        if not isinstance(snapshot, dict):
            return
        self._set_runtime_metric_gauge(
            "custom_ai_request_timeout_seconds",
            snapshot.get("timeout_seconds", 10.0),
        )
        self._set_runtime_metric_label(
            "custom_ai_request_timeout_reason",
            snapshot.get("timeout_reason", "configured"),
        )
        self._commit_custom_ai_latency_mode(
            snapshot.get("configured_latency_mode"),
            snapshot.get("latency_mode"),
            snapshot.get("reason", ""),
            snapshot.get("p90_seconds", 0.0),
            snapshot.get("sample_count", 0),
            profile=snapshot.get("profile"),
        )

    def _commit_custom_ai_latency_mode(
        self,
        configured_mode,
        resolved_mode,
        reason,
        p90_seconds=0.0,
        sample_count=0,
        profile=None,
    ):
        configured_mode = normalize_custom_ai_latency_mode(configured_mode)
        resolved_mode = normalize_custom_ai_latency_mode(resolved_mode)
        if configured_mode != CUSTOM_AI_LATENCY_MODE_ADAPTIVE:
            return
        with self._custom_route_state_lock:
            self._get_custom_latency_advisor(profile=profile).commit_mode(
                resolved_mode
            )
        _log_debug(
            "LATENCY: adaptive_latency_mode "
            f"resolved={resolved_mode} "
            f"reason={reason} "
            f"p90={float(p90_seconds or 0.0):.3f}s "
            f"samples={int(sample_count or 0)}"
        )

    def _build_custom_ai_inflight_key(
        self,
        cleaned_text,
        source_lang,
        target_lang,
        cache_params,
        latency_mode,
    ):
        return (
            "custom_ai",
            cleaned_text,
            source_lang,
            target_lang,
            cache_params.get("profile_id", ""),
            cache_params.get("base_url", ""),
            cache_params.get("model", ""),
            cache_params.get("credential_scope", ""),
            cache_params.get("wire_api", "chat_completions"),
            cache_params.get("reasoning_effort", ""),
            cache_params.get("structured_output_contract", "text"),
            cache_params.get("custom_prompt", ""),
            cache_params.get("keep_linebreaks", False),
            cache_params.get("context", ()),
            latency_mode,
        )

    def _get_custom_ai_latency_mode(self):
        if hasattr(self.app, 'get_custom_ai_latency_mode'):
            return normalize_custom_ai_latency_mode(
                self.app.get_custom_ai_latency_mode()
            )
        var = getattr(self.app, 'custom_ai_latency_mode_var', None)
        try:
            return normalize_custom_ai_latency_mode(var.get() if var is not None else CUSTOM_AI_LATENCY_MODE_SAFE)
        except Exception:
            return CUSTOM_AI_LATENCY_MODE_SAFE

    def _resolve_custom_ai_latency_mode_for_request(
        self,
        current_source=None,
        configured_mode=None,
        commit=False,
        profile=_CUSTOM_AI_PROFILE_UNSET,
    ):
        configured_mode = normalize_custom_ai_latency_mode(
            configured_mode
            if configured_mode is not None
            else self._get_custom_ai_latency_mode()
        )
        if profile is _CUSTOM_AI_PROFILE_UNSET:
            try:
                profile = self.app.custom_ai_profiles.get_active_profile(
                    "translation"
                )
            except Exception:
                profile = None
        profile = dict(profile) if isinstance(profile, dict) else None
        if configured_mode != CUSTOM_AI_LATENCY_MODE_ADAPTIVE:
            with self._custom_route_state_lock:
                return self._get_custom_latency_advisor(
                    profile=profile
                ).resolve(configured_mode)

        stream_supported = self._custom_ai_profile_supports_stream(profile)
        primary_cooldown = self._custom_ai_profile_cooldown_seconds(profile)
        healthy_race_count = len(
            self._get_healthy_custom_ai_race_profiles(profile)
        )
        with self._custom_route_state_lock:
            decision = self._get_custom_latency_advisor(
                profile=profile
            ).resolve(
                configured_mode,
                stream_supported=stream_supported,
                healthy_race_profile_count=healthy_race_count,
                primary_cooldown_seconds=primary_cooldown,
                commit=commit,
            )
        if commit:
            _log_debug(
                "LATENCY: adaptive_latency_mode "
                f"resolved={decision.mode} reason={decision.reason} "
                f"p90={decision.p90_seconds:.3f}s "
                f"samples={decision.sample_count} "
                f"healthy_race_profiles={healthy_race_count} "
                f"cooldown={primary_cooldown:.3f}s"
            )
        return decision

    def _custom_ai_profile_supports_stream(self, profile):
        checker = getattr(
            self.custom_ai_provider,
            "profile_supports_streaming",
            None,
        )
        if callable(checker):
            try:
                return bool(checker(profile))
            except Exception as stream_error:
                _log_debug(
                    "Custom AI stream capability check failed: "
                    f"{type(stream_error).__name__} - {stream_error}"
                )
        return bool(profile)

    def _custom_ai_profile_cooldown_seconds(
        self,
        profile,
        request_kind="translation",
    ):
        if not isinstance(profile, dict):
            return 0.0
        cooldown_getter = getattr(
            self.custom_ai_provider,
            "get_cooldown_remaining",
            None,
        )
        if not callable(cooldown_getter):
            return 0.0
        try:
            try:
                remaining = cooldown_getter(
                    profile,
                    request_kind=request_kind,
                )
            except TypeError:
                remaining = cooldown_getter(profile)
            return max(0.0, float(remaining))
        except Exception as cooldown_error:
            _log_debug(
                "Custom AI cooldown check failed: "
                f"{type(cooldown_error).__name__} - {cooldown_error}"
            )
            return 0.0

    def get_active_custom_ai_ocr_cooldown_seconds(self):
        profiles = getattr(self.app, "custom_ai_profiles", None)
        getter = getattr(profiles, "get_active_profile", None)
        if not callable(getter):
            return 0.0
        return self._custom_ai_profile_cooldown_seconds(
            getter("ocr"),
            request_kind="ocr",
        )

    def _get_healthy_custom_ai_race_profiles(self, active_profile):
        if not isinstance(active_profile, dict):
            return []
        candidates = self._get_custom_ai_race_profiles(active_profile)
        healthy = []
        for candidate in candidates:
            remaining = self._custom_ai_profile_cooldown_seconds(
                candidate,
                request_kind="translation",
            )
            if remaining <= 0.0:
                healthy.append(candidate)
        return healthy

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

    def get_translation_provider_cooldown_seconds(self, latency_mode=None):
        selected_model = self.app.translation_model_var.get()
        if selected_model != 'custom_ai':
            self._set_runtime_metric_gauge("provider_cooldown_seconds", 0.0)
            return 0.0

        profile = self.app.custom_ai_profiles.get_active_profile("translation")
        if not profile:
            self._set_runtime_metric_gauge("provider_cooldown_seconds", 0.0)
            return 0.0

        cooldown_getter = getattr(self.custom_ai_provider, "get_cooldown_remaining", None)
        if not callable(cooldown_getter):
            self._set_runtime_metric_gauge("provider_cooldown_seconds", 0.0)
            return 0.0

        try:
            request_latency_mode = normalize_custom_ai_latency_mode(
                latency_mode
                if latency_mode is not None
                else self._get_custom_ai_latency_mode()
            )
            if request_latency_mode == CUSTOM_AI_LATENCY_MODE_ADAPTIVE:
                request_latency_mode = (
                    self._resolve_custom_ai_latency_mode_for_request().mode
                )
            if request_latency_mode == CUSTOM_AI_LATENCY_MODE_RACE:
                profiles = self._get_custom_ai_race_profiles(profile)
            else:
                profiles = self._get_custom_ai_failover_profiles(profile)
            remaining_values = [
                self._custom_ai_profile_cooldown_seconds(
                    candidate,
                    request_kind="translation",
                )
                for candidate in profiles
            ]
            remaining = min(remaining_values) if remaining_values else 0.0
            self._set_runtime_metric_gauge("provider_cooldown_seconds", remaining)
            return remaining
        except Exception as cooldown_error:
            _log_debug(
                "Custom AI cooldown check failed: "
                f"{type(cooldown_error).__name__} - {cooldown_error}"
            )
            self._set_runtime_metric_gauge("provider_cooldown_seconds", 0.0)
            return 0.0

    def _custom_ai_translate(
        self,
        cleaned_text_main,
        translation_start_monotonic,
        stream_callback=None,
        translation_sequence=None,
        latency_mode=None,
        timeout_seconds=None,
        request_snapshot=None,
    ):
        use_request_snapshot = (
            isinstance(request_snapshot, dict)
            and "profile" in request_snapshot
            and "cache_params" in request_snapshot
        )
        if use_request_snapshot:
            profile_value = request_snapshot.get("profile")
            profile = dict(profile_value) if profile_value else None
            source_lang = request_snapshot.get("source_lang")
            target_lang = request_snapshot.get("target_lang")
            cache_params_value = request_snapshot.get("cache_params")
            cache_params = (
                dict(cache_params_value) if cache_params_value else None
            )
            configured_latency_mode = normalize_custom_ai_latency_mode(
                request_snapshot.get("configured_latency_mode")
                or request_snapshot.get("latency_mode")
                or latency_mode
            )
            latency_mode = normalize_custom_ai_latency_mode(
                request_snapshot.get("latency_mode") or latency_mode
            )
            context_generation = request_snapshot.get("context_generation")
            force_no_reasoning = bool(
                request_snapshot.get("force_no_reasoning", False)
            )
            if context_generation is None:
                with self._custom_context_lock:
                    context_generation = self._custom_context_generation
            decision = None
        else:
            force_no_reasoning = self._speed_translation_policy_enabled()
            with self._custom_context_lock:
                context_generation = self._custom_context_generation
            configured_latency_mode = normalize_custom_ai_latency_mode(
                self._get_custom_ai_latency_mode()
                if latency_mode is None
                else latency_mode
            )
            try:
                profile = self.app.custom_ai_profiles.get_active_profile(
                    "translation"
                )
            except Exception:
                profile = None
            profile = dict(profile) if profile else None
            decision = self._resolve_custom_ai_latency_mode_for_request(
                current_source=cleaned_text_main,
                configured_mode=configured_latency_mode,
                commit=False,
                profile=profile,
            )
            latency_mode = decision.mode
            profile, source_lang, target_lang, cache_params = self._get_custom_ai_cache_profile_and_params(
                current_source=cleaned_text_main,
                latency_mode=latency_mode,
                profile=profile,
                force_no_reasoning=force_no_reasoning,
            )
        if not profile:
            return "AI model profile for translation is missing."

        context = list(cache_params.get("context", ()))
        keep_linebreaks = bool(cache_params.get("keep_linebreaks", False))
        custom_prompt = cache_params.get("custom_prompt", "")
        if latency_mode != CUSTOM_AI_LATENCY_MODE_RACE:
            return self._custom_ai_translate_with_failover(
                profile,
                cleaned_text_main,
                source_lang,
                target_lang,
                context,
                keep_linebreaks,
                custom_prompt=custom_prompt,
                latency_mode=latency_mode,
                stream_callback=stream_callback,
                timeout_seconds=timeout_seconds,
                translation_start_monotonic=translation_start_monotonic,
                translation_sequence=translation_sequence,
                context_generation=context_generation,
                primary_cache_params=cache_params,
                force_no_reasoning=force_no_reasoning,
                request_snapshot=(
                    request_snapshot if use_request_snapshot else None
                ),
            )

        cached_result = self._get_custom_ai_cached_translation(
            cleaned_text_main,
            translation_sequence=translation_sequence,
            context_generation=context_generation,
            profile=profile,
            source_lang=source_lang,
            target_lang=target_lang,
            cache_params=cache_params,
        )
        if cached_result:
            return cached_result

        if decision is not None:
            self._commit_custom_ai_latency_mode(
                configured_latency_mode,
                latency_mode,
                decision.reason,
                decision.p90_seconds,
                decision.sample_count,
                profile=profile,
            )

        try:
            if latency_mode == CUSTOM_AI_LATENCY_MODE_RACE:
                # Pre-start gate only: mirror failover stopped/obsolete/deadline
                # checks before opening N parallel race HTTP calls. Does not
                # cancel already-in-flight race candidates.
                deadline_monotonic = (
                    self._resolve_custom_ai_request_deadline_monotonic(
                        request_snapshot=(
                            request_snapshot if use_request_snapshot else None
                        ),
                        timeout_seconds=timeout_seconds,
                        translation_start_monotonic=translation_start_monotonic,
                    )
                )
                preflight = self._custom_ai_failover_preflight(
                    translation_sequence,
                    deadline_monotonic=deadline_monotonic,
                    timeout_seconds=timeout_seconds,
                )
                if preflight["action"] != "continue":
                    return preflight["result"]
                race_timeout_seconds = preflight["timeout_seconds"]
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
                    custom_prompt=custom_prompt,
                    timeout_seconds=race_timeout_seconds,
                    force_no_reasoning=force_no_reasoning,
                )
            else:
                translated_api_text, usage, duration = self.custom_ai_provider.translate(
                    profile,
                    cleaned_text_main,
                    source_lang,
                    target_lang,
                    custom_prompt=custom_prompt,
                    context=context,
                    keep_linebreaks=keep_linebreaks,
                    latency_mode=latency_mode,
                    stream_callback=stream_callback if latency_mode == CUSTOM_AI_LATENCY_MODE_STREAM else None,
                    timeout_seconds=timeout_seconds,
                )
                winning_profile = profile
                cache_params = self._cache_params_for_profile(
                    profile,
                    custom_prompt=custom_prompt,
                    keep_linebreaks=keep_linebreaks,
                    context=context,
                    latency_mode=latency_mode,
            )
            self._log_custom_short_call("translation", winning_profile, translated_api_text, usage, duration)
        except Exception as e:
            self._record_custom_ai_latency_observation(
                time.monotonic() - translation_start_monotonic,
                success=False,
                profile=profile,
            )
            error_text = self._sanitize_custom_ai_profile_error(
                e,
                profile,
            )
            _log_debug(
                "Custom AI translation error: "
                f"{type(e).__name__} - {error_text}"
            )
            return (
                "Custom AI translation error: "
                f"{type(e).__name__} - {error_text}"
            )

        if translated_api_text and not self._is_error_message(translated_api_text):
            # Each target keeps the profile that owns those cache params so
            # H2 AUTO peer dual-write is never decided by a different profile.
            cache_targets = [(cache_params, winning_profile)]
            if latency_mode == CUSTOM_AI_LATENCY_MODE_RACE:
                active_cache_params = self._cache_params_for_profile(
                    profile,
                    custom_prompt=custom_prompt,
                    keep_linebreaks=keep_linebreaks,
                    context=context,
                    latency_mode=latency_mode,
                )
                if (
                    active_cache_params != cache_params
                    and active_cache_params.get(
                        "structured_output_contract"
                    )
                    == cache_params.get("structured_output_contract")
                ):
                    cache_targets.append((active_cache_params, profile))
            self._store_custom_ai_cached_translation(
                cleaned_text_main,
                source_lang,
                target_lang,
                translated_api_text,
                cache_targets,
            )
            self._update_custom_context(
                cleaned_text_main,
                translated_api_text,
                translation_sequence=translation_sequence,
                context_generation=context_generation,
            )

        _log_debug(
            "Custom AI translation completed "
            f"source {summarize_text_for_log(cleaned_text_main)} "
            f"result {summarize_text_for_log(translated_api_text)} "
            f"took {time.monotonic() - translation_start_monotonic:.3f}s"
        )
        return self._format_dialog_text(translated_api_text)

    def _is_custom_ai_translation_failover_eligible(
        self,
        profile,
        active_profile,
    ):
        """Return whether a profile may join translation failover/race.

        The active translation profile is always eligible. Other profiles must
        be enabled and explicitly opt in via translation_failover_enabled.
        """
        if not isinstance(profile, dict) or not profile.get("enabled", True):
            return False
        active = active_profile if isinstance(active_profile, dict) else {}
        profile_id = str(profile.get("id") or "").strip()
        active_id = str(active.get("id") or "").strip()
        if profile_id and active_id and profile_id == active_id:
            return True
        if not profile_id and not active_id and profile is active_profile:
            return True
        return bool(profile.get("translation_failover_enabled", False))

    def _iter_custom_ai_translation_failover_source_profiles(
        self,
        active_profile,
    ):
        """Yield active profile first, then enabled profiles from storage."""
        if isinstance(active_profile, dict):
            yield active_profile
        try:
            listed = self.app.custom_ai_profiles.list_profiles(enabled_only=True)
        except Exception as error:
            _log_debug(
                "Custom AI failover profile lookup failed: "
                f"{type(error).__name__} - {error}"
            )
            listed = []
        for profile in listed:
            if isinstance(profile, dict):
                yield profile

    def _get_custom_ai_failover_profiles(
        self,
        active_profile,
        force_no_reasoning=None,
    ):
        candidates = []
        seen = set()
        for candidate in self._iter_custom_ai_translation_failover_source_profiles(
            active_profile
        ):
            if not self._is_custom_ai_translation_failover_eligible(
                candidate,
                active_profile,
            ):
                continue
            candidate = self._translation_request_profile(
                candidate,
                force_no_reasoning=force_no_reasoning,
            )
            candidate_id = str(candidate.get("id") or "").strip()
            identity = (
                "id",
                candidate_id,
            ) if candidate_id else (
                "endpoint",
                candidate.get("base_url", ""),
                candidate.get("model", ""),
                candidate.get("wire_api", "chat_completions"),
            )
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(candidate)
        return candidates

    def _custom_ai_translation_is_obsolete(self, translation_sequence):
        """Return whether a newer translation has started or reached the display."""
        try:
            request_sequence = int(translation_sequence)
        except (TypeError, ValueError):
            return False
        try:
            latest_started_sequence = int(
                getattr(
                    self.app,
                    "latest_translation_sequence_started",
                    0,
                )
                or 0
            )
        except (TypeError, ValueError):
            latest_started_sequence = 0
        try:
            displayed_sequence = int(
                getattr(
                    self.app,
                    "last_displayed_translation_sequence",
                    0,
                )
                or 0
            )
        except (TypeError, ValueError):
            displayed_sequence = 0
        return request_sequence > 0 and (
            (
                latest_started_sequence > 0
                and request_sequence < latest_started_sequence
            )
            or (
                displayed_sequence > 0
                and request_sequence <= displayed_sequence
            )
        )

    def _abort_obsolete_custom_ai_failover(self, translation_sequence):
        if not self._custom_ai_translation_is_obsolete(translation_sequence):
            return False
        _log_debug(
            "LATENCY: custom_ai failover aborted for obsolete translation "
            f"sequence={translation_sequence} "
            "latest_started="
            f"{getattr(self.app, 'latest_translation_sequence_started', 0)} "
            "last_displayed="
            f"{getattr(self.app, 'last_displayed_translation_sequence', 0)}"
        )
        return True

    def _abort_stopped_custom_ai_failover(self, translation_sequence):
        if not hasattr(self.app, "is_running"):
            return False
        if bool(getattr(self.app, "is_running", True)):
            return False
        _log_debug(
            "LATENCY: custom_ai failover aborted because app stopped "
            f"sequence={translation_sequence}"
        )
        return True

    def _abort_custom_ai_failover(self, translation_sequence):
        return (
            self._abort_stopped_custom_ai_failover(translation_sequence)
            or self._abort_obsolete_custom_ai_failover(translation_sequence)
        )

    def _resolve_custom_ai_request_deadline_monotonic(
        self,
        request_snapshot=None,
        timeout_seconds=None,
        translation_start_monotonic=None,
    ):
        """Return the shared request deadline, or None when no budget is frozen."""
        snapshot_timeout = None
        if isinstance(request_snapshot, dict):
            raw_deadline = request_snapshot.get("deadline_monotonic")
            try:
                deadline = float(raw_deadline)
            except (TypeError, ValueError):
                deadline = None
            if deadline is not None and math.isfinite(deadline):
                return deadline
            raw_timeout = request_snapshot.get("timeout_seconds", timeout_seconds)
            try:
                snapshot_timeout = float(raw_timeout)
            except (TypeError, ValueError):
                snapshot_timeout = None

        budget = timeout_seconds if timeout_seconds is not None else snapshot_timeout
        if budget is None:
            return None
        try:
            budget = float(budget)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(budget) or budget < 0.0:
            return None

        try:
            start = float(translation_start_monotonic)
            if not math.isfinite(start):
                start = time.monotonic()
        except (TypeError, ValueError):
            start = time.monotonic()
        return start + budget

    def _remaining_custom_ai_request_timeout(self, deadline_monotonic, now=None):
        if deadline_monotonic is None:
            return None
        try:
            deadline = float(deadline_monotonic)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(deadline):
            return None
        if now is None:
            now = time.monotonic()
        try:
            current = float(now)
        except (TypeError, ValueError):
            current = time.monotonic()
        return max(0.0, deadline - current)

    def _custom_ai_failover_deadline_result(self):
        return (
            "Custom AI translation error: request deadline exhausted "
            "before a successful translation."
        )

    def _custom_ai_failover_preflight(
        self,
        translation_sequence,
        deadline_monotonic=None,
        timeout_seconds=None,
    ):
        """Shared stopped/obsolete/deadline gate for failover and race pre-start.

        Returns a dict:
        - action="abort": stop without another candidate; result is the return value
        - action="deadline": budget exhausted; result is a diagnostic timeout message
        - action="continue": proceed with timeout_seconds for this candidate
        """
        if self._abort_stopped_custom_ai_failover(translation_sequence):
            return {"action": "abort", "result": None, "timeout_seconds": 0.0}
        if self._custom_ai_translation_is_obsolete(translation_sequence):
            # Reuse the existing diagnostic log path without double-checking
            # display/start state elsewhere.
            self._abort_obsolete_custom_ai_failover(translation_sequence)
            self._increment_runtime_metric("fallback_obsolete_abort")
            return {"action": "abort", "result": None, "timeout_seconds": 0.0}

        if deadline_monotonic is None:
            return {
                "action": "continue",
                "result": None,
                "timeout_seconds": timeout_seconds,
            }

        remaining = self._remaining_custom_ai_request_timeout(deadline_monotonic)
        if remaining is None:
            return {
                "action": "continue",
                "result": None,
                "timeout_seconds": timeout_seconds,
            }
        if remaining <= 0.0:
            self._increment_runtime_metric("fallback_deadline_exhausted")
            _log_debug(
                "LATENCY: custom_ai failover stopped because request deadline "
                f"was exhausted sequence={translation_sequence}"
            )
            return {
                "action": "deadline",
                "result": self._custom_ai_failover_deadline_result(),
                "timeout_seconds": 0.0,
            }
        return {
            "action": "continue",
            "result": None,
            "timeout_seconds": remaining,
        }

    def _custom_ai_translate_with_failover(
        self,
        primary_profile,
        text,
        source_lang,
        target_lang,
        context,
        keep_linebreaks,
        custom_prompt="",
        latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE,
        stream_callback=None,
        timeout_seconds=None,
        translation_start_monotonic=None,
        translation_sequence=None,
        context_generation=None,
        primary_cache_params=None,
        force_no_reasoning=False,
        request_snapshot=None,
    ):
        failures = []
        primary_id = str(primary_profile.get("id") or "").strip()
        deadline_monotonic = self._resolve_custom_ai_request_deadline_monotonic(
            request_snapshot=request_snapshot,
            timeout_seconds=timeout_seconds,
            translation_start_monotonic=translation_start_monotonic,
        )
        candidate_states = []
        for candidate in self._get_custom_ai_failover_profiles(
            primary_profile,
            force_no_reasoning=force_no_reasoning,
        ):
            candidate_id = str(candidate.get("id") or "").strip()
            if candidate_id and candidate_id == primary_id and primary_cache_params:
                cache_params = dict(primary_cache_params)
            else:
                cache_params = self._cache_params_for_profile(
                    candidate,
                    custom_prompt=custom_prompt,
                    keep_linebreaks=keep_linebreaks,
                    context=context,
                    latency_mode=latency_mode,
                )
            cached_result = self._get_custom_ai_cached_translation(
                text,
                translation_sequence=translation_sequence,
                context_generation=context_generation,
                profile=candidate,
                source_lang=source_lang,
                target_lang=target_lang,
                cache_params=cache_params,
            )
            if cached_result:
                return cached_result
            candidate_states.append((candidate, cache_params))

        for candidate, cache_params in candidate_states:
            decision = self._custom_ai_failover_preflight(
                translation_sequence,
                deadline_monotonic=deadline_monotonic,
                timeout_seconds=timeout_seconds,
            )
            if decision["action"] != "continue":
                return decision["result"]
            candidate_timeout = decision["timeout_seconds"]
            cooldown_seconds = self._custom_ai_profile_cooldown_seconds(candidate)
            if cooldown_seconds > 0:
                _log_debug(
                    "LATENCY: custom_ai failover skipped cooling profile "
                    f"provider={candidate.get('name', 'Custom AI')} "
                    f"seconds={cooldown_seconds:.1f}"
                )
                continue

            request_sequence = self._begin_custom_ai_profile_request(
                candidate,
                "translation",
            )
            self._increment_runtime_metric("fallback_candidates_started")
            try:
                translated_text, usage, duration = self.custom_ai_provider.translate(
                    candidate,
                    text,
                    source_lang,
                    target_lang,
                    custom_prompt=custom_prompt,
                    context=context,
                    keep_linebreaks=keep_linebreaks,
                    latency_mode=latency_mode,
                    stream_callback=(
                        stream_callback
                        if latency_mode == CUSTOM_AI_LATENCY_MODE_STREAM
                        else None
                    ),
                    timeout_seconds=candidate_timeout,
                )
            except Exception as error:
                error_text = self._sanitize_custom_ai_profile_error(
                    error,
                    candidate,
                )
                failures.append(
                    f"{candidate.get('name', 'Custom AI')}: {error_text}"
                )
                self._record_custom_ai_latency_observation(
                    time.monotonic() - (translation_start_monotonic or time.monotonic()),
                    success=False,
                    profile=candidate,
                )
                self._mark_custom_ai_profile_failure(
                    candidate,
                    error_text,
                    request_kind="translation",
                    request_sequence=request_sequence,
                )
                _log_debug(
                    "Custom AI failover translation error: "
                    f"provider={candidate.get('name', 'Custom AI')} "
                    f"{type(error).__name__} - {error_text}"
                )
                decision = self._custom_ai_failover_preflight(
                    translation_sequence,
                    deadline_monotonic=deadline_monotonic,
                    timeout_seconds=timeout_seconds,
                )
                if decision["action"] != "continue":
                    return decision["result"]
                continue

            if translated_text and not self._is_error_message(translated_text):
                available = getattr(
                    self.custom_ai_provider,
                    "mark_profile_available",
                    None,
                )
                if callable(available):
                    available(
                        candidate,
                        request_kind="translation",
                        request_sequence=request_sequence,
                    )
                cache_params = self._cache_params_for_profile(
                    candidate,
                    custom_prompt=custom_prompt,
                    keep_linebreaks=keep_linebreaks,
                    context=context,
                    latency_mode=latency_mode,
                )
                self._log_custom_short_call(
                    "translation",
                    candidate,
                    translated_text,
                    usage,
                    duration,
                )
                self._store_custom_ai_cached_translation(
                    text,
                    source_lang,
                    target_lang,
                    translated_text,
                    [cache_params],
                    profile=candidate,
                )
                self._update_custom_context(
                    text,
                    translated_text,
                    translation_sequence=translation_sequence,
                    context_generation=context_generation,
                )
                _log_debug(
                    "Custom AI failover translation completed "
                    f"provider={candidate.get('name', 'Custom AI')} "
                    f"source {summarize_text_for_log(text)} "
                    f"result {summarize_text_for_log(translated_text)}"
                )
                return self._format_dialog_text(translated_text)

            failures.append(
                f"{candidate.get('name', 'Custom AI')}: returned an invalid translation"
            )
            self._mark_custom_ai_profile_failure(
                candidate,
                "returned an invalid translation",
                request_kind="translation",
                request_sequence=request_sequence,
            )

        if failures:
            _log_debug(
                "Custom AI failover exhausted enabled profiles: "
                + "; ".join(failures)
            )
        if len(failures) == 1:
            return "Custom AI translation error: " + failures[0]
        return (
            "Custom AI translation error: all enabled profiles are unavailable; "
            "requests are paused until a profile cooldown expires."
        )

    def _settle_custom_ai_race_health(
        self,
        future,
        profile,
        request_sequence,
        identity=None,
    ):
        """Update profile health for a finished race candidate.

        Losers cannot cancel already-issued HTTP requests; they only update
        health after completion. Older request sequences cannot overwrite newer
        health events because transport compares request_sequence.
        """
        if identity is not None:
            self._release_custom_ai_race_profile(identity)
        with self._custom_race_state_lock:
            self._custom_race_futures.discard(future)
            self._custom_race_state_lock.notify_all()
        if future.cancelled():
            self._close_custom_ai_provider_if_safe()
            return
        try:
            exc = future.exception()
        except Exception:
            self._close_custom_ai_provider_if_safe()
            return
        if exc is not None:
            if isinstance(exc, concurrent.futures.CancelledError):
                return
            error_text = self._sanitize_custom_ai_profile_error(exc, profile)
            self._mark_custom_ai_profile_failure(
                profile,
                error_text,
                request_kind="translation",
                request_sequence=request_sequence,
            )
            self._close_custom_ai_provider_if_safe()
            return
        available = getattr(
            self.custom_ai_provider,
            "mark_profile_available",
            None,
        )
        if callable(available):
            available(
                profile,
                request_kind="translation",
                request_sequence=request_sequence,
            )
        self._close_custom_ai_provider_if_safe()

    def _reserve_custom_ai_race_candidates(self, candidates, timeout_seconds=None):
        identities = [
            (candidate, self._custom_ai_race_profile_identity(candidate))
            for candidate in candidates
        ]
        try:
            wait_seconds = max(0.0, float(timeout_seconds or 0.0))
        except (TypeError, ValueError):
            wait_seconds = 0.0
        deadline = time.monotonic() + wait_seconds

        with self._custom_race_state_lock:
            while True:
                if self._custom_race_submissions_frozen:
                    return []
                remaining_budget = max(
                    0,
                    int(self._custom_race_candidate_limit)
                    - len(self._custom_race_inflight_profiles),
                )
                reserved = [
                    (candidate, identity)
                    for candidate, identity in identities
                    if identity not in self._custom_race_inflight_profiles
                ][:remaining_budget]
                if reserved:
                    self._custom_race_inflight_profiles.update(
                        identity for _candidate, identity in reserved
                    )
                    return reserved

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return []
                self._custom_race_state_lock.wait(timeout=remaining)

    def _track_custom_ai_race_future(
        self,
        future,
        profile,
        request_sequence,
        identity,
    ):
        with self._custom_race_state_lock:
            self._custom_race_futures.add(future)
        future.add_done_callback(
            lambda settled_future: self._settle_custom_ai_race_health(
                settled_future,
                profile,
                request_sequence,
                identity=identity,
            )
        )

    def _custom_ai_translate_race(
        self,
        active_profile,
        text,
        source_lang,
        target_lang,
        context,
        keep_linebreaks,
        custom_prompt=None,
        timeout_seconds=None,
        force_no_reasoning=False,
    ):
        if custom_prompt is None:
            custom_prompt = getattr(self.app, 'custom_prompt_text', '')
        candidates = self._get_custom_ai_race_profiles(
            active_profile,
            force_no_reasoning=force_no_reasoning,
            include_busy=True,
        )
        reserved_candidates = self._reserve_custom_ai_race_candidates(
            candidates,
            timeout_seconds=timeout_seconds,
        )
        if not reserved_candidates:
            raise TimeoutError(
                "Custom AI race capacity remained busy until the request deadline"
            )
        if len(reserved_candidates) == 1:
            candidate, identity = reserved_candidates[0]
            request_sequence = self._begin_custom_ai_profile_request(
                candidate,
                "translation",
            )
            try:
                translated, usage, duration = (
                    self.custom_ai_provider.translate(
                        candidate,
                        text,
                        source_lang,
                        target_lang,
                        custom_prompt=custom_prompt,
                        context=context,
                        keep_linebreaks=keep_linebreaks,
                        latency_mode=CUSTOM_AI_LATENCY_MODE_RACE,
                        timeout_seconds=timeout_seconds,
                    )
                )
            except Exception as error:
                error_text = self._sanitize_custom_ai_profile_error(
                    error,
                    candidate,
                )
                self._mark_custom_ai_profile_failure(
                    candidate,
                    error_text,
                    request_kind="translation",
                    request_sequence=request_sequence,
                )
                raise ValueError(
                    f"{candidate.get('name', 'Custom AI')}: "
                    f"{error_text}"
                ) from None
            finally:
                self._release_custom_ai_race_profile(identity)
            available = getattr(
                self.custom_ai_provider,
                "mark_profile_available",
                None,
            )
            if callable(available):
                available(
                    candidate,
                    request_kind="translation",
                    request_sequence=request_sequence,
                )
            self._note_custom_ai_race_winner(candidate)
            return (
                translated,
                usage,
                duration,
                self._cache_params_for_profile(
                    candidate,
                    custom_prompt=custom_prompt,
                    keep_linebreaks=keep_linebreaks,
                    context=context,
                    latency_mode=CUSTOM_AI_LATENCY_MODE_RACE,
                ),
                candidate,
            )

        future_to_profile = {}
        errors = []
        for index, (candidate, identity) in enumerate(reserved_candidates):
            request_sequence = self._begin_custom_ai_profile_request(
                candidate,
                "translation",
            )
            try:
                future = self._custom_race_executor.submit(
                    self.custom_ai_provider.translate,
                    candidate,
                    text,
                    source_lang,
                    target_lang,
                    custom_prompt=custom_prompt,
                    context=context,
                    keep_linebreaks=keep_linebreaks,
                    latency_mode=CUSTOM_AI_LATENCY_MODE_RACE,
                    timeout_seconds=timeout_seconds,
                )
            except Exception:
                for _candidate, unsubmitted_identity in reserved_candidates[index:]:
                    self._release_custom_ai_race_profile(unsubmitted_identity)
                raise
            future_to_profile[future] = (candidate, request_sequence)
            self._track_custom_ai_race_future(
                future,
                candidate,
                request_sequence,
                identity,
            )

        for future in concurrent.futures.as_completed(future_to_profile):
            candidate, _request_sequence = future_to_profile[future]
            try:
                translated, usage, duration = future.result()
            except Exception as e:
                error_text = self._sanitize_custom_ai_profile_error(
                    e,
                    candidate,
                )
                errors.append(
                    f"{candidate.get('name', 'Custom AI')}: "
                    f"{error_text}"
                )
                continue
            _log_debug(
                "LATENCY: custom_ai race winner "
                f"profile={candidate.get('name', 'Custom AI')} "
                f"duration={duration:.3f}s candidates={len(reserved_candidates)}"
            )
            self._note_custom_ai_race_winner(candidate)
            for losing_future in future_to_profile:
                if losing_future is not future:
                    losing_future.cancel()
            return (
                translated,
                usage,
                duration,
                self._cache_params_for_profile(
                    candidate,
                    custom_prompt=custom_prompt,
                    keep_linebreaks=keep_linebreaks,
                    context=context,
                    latency_mode=CUSTOM_AI_LATENCY_MODE_RACE,
                ),
                candidate,
            )

        raise ValueError("All Custom AI race endpoints failed. Tried: " + "; ".join(errors))

    def _get_custom_ai_race_profiles(
        self,
        active_profile,
        force_no_reasoning=None,
        include_busy=False,
    ):
        active_profile = self._translation_request_profile(
            active_profile,
            force_no_reasoning=force_no_reasoning,
        )
        active_signature = self._custom_ai_race_signature(active_profile)
        candidates = []
        seen = set()

        def add_candidate(profile):
            if not isinstance(profile, dict):
                return
            if not self._is_custom_ai_translation_failover_eligible(
                profile,
                active_profile,
            ):
                return
            profile = self._translation_request_profile(
                profile,
                force_no_reasoning=force_no_reasoning,
            )
            if self._custom_ai_race_signature(profile) != active_signature:
                return
            identity = self._custom_ai_race_profile_identity(profile)
            if identity in seen:
                return
            seen.add(identity)
            candidates.append(profile)

        for profile in self._iter_custom_ai_translation_failover_source_profiles(
            active_profile
        ):
            add_candidate(profile)

        eligible_candidates = candidates
        if candidates:
            healthy_candidates = []
            for candidate in candidates:
                try:
                    remaining = self._custom_ai_profile_cooldown_seconds(
                        candidate,
                        request_kind="translation",
                    )
                except Exception as cooldown_error:
                    _log_debug(
                        "Custom AI race cooldown check failed: "
                        f"{type(cooldown_error).__name__} - {cooldown_error}"
                    )
                    remaining = 0.0
                if remaining <= 0.0:
                    healthy_candidates.append(candidate)
            eligible_candidates = healthy_candidates or candidates

        with self._custom_race_state_lock:
            busy_profiles = set(self._custom_race_inflight_profiles)
        idle_candidates = [
            candidate
            for candidate in eligible_candidates
            if self._custom_ai_race_profile_identity(candidate)
            not in busy_profiles
        ]
        return eligible_candidates if include_busy else idle_candidates

    def _custom_ai_race_signature(self, profile):
        profile = profile if isinstance(profile, dict) else {}
        return (
            str(profile.get("model") or "").strip(),
            normalize_custom_ai_wire_api(profile.get("wire_api")),
            str(
                self.custom_ai_provider.reasoning_effort_request_contract(
                    profile,
                    "translation",
                )
            ),
            normalize_custom_ai_structured_output_mode(
                profile.get("structured_output_mode")
            ),
        )

    def _custom_ai_race_profile_identity(self, profile):
        profile = profile if isinstance(profile, dict) else {}
        base_url = self._canonical_custom_ai_profile_endpoint(profile)
        return (
            "endpoint",
            base_url,
            *self._custom_ai_race_signature(profile),
            self.custom_ai_provider._credential_scope_key(profile),
        )

    def _canonical_custom_ai_profile_endpoint(self, profile):
        profile = profile if isinstance(profile, dict) else {}
        base_url = str(
            profile.get("base_url") or ""
        ).strip().rstrip("/")
        try:
            if (
                normalize_custom_ai_wire_api(profile.get("wire_api"))
                == "responses"
            ):
                base_url = (
                    self.custom_ai_provider
                    .normalize_responses_url_candidates(base_url)[0]
                )
            else:
                base_url = (
                    self.custom_ai_provider
                    .normalize_chat_completions_url_candidates(base_url)[0]
                )
        except (TypeError, ValueError, IndexError):
            pass
        return self.custom_ai_provider._base_url_cache_key(
            base_url
        )

    def _release_custom_ai_race_profile(self, identity):
        with self._custom_race_state_lock:
            self._custom_race_inflight_profiles.discard(identity)
            self._custom_race_state_lock.notify_all()

    def _cache_params_for_profile(
        self,
        profile,
        current_source=None,
        custom_prompt=None,
        keep_linebreaks=None,
        context=None,
        latency_mode=None,
    ):
        if keep_linebreaks is None:
            keep_linebreaks_var = getattr(self.app, "keep_linebreaks_var", None)
            try:
                keep_linebreaks = bool(keep_linebreaks_var.get()) if keep_linebreaks_var is not None else False
            except Exception:
                keep_linebreaks = False
        if custom_prompt is None:
            custom_prompt = getattr(self.app, "custom_prompt_text", "")
        if context is None:
            context = self._get_custom_context_for_request(
                current_source=current_source,
                profile=profile,
            )
        try:
            request_latency_mode = normalize_custom_ai_latency_mode(
                latency_mode
                if latency_mode is not None
                else self._get_custom_ai_latency_mode()
            )
        except Exception:
            request_latency_mode = CUSTOM_AI_LATENCY_MODE_SAFE
        structured_output_contract = (
            self.custom_ai_provider.structured_output_request_contract(
                profile,
                latency_mode=request_latency_mode,
            )
        )

        return {
            "profile_id": profile.get("id", ""),
            "base_url": self._canonical_custom_ai_profile_endpoint(
                profile
            ),
            "model": profile.get("model", ""),
            "credential_scope": (
                self.custom_ai_provider._credential_scope_key(profile)
            ),
            "wire_api": str(
                profile.get("wire_api") or "chat_completions"
            ).strip().lower(),
            "reasoning_effort": (
                self.custom_ai_provider.reasoning_effort_request_contract(
                    profile,
                    "translation",
                )
            ),
            "structured_output_contract": structured_output_contract,
            "custom_prompt": custom_prompt,
            "keep_linebreaks": keep_linebreaks,
            "context": tuple(context),
        }

    def _custom_ai_display_cache_param_variants(
        self,
        cache_params,
        profile=None,
    ):
        """Return primary + peer display-contract params for cache reuse.

        Successful Custom AI stores are plain normalized translation strings.
        Under AUTO, stream uses ``text`` while safe/race use ``json_schema``,
        so mirroring the peer contract avoids a paid re-translate on mode flip
        without removing contract from the cache key.

        Peer expansion is gated on profile structured_output_mode=AUTO only.
        off always uses text and strict always uses json_schema, so those modes
        must not dual-write or dual-read a peer contract.
        """
        if not isinstance(cache_params, dict):
            return []
        primary = dict(cache_params)
        variants = [primary]
        if not self._custom_ai_display_cache_peer_allowed(profile):
            return variants
        contract = str(primary.get("structured_output_contract") or "").strip()
        if contract == CUSTOM_AI_STRUCTURED_OUTPUT_CONTRACT_TEXT:
            peer_contract = CUSTOM_AI_STRUCTURED_OUTPUT_CONTRACT_JSON_SCHEMA
        elif contract == CUSTOM_AI_STRUCTURED_OUTPUT_CONTRACT_JSON_SCHEMA:
            peer_contract = CUSTOM_AI_STRUCTURED_OUTPUT_CONTRACT_TEXT
        else:
            return variants
        peer = dict(primary)
        peer["structured_output_contract"] = peer_contract
        variants.append(peer)
        return variants

    def _custom_ai_display_cache_peer_allowed(self, profile):
        """Return True only when AUTO can flip text/json_schema by latency mode."""
        if not isinstance(profile, dict):
            return False
        mode = normalize_custom_ai_structured_output_mode(
            profile.get("structured_output_mode")
        )
        return mode == CUSTOM_AI_STRUCTURED_OUTPUT_AUTO

    def _store_custom_ai_cached_translation(
        self,
        text,
        source_lang,
        target_lang,
        translation,
        cache_targets,
        profile=None,
    ):
        """Store a successful translation under primary and peer contracts.

        ``cache_targets`` items may be bare cache-param dicts or
        ``(cache_params, owner_profile)`` pairs. Peer dual-write is gated by
        each target's owner profile (AUTO only). Bare dicts fall back to the
        shared ``profile`` kwarg for single-target call sites.
        """
        if not translation or self._is_error_message(translation):
            return
        seen = set()
        dual_store_count = 0
        for target in cache_targets or ():
            target_params, target_profile = (
                self._normalize_custom_ai_cache_store_target(
                    target,
                    default_profile=profile,
                )
            )
            variants = self._custom_ai_display_cache_param_variants(
                target_params,
                profile=target_profile,
            )
            primary_contract = None
            if variants:
                primary_contract = variants[0].get(
                    "structured_output_contract"
                )
            for variant in variants:
                key = (
                    variant.get("profile_id"),
                    variant.get("base_url"),
                    variant.get("model"),
                    variant.get("credential_scope"),
                    variant.get("wire_api"),
                    variant.get("reasoning_effort"),
                    variant.get("structured_output_contract"),
                    variant.get("custom_prompt"),
                    variant.get("keep_linebreaks"),
                    variant.get("context"),
                )
                if key in seen:
                    continue
                seen.add(key)
                self.unified_cache.store(
                    text,
                    source_lang,
                    target_lang,
                    "custom_ai",
                    translation,
                    **variant,
                )
                if (
                    primary_contract is not None
                    and variant.get("structured_output_contract")
                    != primary_contract
                ):
                    dual_store_count += 1
        if dual_store_count:
            self._increment_runtime_metric(
                "translation_cache_contract_dual_store",
                dual_store_count,
            )

    def _normalize_custom_ai_cache_store_target(
        self,
        target,
        default_profile=None,
    ):
        """Return ``(cache_params, owner_profile)`` for one store target."""
        if (
            isinstance(target, tuple)
            and len(target) == 2
            and isinstance(target[0], dict)
        ):
            return target[0], target[1]
        if isinstance(target, dict):
            return target, default_profile
        return {}, default_profile
