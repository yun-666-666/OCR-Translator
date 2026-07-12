"""HTTP recovery, cooldowns, streaming, and provider operations."""

from html.parser import HTMLParser
import json
import re
import sys
import time

from diagnostic_sanitizer import sanitize_error_text, sanitize_url
from custom_ai_policy import (
    CUSTOM_AI_LATENCY_MODE_NONE,
    CUSTOM_AI_LATENCY_MODE_SAFE,
    CUSTOM_AI_LATENCY_MODE_STREAM,
    CUSTOM_AI_STRUCTURED_OUTPUT_AUTO,
    DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    normalize_custom_ai_latency_mode,
    normalize_custom_ai_reasoning_request_kind,
)


class _HTMLTitleParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._in_title = False
        self.title_parts = []

    def handle_starttag(self, tag, attrs):
        if str(tag).lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if str(tag).lower() == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(str(data))


def _log_debug(message):
    facade = sys.modules.get("custom_ai")
    if facade is not None:
        return facade.log_debug(message)


class CustomAITransportMixin:
    def _get_http_client(self, latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE):
        latency_mode = normalize_custom_ai_latency_mode(latency_mode)
        if latency_mode == CUSTOM_AI_LATENCY_MODE_NONE and self._owns_http_client:
            import requests
            return requests

        if self.http_client is not None:
            return self.http_client

        with self._client_lock:
            if self.http_client is not None:
                return self.http_client

            import requests

            session = requests.Session()
            adapter_class = getattr(getattr(requests, "adapters", None), "HTTPAdapter", None)
            if adapter_class and hasattr(session, "mount"):
                adapter = adapter_class(pool_connections=8, pool_maxsize=16, max_retries=0)
                session.mount("http://", adapter)
                session.mount("https://", adapter)
            self.http_client = session
            self._owns_http_client = True
            _log_debug("LATENCY: Custom AI HTTP session initialized with connection pooling")
            return self.http_client

    def close(self):
        if self.http_client is not None and self._owns_http_client and hasattr(self.http_client, "close"):
            try:
                self.http_client.close()
                _log_debug("LATENCY: Custom AI HTTP session closed")
            except Exception as e:
                _log_debug(f"Custom AI HTTP session close failed: {e}")
        self.http_client = None
        self._owns_http_client = True

    def _post_with_output_limit_fallback(
        self,
        http_client,
        url,
        headers,
        payload,
        profile,
        api_key,
        stream=False,
        request_kind=None,
        timeout_seconds=None,
    ):
        request_payload = payload
        request_kind = normalize_custom_ai_reasoning_request_kind(request_kind)
        reasoning_effort = self._reasoning_effort_mode(profile)
        if (
            request_kind
            and self._reasoning_effort_is_known_unsupported(
                profile,
                request_kind,
                reasoning_effort,
            )
        ):
            request_payload = self._without_reasoning_effort(request_payload)
        if (
            self._structured_output_mode(profile)
            == CUSTOM_AI_STRUCTURED_OUTPUT_AUTO
            and self._structured_output_is_known_unsupported(profile)
        ):
            request_payload = self._without_structured_output(
                request_payload,
            )
        request_payload = self._without_unsupported_output_limit(
            profile,
            request_payload,
        )

        def send(current_payload):
            kwargs = {
                "headers": headers,
                "json": current_payload,
                "timeout": self._request_timeout(timeout_seconds),
            }
            if stream:
                kwargs["stream"] = True
            return http_client.post(url, **kwargs)

        response = send(request_payload)
        output_limit_field = self._output_limit_field(profile)
        pending_reasoning_memory = None
        for _attempt in range(4):
            if (
                "prompt_cache_key" in request_payload
                and self._response_rejects_compatibility_feature(
                    response,
                    api_key,
                    required_markers=("prompt_cache_key",),
                    rejection_markers=(
                        "unsupported",
                        "not supported",
                        "unknown",
                        "unrecognized",
                        "not permitted",
                        "not allowed",
                        "extra input",
                        "extra field",
                    ),
                )
            ):
                self._remember_unsupported_prompt_cache_key(profile)
                request_payload = self._without_prompt_cache_key(
                    request_payload,
                )
                _log_debug(
                    "COMPAT: retrying Custom AI request without unsupported "
                    "prompt_cache_key"
                )
                response = send(request_payload)
                continue

            if (
                request_kind
                and self._payload_has_reasoning_effort(request_payload)
                and self._response_rejects_compatibility_feature(
                    response,
                    api_key,
                    required_markers=(
                        "reasoning",
                        "reasoning_effort",
                        "thinking",
                        "effort",
                    ),
                    rejection_markers=(
                        "does not support parameter",
                        "does not support",
                        "doesn't support",
                        "do not support",
                        "not support",
                        "unsupported parameter",
                        "unknown parameter",
                        "unknown field",
                        "invalid field",
                        "invalid parameter",
                        "unsupported",
                        "not supported",
                        "unrecognized",
                        "not permitted",
                        "not allowed",
                        "extra input",
                        "extra field",
                    ),
                )
            ):
                pending_reasoning_memory = (
                    request_kind,
                    self._payload_reasoning_effort(request_payload, profile),
                )
                request_payload = self._without_reasoning_effort(
                    request_payload,
                )
                _log_debug(
                    "COMPAT: retrying Custom AI request without unsupported "
                    "reasoning effort"
                )
                response = send(request_payload)
                continue

            if (
                self._payload_has_structured_output(request_payload)
                and self._response_rejects_compatibility_feature(
                    response,
                    api_key,
                    required_markers=(
                        "response_format",
                        "json_schema",
                        "json schema",
                        "text.format",
                        '"format"',
                        "'format'",
                        "structured",
                    ),
                    rejection_markers=(
                        "unsupported",
                        "not supported",
                        "unknown",
                        "unrecognized",
                        "not permitted",
                        "not allowed",
                        "extra input",
                        "extra field",
                        "invalid parameter",
                        "unknown parameter",
                    ),
                )
            ):
                if (
                    self._structured_output_mode(profile)
                    != CUSTOM_AI_STRUCTURED_OUTPUT_AUTO
                ):
                    raise ValueError(
                        self._structured_output_error_message(profile)
                    )
                self._remember_unsupported_structured_output(profile)
                request_payload = self._without_structured_output(
                    request_payload,
                )
                _log_debug(
                    "COMPAT: retrying Custom AI request without unsupported "
                    "structured output"
                )
                response = send(request_payload)
                continue

            if (
                output_limit_field in request_payload
                and self._response_rejects_compatibility_feature(
                    response,
                    api_key,
                    required_markers=(output_limit_field.lower(),),
                    rejection_markers=(
                        "unsupported",
                        "not supported",
                        "unknown",
                        "unrecognized",
                        "not permitted",
                        "not allowed",
                        "extra input",
                        "extra field",
                    ),
                    excluded_markers=(
                        " must be ",
                        "less than",
                        "greater than",
                        "maximum",
                        "minimum",
                        "between",
                        "out of range",
                        f"{output_limit_field.lower()} value ",
                    ),
                )
            ):
                self._remember_unsupported_output_limit(profile)
                retry_payload = dict(request_payload)
                retry_payload.pop(output_limit_field, None)
                request_payload = retry_payload
                _log_debug(
                    "COMPAT: retrying Custom AI request without unsupported "
                    f"{output_limit_field}"
                )
                response = send(request_payload)
                continue
            break
        if (
            pending_reasoning_memory
            and int(getattr(response, "status_code", 200) or 200) < 400
        ):
            self._remember_unsupported_reasoning_effort(
                profile,
                pending_reasoning_memory[0],
                pending_reasoning_memory[1],
            )
        return response

    def _response_classification_detail(self, response):
        """Read provider detail only for private boolean classification."""
        detail = ""
        try:
            payload = self._load_response_json(response)
            if isinstance(payload, dict) and "error" in payload:
                error = payload["error"]
                if isinstance(error, dict):
                    detail = error.get("message") or json.dumps(
                        error,
                        ensure_ascii=False,
                    )
                else:
                    detail = str(error)
            elif payload:
                detail = json.dumps(payload, ensure_ascii=False)
        except Exception:
            response_text = str(getattr(response, "text", "") or "").strip()
            detail = (
                self._compact_html_error_detail(response, response_text)
                or response_text
            )
        return detail

    def _response_error_text_for_classification(self, response, api_key):
        """Return sanitized text for private compatibility checks only."""
        return sanitize_error_text(
            self._response_classification_detail(response),
            known_secrets=[api_key],
            max_length=500,
        ).lower()

    def _response_rejects_compatibility_feature(
        self,
        response,
        api_key,
        required_markers,
        rejection_markers,
        excluded_markers=(),
    ):
        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code not in {400, 422}:
            return False
        error_text = self._response_error_text_for_classification(
            response,
            api_key,
        )
        if not any(marker in error_text for marker in required_markers):
            return False
        if any(marker in error_text for marker in excluded_markers):
            return False
        return any(marker in error_text for marker in rejection_markers)

    def _response_has_retry_after(self, response):
        headers = getattr(response, "headers", None) or {}
        if not hasattr(headers, "get"):
            return False
        return bool(
            headers.get("Retry-After")
            or headers.get("retry-after")
        )

    def _should_retry_transient_response(self, response, latency_mode):
        if normalize_custom_ai_latency_mode(latency_mode) == CUSTOM_AI_LATENCY_MODE_NONE:
            return False
        status_code = int(getattr(response, "status_code", 0) or 0)
        return (
            status_code in {502, 504}
            and not self._response_has_retry_after(response)
        )

    def _should_retry_transient_exception(self, error, latency_mode):
        return (
            normalize_custom_ai_latency_mode(latency_mode)
            != CUSTOM_AI_LATENCY_MODE_NONE
            and self._is_transport_reset_error(error)
        )

    def _close_response_quietly(self, response):
        close = getattr(response, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            pass

    def _post_with_transient_recovery(
        self,
        http_client,
        url,
        headers,
        payload,
        profile,
        api_key,
        latency_mode,
        request_kind=None,
        timeout_seconds=None,
    ):
        current_client = http_client
        for attempt in range(2):
            try:
                response = self._post_with_output_limit_fallback(
                    current_client,
                    url,
                    headers,
                    payload,
                    profile,
                    api_key,
                    request_kind=request_kind,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as error:
                if (
                    attempt == 0
                    and self._should_retry_transient_exception(
                        error,
                        latency_mode,
                    )
                ):
                    if self._discard_owned_http_client_for_transport_error(error):
                        current_client = self._get_http_client(latency_mode)
                    _log_debug(
                        "RECOVERY: retrying Custom AI request on the same URL "
                        f"after {type(error).__name__}"
                    )
                    continue
                raise

            if (
                attempt == 0
                and self._should_retry_transient_response(
                    response,
                    latency_mode,
                )
            ):
                status_code = int(
                    getattr(response, "status_code", 0) or 0
                )
                self._close_response_quietly(response)
                _log_debug(
                    "RECOVERY: retrying Custom AI request on the same URL "
                    f"after HTTP {status_code}"
                )
                continue
            return response, current_client

        raise RuntimeError("Transient recovery loop ended without a response")

    def _parse_retry_after_seconds(self, response):
        headers = getattr(response, "headers", None) or {}
        header_value = None
        if hasattr(headers, "get"):
            header_value = headers.get("Retry-After") or headers.get("retry-after")
        if header_value in (None, ""):
            return DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
        try:
            return max(1.0, min(300.0, float(header_value)))
        except Exception:
            return DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS

    def _activate_rate_limit_cooldown(self, profile, response, detail=""):
        status_code = int(getattr(response, "status_code", 0) or 0)
        is_rate_limit = self._looks_like_rate_limit_error(detail)
        is_capacity = self._looks_like_capacity_error(detail)
        if (
            status_code not in {429, 503}
            and not self._response_has_retry_after(response)
            and not is_rate_limit
            and not is_capacity
        ):
            return 0.0

        base_cooldown_seconds = self._parse_retry_after_seconds(response)
        scope = self._cooldown_scope_for_failure(response, detail)
        cache_key = self._cooldown_cache_key(profile, scope)
        if not cache_key:
            return base_cooldown_seconds

        now = time.monotonic()
        with self._rate_limit_lock:
            existing_until = self._rate_limit_cooldowns.get(cache_key, 0.0)
            if existing_until > now:
                cooldown_until = max(
                    existing_until,
                    now + base_cooldown_seconds,
                )
                cooldown_seconds = cooldown_until - now
            else:
                backoff_count = (
                    self._rate_limit_backoff_counts.get(cache_key, 0) + 1
                )
                self._rate_limit_backoff_counts[cache_key] = backoff_count
                backoff_multiplier = 2 ** min(max(0, backoff_count - 1), 8)
                cooldown_seconds = min(
                    300.0,
                    base_cooldown_seconds * backoff_multiplier,
                )
                cooldown_until = now + cooldown_seconds
            self._rate_limit_cooldowns[cache_key] = cooldown_until

        profile_values = profile if isinstance(profile, dict) else {}
        model_name = " ".join(
            str(profile_values.get("model") or "").split()
        )[:80] or "unknown"
        classification = (
            "rate_limit"
            if is_rate_limit
            else "capacity"
            if is_capacity
            else f"http_{status_code}"
        )
        _log_debug(
            "LATENCY: custom_ai provider cooldown activated "
            f"provider={profile_values.get('name', 'Custom AI')} "
            f"scope={scope} model={model_name} "
            f"seconds={cooldown_seconds:.1f} classification={classification}"
        )
        return cooldown_seconds

    def _note_rate_limit_success(self, profile):
        cache_keys = self._cooldown_cache_keys_for_profile(profile)
        with self._rate_limit_lock:
            for cache_key in cache_keys:
                self._rate_limit_backoff_counts.pop(cache_key, None)

    def _profile_unavailable_key(self, profile):
        values = profile if isinstance(profile, dict) else {}
        profile_id = str(values.get("id") or "").strip()
        if profile_id:
            return ("profile", profile_id)
        return ("profile-fallback", *self._cooldown_cache_keys_for_profile(values))

    def mark_profile_unavailable(self, profile, detail="", seconds=60.0):
        try:
            cooldown_seconds = max(1.0, min(300.0, float(seconds)))
        except (TypeError, ValueError):
            cooldown_seconds = 60.0
        cache_key = self._profile_unavailable_key(profile)
        now = time.monotonic()
        with self._rate_limit_lock:
            existing_until = self._profile_unavailable_cooldowns.get(cache_key, 0.0)
            cooldown_until = max(existing_until, now + cooldown_seconds)
            self._profile_unavailable_cooldowns[cache_key] = cooldown_until
        values = profile if isinstance(profile, dict) else {}
        _log_debug(
            "LATENCY: custom_ai profile unavailable "
            f"provider={values.get('name', 'Custom AI')} "
            f"seconds={cooldown_until - now:.1f} "
            f"detail={str(detail or '').strip()[:160]}"
        )
        return cooldown_until - now

    def mark_profile_available(self, profile):
        cache_key = self._profile_unavailable_key(profile)
        with self._rate_limit_lock:
            self._profile_unavailable_cooldowns.pop(cache_key, None)

    def get_cooldown_remaining(self, profile):
        cache_keys = self._cooldown_cache_keys_for_profile(profile)
        unavailable_key = self._profile_unavailable_key(profile)

        with self._rate_limit_lock:
            rate_limit_until = max(
                (
                    self._rate_limit_cooldowns.get(cache_key, 0.0)
                    for cache_key in cache_keys
                ),
                default=0.0,
            )
            profile_unavailable_until = self._profile_unavailable_cooldowns.get(
                unavailable_key,
                0.0,
            )
            cooldown_until = max(rate_limit_until, profile_unavailable_until)

        now = time.monotonic()
        remaining = cooldown_until - now
        if remaining <= 0:
            with self._rate_limit_lock:
                for cache_key in cache_keys:
                    if self._rate_limit_cooldowns.get(cache_key, 0.0) <= now:
                        self._rate_limit_cooldowns.pop(cache_key, None)
                if (
                    self._profile_unavailable_cooldowns.get(unavailable_key, 0.0)
                    <= now
                ):
                    self._profile_unavailable_cooldowns.pop(unavailable_key, None)
            return 0.0
        return remaining

    def _raise_if_rate_limited(self, profile):
        remaining = self.get_cooldown_remaining(profile)
        if remaining > 0:
            raise ValueError(self._rate_limit_message(profile, remaining))

    def translate(
        self,
        profile,
        text,
        source_lang,
        target_lang,
        custom_prompt="",
        context=None,
        keep_linebreaks=False,
        latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE,
        stream_callback=None,
        timeout_seconds=None,
    ):
        latency_mode = normalize_custom_ai_latency_mode(latency_mode)
        payload = self.build_translation_payload(
            profile,
            text,
            source_lang,
            target_lang,
            custom_prompt=custom_prompt,
            context=context,
            keep_linebreaks=keep_linebreaks,
            latency_mode=latency_mode,
            stream=latency_mode == CUSTOM_AI_LATENCY_MODE_STREAM,
        )
        effective_stream_callback = stream_callback
        if (
            latency_mode == CUSTOM_AI_LATENCY_MODE_STREAM
            and callable(stream_callback)
        ):
            effective_stream_callback = (
                self._build_translation_stream_callback(
                    text,
                    stream_callback,
                )
            )

        def request(current_payload):
            if latency_mode == CUSTOM_AI_LATENCY_MODE_STREAM:
                stream_kwargs = {
                    "stream_callback": effective_stream_callback,
                    "latency_mode": latency_mode,
                }
                if timeout_seconds is not None:
                    stream_kwargs["timeout_seconds"] = timeout_seconds
                return self._stream_post(
                    profile,
                    current_payload,
                    request_kind="translation",
                    **stream_kwargs,
                )
            return self._post(
                profile,
                current_payload,
                latency_mode=latency_mode,
                request_kind="translation",
                timeout_seconds=timeout_seconds,
            )

        active_request = request
        active_payload = payload
        response_latency_mode = latency_mode
        response_stream = latency_mode == CUSTOM_AI_LATENCY_MODE_STREAM
        used_stream_to_non_stream_fallback = False
        try:
            response_json, duration = active_request(active_payload)
        except Exception as stream_error:
            if (
                latency_mode != CUSTOM_AI_LATENCY_MODE_STREAM
                or not self._stream_error_should_fallback_to_non_stream(stream_error)
            ):
                raise
            _log_debug(
                "LATENCY: custom_ai stream translation failed transiently; "
                "retrying non-stream request: "
                f"{type(stream_error).__name__} - {stream_error}"
            )
            active_payload = self.build_translation_payload(
                profile,
                text,
                source_lang,
                target_lang,
                custom_prompt=custom_prompt,
                context=context,
                keep_linebreaks=keep_linebreaks,
                latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE,
                stream=False,
            )

            def active_request(current_payload):
                return self._post(
                    profile,
                    current_payload,
                    latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE,
                    request_kind="translation",
                    timeout_seconds=timeout_seconds,
                )

            response_latency_mode = CUSTOM_AI_LATENCY_MODE_SAFE
            response_stream = False
            used_stream_to_non_stream_fallback = True
            response_json, duration = active_request(active_payload)
        if (
            "max_tokens" in active_payload
            and not self._output_limit_is_known_unsupported(profile)
            and self._response_was_output_limited(profile, response_json)
        ):
            retry_payload = dict(active_payload)
            retry_payload.pop("max_tokens", None)
            _log_debug(
                "QUALITY: retrying truncated Custom AI translation "
                "without output limit"
            )
            response_json, retry_duration = active_request(retry_payload)
            duration += retry_duration
        terminal_error = self._translation_terminal_error(
            profile,
            response_json,
        )
        if terminal_error:
            raise ValueError(terminal_error)

        try:
            result = self._normalize_translation_output(
                text,
                self._parse_translation_response_text(
                    profile,
                    response_json,
                    latency_mode=response_latency_mode,
                    stream=response_stream,
                ),
            )
        except ValueError as parse_error:
            empty_responses_output = self._is_empty_responses_output_error(
                profile,
                parse_error,
            )
            should_retry_plain_text = (
                used_stream_to_non_stream_fallback
                and self._structured_output_mode(profile)
                == CUSTOM_AI_STRUCTURED_OUTPUT_AUTO
                and self._payload_has_structured_output(active_payload)
                and "structured translation response" in str(parse_error).casefold()
            )
            if not should_retry_plain_text and not empty_responses_output:
                raise
            retry_without_structured_output = (
                should_retry_plain_text
                or (
                    empty_responses_output
                    and self._structured_output_mode(profile)
                    == CUSTOM_AI_STRUCTURED_OUTPUT_AUTO
                    and self._payload_has_structured_output(active_payload)
                )
            )
            retry_payload = (
                self._without_structured_output(active_payload)
                if retry_without_structured_output
                else dict(active_payload)
            )
            if empty_responses_output:
                retry_contract = (
                    "plain_text"
                    if retry_without_structured_output
                    else "preserved"
                )
                _log_debug(
                    "RECOVERY: custom_ai responses empty output recovery; "
                    f"retry_contract={retry_contract} "
                    f"{self._responses_response_shape_summary(response_json)}"
                )
            else:
                _log_debug(
                    "LATENCY: custom_ai structured stream fallback failed; "
                    "retrying plain-text request: "
                    f"{type(parse_error).__name__} - {parse_error}"
                )
            response_json, retry_duration = active_request(retry_payload)
            duration += retry_duration
            terminal_error = self._translation_terminal_error(
                profile,
                response_json,
            )
            if terminal_error:
                raise ValueError(terminal_error)
            result = self._normalize_translation_output(
                text,
                (
                    self._parse_response_text(profile, response_json)
                    if retry_without_structured_output
                    else self._parse_translation_response_text(
                        profile,
                        response_json,
                        latency_mode=response_latency_mode,
                        stream=response_stream,
                    )
                ),
            )
        return result, self._extract_usage(response_json), duration

    def recognize(
        self,
        profile,
        image_data,
        source_lang,
        keep_linebreaks=False,
        latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE,
        image_detail="auto",
        image_mime_type="image/webp",
        timeout_seconds=None,
    ):
        payload = self.build_ocr_payload(
            profile,
            image_data,
            source_lang,
            keep_linebreaks=keep_linebreaks,
            image_detail=image_detail,
            image_mime_type=image_mime_type,
        )
        response_json, duration = self._post(
            profile,
            payload,
            latency_mode=latency_mode,
            request_kind="ocr",
            timeout_seconds=timeout_seconds,
        )
        result = self._parse_response_text(profile, response_json)
        if keep_linebreaks:
            result = result.replace("\n", "<br>")
        else:
            result = result.replace("\n", " ")
        if not result or "<EMPTY>" in result:
            result = "<EMPTY>"
        return result, self._extract_usage(response_json), duration

    def _get_with_light_retry(self, http_client, url, headers, latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE):
        if normalize_custom_ai_latency_mode(latency_mode) == CUSTOM_AI_LATENCY_MODE_NONE:
            return http_client.get(url, headers=headers, timeout=self.timeout)

        last_error = None
        for attempt in range(2):
            try:
                return http_client.get(url, headers=headers, timeout=self.timeout)
            except Exception as e:
                last_error = e
                if self._discard_owned_http_client_for_transport_error(e):
                    http_client = self._get_http_client(latency_mode)
                if attempt == 0:
                    _log_debug(
                        "LATENCY: custom_ai models GET retry for transient "
                        f"error at {sanitize_url(url)}: {type(e).__name__}"
                    )
        raise last_error

    def _post(
        self,
        profile,
        payload,
        latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE,
        request_kind="translation",
        timeout_seconds=None,
    ):
        latency_mode = normalize_custom_ai_latency_mode(latency_mode)
        http_client = self._get_http_client(latency_mode)
        self._raise_if_rate_limited(profile)
        headers = {
            "Authorization": f"Bearer {profile.get('api_key', '')}",
            "Content-Type": "application/json",
        }
        headers = self._xai_chat_prompt_cache_headers(
            profile,
            headers,
            request_kind,
        )
        if self._uses_responses_api(profile):
            return self._responses_post(
                profile,
                payload,
                http_client,
                headers,
                latency_mode,
                request_kind=request_kind,
                timeout_seconds=timeout_seconds,
            )
        errors = []
        api_key = profile.get("api_key", "")
        request_payload, openrouter_latency_routing = self._prepare_payload_for_profile(profile, payload, latency_mode)
        cache_key, urls, cached_url = self._ordered_candidates(
            profile.get("base_url"),
            self._successful_chat_urls,
            self.normalize_chat_completions_url_candidates,
            latency_mode,
        )
        for url in urls:
            try:
                start = time.monotonic()
                response, http_client = self._post_with_transient_recovery(
                    http_client,
                    url,
                    headers,
                    request_payload,
                    profile,
                    api_key,
                    latency_mode,
                    request_kind=request_kind,
                    timeout_seconds=timeout_seconds,
                )
                duration = time.monotonic() - start
                status_code = int(getattr(response, "status_code", 200) or 200)
                _log_debug(
                    "LATENCY: custom_ai post "
                    f"provider={profile.get('name', 'Custom AI')} "
                    f"url_cached={url == cached_url} "
                    f"openrouter_latency_routing={openrouter_latency_routing} "
                    f"status={status_code} duration={duration:.3f}s"
                )

                if status_code >= 400:
                    self._forget_successful_url(self._successful_chat_urls, cache_key, url)
                    classification_detail = self._response_classification_detail(
                        response
                    )
                    error_message = self._response_error_message(response, url, api_key)
                    cooldown_seconds = self._activate_rate_limit_cooldown(
                        profile,
                        response,
                        classification_detail,
                    )
                    errors.append(error_message)
                    if (
                        cooldown_seconds > 0
                        or self._should_stop_endpoint_fallback(
                            status_code,
                            classification_detail,
                        )
                    ):
                        break
                    continue

                try:
                    response_json = self._load_response_json(response)
                    self._note_rate_limit_success(profile)
                    self._remember_successful_url(self._successful_chat_urls, cache_key, url)
                    return response_json, duration
                except Exception:
                    self._forget_successful_url(self._successful_chat_urls, cache_key, url)
                    errors.append(self._non_json_response_message(response, url, api_key))
                    continue
            except Exception as e:
                self._forget_successful_url(self._successful_chat_urls, cache_key, url)
                if self._discard_owned_http_client_for_transport_error(e):
                    http_client = self._get_http_client(latency_mode)
                errors.append(
                    f"{sanitize_url(url)}: {self._sanitize_error(str(e), api_key)}"
                )

        if len(errors) == 1:
            raise ValueError(errors[0])
        raise ValueError("Unable to call chat completions. Tried: " + "; ".join(errors))

    def _responses_post(
        self,
        profile,
        payload,
        http_client,
        headers,
        latency_mode,
        request_kind=None,
        timeout_seconds=None,
    ):
        api_key = profile.get("api_key", "")
        self._raise_if_rate_limited(profile)
        request_payload = self.build_responses_payload_from_chat_payload(profile, payload, stream=False)
        request_payload = self._with_responses_prompt_cache_key(
            profile,
            request_payload,
            request_kind,
        )
        cache_key, urls, cached_url = self._ordered_candidates(
            profile.get("base_url"),
            self._successful_responses_urls,
            self.normalize_responses_url_candidates,
            latency_mode,
        )
        errors = []
        for url in urls:
            try:
                start = time.monotonic()
                response, http_client = self._post_with_transient_recovery(
                    http_client,
                    url,
                    headers,
                    request_payload,
                    profile,
                    api_key,
                    latency_mode,
                    request_kind=request_kind,
                    timeout_seconds=timeout_seconds,
                )
                duration = time.monotonic() - start
                status_code = int(getattr(response, "status_code", 200) or 200)
                _log_debug(
                    "LATENCY: custom_ai responses "
                    f"provider={profile.get('name', 'Custom AI')} "
                    f"url_cached={url == cached_url} status={status_code} duration={duration:.3f}s"
                )
                if status_code >= 400:
                    self._forget_successful_url(self._successful_responses_urls, cache_key, url)
                    classification_detail = self._response_classification_detail(
                        response
                    )
                    error_message = self._response_error_message(response, url, api_key)
                    cooldown_seconds = self._activate_rate_limit_cooldown(
                        profile,
                        response,
                        classification_detail,
                    )
                    errors.append(error_message)
                    if (
                        cooldown_seconds > 0
                        or self._should_stop_endpoint_fallback(
                            status_code,
                            classification_detail,
                        )
                    ):
                        break
                    continue
                try:
                    response_json = self._load_response_json(response)
                    self._note_rate_limit_success(profile)
                    self._remember_successful_url(self._successful_responses_urls, cache_key, url)
                    return response_json, duration
                except Exception:
                    self._forget_successful_url(self._successful_responses_urls, cache_key, url)
                    errors.append(self._non_json_response_message(response, url, api_key))
                    continue
            except Exception as e:
                self._forget_successful_url(self._successful_responses_urls, cache_key, url)
                if self._discard_owned_http_client_for_transport_error(e):
                    http_client = self._get_http_client(latency_mode)
                errors.append(
                    f"{sanitize_url(url)}: {self._sanitize_error(str(e), api_key)}"
                )

        if len(errors) == 1:
            raise ValueError(errors[0])
        raise ValueError("Unable to call responses API. Tried: " + "; ".join(errors))

    def _stream_post(
        self,
        profile,
        payload,
        stream_callback=None,
        latency_mode=CUSTOM_AI_LATENCY_MODE_STREAM,
        request_kind=None,
        timeout_seconds=None,
    ):
        latency_mode = normalize_custom_ai_latency_mode(latency_mode)
        http_client = self._get_http_client(latency_mode)
        self._raise_if_rate_limited(profile)
        headers = {
            "Authorization": f"Bearer {profile.get('api_key', '')}",
            "Content-Type": "application/json",
        }
        headers = self._xai_chat_prompt_cache_headers(
            profile,
            headers,
            request_kind,
        )
        if self._uses_responses_api(profile):
            return self._stream_responses_post(
                profile,
                payload,
                stream_callback=stream_callback,
                http_client=http_client,
                headers=headers,
                latency_mode=latency_mode,
                request_kind=request_kind,
                timeout_seconds=timeout_seconds,
            )
        api_key = profile.get("api_key", "")
        request_payload, openrouter_latency_routing = self._prepare_payload_for_profile(
            profile,
            payload,
            latency_mode,
            stream=True,
        )
        cache_key, urls, cached_url = self._ordered_candidates(
            profile.get("base_url"),
            self._successful_chat_urls,
            self.normalize_chat_completions_url_candidates,
            latency_mode,
        )
        errors = []

        for url in urls:
            try:
                start = time.monotonic()
                response = self._post_with_output_limit_fallback(
                    http_client,
                    url,
                    headers,
                    request_payload,
                    profile,
                    api_key,
                    stream=True,
                    request_kind=request_kind,
                    timeout_seconds=timeout_seconds,
                )
                status_code = int(getattr(response, "status_code", 200) or 200)
                _log_debug(
                    "LATENCY: custom_ai stream "
                    f"provider={profile.get('name', 'Custom AI')} "
                    f"url_cached={url == cached_url} "
                    f"openrouter_latency_routing={openrouter_latency_routing} "
                    f"status={status_code}"
                )
                if status_code >= 400:
                    self._forget_successful_url(self._successful_chat_urls, cache_key, url)
                    classification_detail = self._response_classification_detail(
                        response
                    )
                    error_message = self._response_error_message(response, url, api_key)
                    cooldown_seconds = self._activate_rate_limit_cooldown(
                        profile,
                        response,
                        classification_detail,
                    )
                    errors.append(error_message)
                    if (
                        cooldown_seconds > 0
                        or self._should_stop_endpoint_fallback(
                            status_code,
                            classification_detail,
                        )
                    ):
                        break
                    continue
                if hasattr(response, "raise_for_status"):
                    response.raise_for_status()
                response_json = self._parse_streaming_chat_response(response, stream_callback)
                duration = time.monotonic() - start
                self._note_rate_limit_success(profile)
                self._remember_successful_url(self._successful_chat_urls, cache_key, url)
                return response_json, duration
            except Exception as e:
                self._forget_successful_url(self._successful_chat_urls, cache_key, url)
                if self._discard_owned_http_client_for_transport_error(e):
                    http_client = self._get_http_client(latency_mode)
                errors.append(
                    f"{sanitize_url(url)}: {self._sanitize_error(str(e), api_key)}"
                )

        if len(errors) == 1:
            raise ValueError(errors[0])
        raise ValueError("Unable to call streaming chat completions. Tried: " + "; ".join(errors))

    def _stream_responses_post(
        self,
        profile,
        payload,
        stream_callback=None,
        http_client=None,
        headers=None,
        latency_mode=CUSTOM_AI_LATENCY_MODE_STREAM,
        request_kind=None,
        timeout_seconds=None,
    ):
        latency_mode = normalize_custom_ai_latency_mode(latency_mode)
        http_client = http_client or self._get_http_client(latency_mode)
        self._raise_if_rate_limited(profile)
        headers = headers or {
            "Authorization": f"Bearer {profile.get('api_key', '')}",
            "Content-Type": "application/json",
        }
        api_key = profile.get("api_key", "")
        request_payload = self.build_responses_payload_from_chat_payload(profile, payload, stream=True)
        request_payload = self._with_responses_prompt_cache_key(
            profile,
            request_payload,
            request_kind,
        )
        cache_key, urls, cached_url = self._ordered_candidates(
            profile.get("base_url"),
            self._successful_responses_urls,
            self.normalize_responses_url_candidates,
            latency_mode,
        )
        errors = []
        for url in urls:
            try:
                start = time.monotonic()
                response = self._post_with_output_limit_fallback(
                    http_client,
                    url,
                    headers,
                    request_payload,
                    profile,
                    api_key,
                    stream=True,
                    request_kind=request_kind,
                    timeout_seconds=timeout_seconds,
                )
                status_code = int(getattr(response, "status_code", 200) or 200)
                _log_debug(
                    "LATENCY: custom_ai responses stream "
                    f"provider={profile.get('name', 'Custom AI')} "
                    f"url_cached={url == cached_url} status={status_code}"
                )
                if status_code >= 400:
                    self._forget_successful_url(self._successful_responses_urls, cache_key, url)
                    classification_detail = self._response_classification_detail(
                        response
                    )
                    error_message = self._response_error_message(response, url, api_key)
                    cooldown_seconds = self._activate_rate_limit_cooldown(
                        profile,
                        response,
                        classification_detail,
                    )
                    errors.append(error_message)
                    if (
                        cooldown_seconds > 0
                        or self._should_stop_endpoint_fallback(
                            status_code,
                            classification_detail,
                        )
                    ):
                        break
                    continue
                if hasattr(response, "raise_for_status"):
                    response.raise_for_status()
                response_json = self._parse_streaming_responses_response(response, stream_callback)
                duration = time.monotonic() - start
                self._note_rate_limit_success(profile)
                self._remember_successful_url(self._successful_responses_urls, cache_key, url)
                return response_json, duration
            except Exception as e:
                self._forget_successful_url(self._successful_responses_urls, cache_key, url)
                if self._discard_owned_http_client_for_transport_error(e):
                    http_client = self._get_http_client(latency_mode)
                errors.append(
                    f"{sanitize_url(url)}: {self._sanitize_error(str(e), api_key)}"
                )
        if len(errors) == 1:
            raise ValueError(errors[0])
        raise ValueError("Unable to call streaming responses. Tried: " + "; ".join(errors))

    def _parse_streaming_responses_response(self, response, stream_callback=None):
        accumulated = ""
        usage = None
        event_type = None
        terminal_event_type = None
        terminal_response = None
        for raw_line in self._iter_utf8_response_lines(response):
            if not raw_line:
                continue
            line = str(raw_line).strip()
            if not line:
                continue
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
                continue
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk, dict) and isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            chunk_type = str(chunk.get("type") or event_type or "").strip()
            if chunk_type in {
                "response.completed",
                "response.incomplete",
                "response.failed",
            }:
                response_obj = chunk.get("response")
                if isinstance(response_obj, dict):
                    terminal_response = response_obj
                    if isinstance(response_obj.get("usage"), dict):
                        usage = response_obj["usage"]
                    if not accumulated:
                        try:
                            accumulated = self.parse_responses_response(response_obj)
                        except Exception:
                            pass
                terminal_event_type = chunk_type
                break
            if chunk_type in {"response.output_text.delta", "response.output_text"}:
                delta = chunk.get("delta")
                if delta is None:
                    delta = chunk.get("text")
                if delta is None and isinstance(chunk.get("output_text"), str):
                    delta = chunk["output_text"]
                if delta:
                    accumulated += str(delta)
                    if stream_callback:
                        try:
                            stream_callback(accumulated)
                        except Exception as e:
                            _log_debug(f"Custom AI stream callback failed: {e}")
                continue
            if chunk_type == "response.output_text.done":
                text = chunk.get("text") or chunk.get("output_text") or ""
                if text and not accumulated:
                    accumulated = str(text)
                    if stream_callback:
                        try:
                            stream_callback(accumulated)
                        except Exception as e:
                            _log_debug(f"Custom AI stream callback failed: {e}")
        if (
            not accumulated
            and terminal_event_type != "response.failed"
        ):
            raise ValueError("Streaming Responses API response did not contain output text")
        result = {"output_text": accumulated}
        if usage:
            result["usage"] = usage
        if terminal_response is not None:
            if terminal_response.get("status"):
                result["status"] = terminal_response["status"]
            incomplete_details = terminal_response.get("incomplete_details")
            if isinstance(incomplete_details, dict):
                result["incomplete_details"] = incomplete_details
            if terminal_response.get("error") is not None:
                result["error"] = terminal_response["error"]
        if (
            terminal_event_type == "response.incomplete"
            and "status" not in result
        ):
            result["status"] = "incomplete"
        elif (
            terminal_event_type == "response.failed"
            and "status" not in result
        ):
            result["status"] = "failed"
        return result

    def _parse_streaming_chat_response(self, response, stream_callback=None):
        accumulated = ""
        usage = None
        finish_reason = None
        for raw_line in self._iter_utf8_response_lines(response):
            if not raw_line:
                continue
            line = str(raw_line).strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk, dict) and isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices") if isinstance(chunk, dict) else None
            if not choices:
                continue
            choice = choices[0] if isinstance(choices[0], dict) else {}
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            content = delta.get("content")
            if content is None and isinstance(choice.get("message"), dict):
                content = choice["message"].get("content")
            if not content:
                continue
            accumulated += str(content)
            if stream_callback:
                try:
                    stream_callback(accumulated)
                except Exception as e:
                    _log_debug(f"Custom AI stream callback failed: {e}")
        if not accumulated:
            raise ValueError("Streaming API response did not contain message content")
        result_choice = {"message": {"content": accumulated}}
        if finish_reason is not None:
            result_choice["finish_reason"] = finish_reason
        result = {"choices": [result_choice]}
        if usage:
            result["usage"] = usage
        return result

    def _compact_html_error_detail(self, response, text):
        text = str(text or "").strip()
        headers = getattr(response, "headers", None) or {}
        content_type = ""
        if hasattr(headers, "get"):
            content_type = str(
                headers.get("Content-Type")
                or headers.get("content-type")
                or ""
            ).lower()
        leading = text.lstrip().lower()
        if (
            "text/html" not in content_type
            and not leading.startswith(("<!doctype html", "<html"))
        ):
            return None

        parser = _HTMLTitleParser()
        try:
            parser.feed(text[:16384])
        except Exception:
            pass
        title = " ".join("".join(parser.title_parts).split())
        if title:
            return f"Upstream HTML error page: {title[:160]}"
        return "Upstream returned an HTML error page"

    def _response_error_message(self, response, url, api_key):
        status_code = int(getattr(response, "status_code", 0) or 0)
        safe_url = sanitize_url(url)
        details = []
        try:
            payload = self._load_response_json(response)
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                for field_name in ("type", "code"):
                    value = error.get(field_name)
                    if not isinstance(value, (str, int, float, bool)):
                        continue
                    safe_value = self._safe_error_identifier(
                        value,
                        api_key,
                    )
                    if safe_value:
                        details.append(f"{field_name}={safe_value}")
        except Exception:
            pass

        message = (
            f"Chat completions request failed (HTTP {status_code}) at {safe_url}"
        )
        if details:
            return f"{message}: {', '.join(details)}"
        return message

    def _safe_error_identifier(self, value, api_key):
        sanitized = sanitize_error_text(
            value,
            known_secrets=[api_key],
            max_length=0,
        )
        if re.fullmatch(r"[A-Za-z0-9_.:]{1,80}", sanitized):
            return sanitized
        return ""

    def _non_json_response_message(self, response, url, api_key):
        text = str(getattr(response, "text", "") or "").strip()
        safe_url = sanitize_url(url)
        if self._compact_html_error_detail(response, text) is not None:
            return (
                f"Chat completions response from {safe_url} was non-JSON. "
                "Upstream returned an HTML error page."
            )
        return (
            f"Chat completions response from {safe_url} was non-JSON or empty."
        )

    def _sanitize_error(self, message, api_key):
        safe_message = sanitize_error_text(
            message,
            known_secrets=[api_key],
            max_length=500,
        )
        if self._is_tls_eof_error(message):
            return (
                "TLS/SSL connection was closed by the server or proxy before a response was received. "
                "This usually means the API URL, network route, or relay endpoint rejected the HTTPS connection. "
                f"Sanitized error: {safe_message}"
            )
        return safe_message

    def _is_tls_eof_error(self, message):
        lowered = str(message).lower()
        return "ssleoferror" in lowered or "unexpected_eof_while_reading" in lowered

    def _is_transport_reset_error(self, error):
        lowered = str(error).lower()
        return any(
            marker in lowered
            for marker in [
                "ssleoferror",
                "unexpected_eof_while_reading",
                "connection reset",
                "remote end closed connection",
                "protocol violation",
            ]
        )

    def _discard_owned_http_client_for_transport_error(self, error):
        if not self._owns_http_client or not self._is_transport_reset_error(error):
            return False
        self.close()
        _log_debug(f"LATENCY: discarded Custom AI HTTP session after transport error: {type(error).__name__}")
        return True
