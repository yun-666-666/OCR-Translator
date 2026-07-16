"""Response normalization, parsing, model listing, and usage extraction."""

import json
import sys
import time

from custom_ai_policy import (
    CUSTOM_AI_LATENCY_MODE_SAFE,
    CUSTOM_AI_STRUCTURED_OUTPUT_CONTRACT_JSON_SCHEMA,
    TRANSLATION_OUTPUT_PREAMBLES,
    TRANSLATION_OUTPUT_WRAPPER_LABELS,
    normalize_custom_ai_latency_mode,
)


def _log_debug(message):
    facade = sys.modules.get("custom_ai")
    if facade is not None:
        return facade.log_debug(message)


class CustomAIRequestsMixin:
    def _normalize_translation_output(self, source_text, output_text):
        normalized = str(output_text or "").lstrip("\ufeff").strip()
        if not normalized:
            raise ValueError("Translation response was empty")

        original = normalized
        source = str(source_text or "").strip()
        source_lines = source.splitlines()
        source_is_fenced = (
            len(source_lines) >= 2
            and source_lines[0].strip().startswith("```")
            and source_lines[-1].strip() == "```"
        )

        lines = normalized.splitlines()
        if (
            not source_is_fenced
            and len(lines) >= 2
            and lines[0].strip().startswith("```")
            and lines[-1].strip() == "```"
        ):
            fence_tag = lines[0].strip()[3:].strip()
            valid_fence_tag = (
                not fence_tag
                or all(
                    char.isalnum() or char in {"_", "+", "-"}
                    for char in fence_tag
                )
            )
            if valid_fence_tag:
                normalized = "\n".join(lines[1:-1]).strip()

        source_first_line = (
            source_lines[0].strip().casefold()
            if source_lines
            else ""
        )
        lines = normalized.splitlines()
        if normalized.strip().casefold() in TRANSLATION_OUTPUT_WRAPPER_LABELS:
            normalized = ""
        elif (
            len(lines) >= 2
            and lines[0].strip().casefold()
            in TRANSLATION_OUTPUT_WRAPPER_LABELS
            and source_first_line not in TRANSLATION_OUTPUT_WRAPPER_LABELS
        ):
            candidate = "\n".join(lines[1:]).strip()
            if candidate:
                normalized = candidate

        normalized_casefold = normalized.casefold()
        source_casefold = source.casefold()
        for preamble in TRANSLATION_OUTPUT_PREAMBLES:
            if normalized_casefold == preamble:
                if source_casefold != preamble:
                    normalized = ""
                break
            if (
                normalized_casefold.startswith(preamble)
                and not source_casefold.startswith(preamble)
            ):
                candidate = normalized[len(preamble):].lstrip()
                if candidate:
                    normalized = candidate
                break

        normalized = normalized.strip()
        if not normalized:
            raise ValueError(
                "Translation response was empty after output normalization"
            )
        if normalized != original:
            _log_debug(
                "QUALITY: removed a clear wrapper from Custom AI translation output"
            )
        return normalized

    def _normalize_translation_stream_partial(
        self,
        source_text,
        partial_text,
    ):
        raw = str(partial_text or "").lstrip("\ufeff")
        if not raw:
            return None

        source = str(source_text or "").strip()
        source_lines = source.splitlines()
        source_is_fenced = (
            len(source_lines) >= 2
            and source_lines[0].strip().startswith("```")
            and source_lines[-1].strip() == "```"
        )
        if source_is_fenced:
            return raw

        candidate = raw.lstrip()
        candidate_casefold = candidate.casefold()
        source_casefold = source.casefold()
        source_first_line = (
            source_lines[0].strip().casefold()
            if source_lines
            else ""
        )

        if "```".startswith(candidate) and len(candidate) < 3:
            return None
        if candidate.startswith("```"):
            newline_index = candidate.find("\n")
            if newline_index < 0:
                fence_tag = candidate[3:].strip()
                if (
                    len(fence_tag) <= 24
                    and all(
                        char.isalnum() or char in {"_", "+", "-"}
                        for char in fence_tag
                    )
                ):
                    return None
                return raw

            fence_tag = candidate[3:newline_index].strip()
            valid_fence_tag = (
                not fence_tag
                or all(
                    char.isalnum() or char in {"_", "+", "-"}
                    for char in fence_tag
                )
            )
            if valid_fence_tag:
                body = candidate[newline_index + 1:].rstrip()
                if body.endswith("```"):
                    body = body[:-3].rstrip()
                else:
                    trailing_backticks = len(body) - len(body.rstrip("`"))
                    if trailing_backticks in {1, 2}:
                        body = body[:-trailing_backticks].rstrip()
                return body or None

        active_preambles = tuple(
            preamble
            for preamble in TRANSLATION_OUTPUT_PREAMBLES
            if not source_casefold.startswith(preamble)
        )
        if any(
            preamble.startswith(candidate_casefold)
            for preamble in active_preambles
        ):
            return None
        for preamble in active_preambles:
            if candidate_casefold.startswith(preamble):
                remainder = candidate[len(preamble):].lstrip()
                return remainder or None

        active_labels = tuple(
            label
            for label in TRANSLATION_OUTPUT_WRAPPER_LABELS
            if source_first_line != label
        )
        if any(
            label.startswith(candidate_casefold)
            for label in active_labels
        ):
            return None
        for label in active_labels:
            if not candidate_casefold.startswith(label):
                continue
            remainder = candidate[len(label):]
            if not remainder or remainder == "\r":
                return None
            if remainder.startswith("\r\n"):
                cleaned = remainder[2:].lstrip()
                return cleaned or None
            if remainder.startswith("\n"):
                cleaned = remainder[1:].lstrip()
                return cleaned or None
            return raw

        return raw

    def _build_translation_stream_callback(
        self,
        source_text,
        stream_callback,
    ):
        last_emitted = [None]

        def filtered_callback(partial_text):
            normalized = self._normalize_translation_stream_partial(
                source_text,
                partial_text,
            )
            if not normalized or normalized == last_emitted[0]:
                return
            last_emitted[0] = normalized
            stream_callback(normalized)

        return filtered_callback

    def _api_error_message(self, response_json):
        if not isinstance(response_json, dict):
            return None
        if "error" in response_json:
            return self._format_api_error_value(response_json.get("error"))
        message = response_json.get("message") or response_json.get("detail")
        if message and (response_json.get("code") or response_json.get("status") in {"error", "failed"}):
            code = str(response_json.get("code") or "").strip()
            message = str(message).strip()
            return f"{code}: {message}" if code else message
        return None

    def _format_api_error_value(self, error):
        if error is None:
            return None
        if isinstance(error, dict):
            message = error.get("message") or error.get("detail") or error.get("code")
            return str(message).strip() if message else json.dumps(error, ensure_ascii=False)
        message = str(error).strip()
        return message or None

    def parse_chat_response(self, response_json):
        if not isinstance(response_json, dict):
            raise ValueError("Invalid API response")
        error_message = self._api_error_message(response_json)
        if error_message:
            raise ValueError(error_message)
        choices = response_json.get("choices")
        if not choices:
            raise ValueError("API response did not contain choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not content:
            raise ValueError("API response did not contain message content")
        return str(content).strip()

    def parse_responses_response(self, response_json):
        if not isinstance(response_json, dict):
            raise ValueError("Invalid API response")
        error_message = self._api_error_message(response_json)
        if error_message:
            raise ValueError(error_message)
        output_text = response_json.get("output_text")
        if output_text:
            normalized_output_text = str(output_text).strip()
            if normalized_output_text:
                return normalized_output_text

        chunks = []
        for output_item in response_json.get("output", []) or []:
            if not isinstance(output_item, dict):
                continue
            for content_item in output_item.get("content", []) or []:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") in {"output_text", "text"}:
                    text = content_item.get("text")
                    if text:
                        chunks.append(str(text))
        content = "".join(chunks).strip()
        if not content:
            raise ValueError("Responses API response did not contain output text")
        return content

    def _responses_response_shape_summary(self, response_json):
        if not isinstance(response_json, dict):
            return f"response_type={type(response_json).__name__}"

        known_statuses = {
            "cancelled",
            "completed",
            "failed",
            "in_progress",
            "incomplete",
            "queued",
        }
        status = str(response_json.get("status") or "missing").strip().lower()
        if status not in known_statuses and status != "missing":
            status = "other"

        incomplete_details = response_json.get("incomplete_details")
        incomplete_reason = "none"
        if isinstance(incomplete_details, dict):
            raw_reason = str(incomplete_details.get("reason") or "").strip().lower()
            if raw_reason in {"content_filter", "max_output_tokens"}:
                incomplete_reason = raw_reason
            elif raw_reason:
                incomplete_reason = "other"

        output_items = response_json.get("output")
        output_items = output_items if isinstance(output_items, list) else []
        message_items = 0
        reasoning_items = 0
        content_items = 0
        top_level_output_text = response_json.get("output_text")
        output_text_items = (
            1
            if top_level_output_text and str(top_level_output_text).strip()
            else 0
        )
        refusal_items = 0
        for output_item in output_items:
            if not isinstance(output_item, dict):
                continue
            item_type = output_item.get("type")
            if item_type == "message":
                message_items += 1
            elif item_type == "reasoning":
                reasoning_items += 1
            content = output_item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                content_items += 1
                content_type = content_item.get("type")
                if content_type in {"output_text", "text"}:
                    output_text_items += 1
                elif content_type == "refusal":
                    refusal_items += 1

        usage = response_json.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        output_details = usage.get("output_tokens_details")
        output_details = output_details if isinstance(output_details, dict) else {}

        def safe_count(value):
            try:
                return max(0, int(value or 0))
            except (OverflowError, TypeError, ValueError):
                return 0

        return (
            f"status={status} incomplete_reason={incomplete_reason} "
            f"output_items={len(output_items)} message_items={message_items} "
            f"reasoning_items={reasoning_items} content_items={content_items} "
            f"output_text_items={output_text_items} refusal_items={refusal_items} "
            f"output_tokens={safe_count(usage.get('output_tokens'))} "
            f"reasoning_tokens={safe_count(output_details.get('reasoning_tokens'))}"
        )

    def _is_empty_responses_output_error(self, profile, error):
        return (
            self._uses_responses_api(profile)
            and str(error).strip()
            == "Responses API response did not contain output text"
        )

    def _parse_response_text(self, profile, response_json):
        if self._uses_responses_api(profile):
            return self.parse_responses_response(response_json)
        return self.parse_chat_response(response_json)

    def _parse_structured_translation_output(self, response_text):
        try:
            payload = json.loads(str(response_text or "").strip())
        except json.JSONDecodeError as error:
            raise ValueError(
                "Structured translation response was not valid JSON"
            ) from error
        if not isinstance(payload, dict):
            raise ValueError(
                "Structured translation response was not a JSON object"
            )
        translation = payload.get("translation")
        if not isinstance(translation, str):
            raise ValueError(
                "Structured translation response missing string translation"
            )
        translation = translation.strip()
        if not translation:
            raise ValueError(
                "Structured translation response contained empty translation"
            )
        return translation

    def _parse_translation_response_text(
        self,
        profile,
        response_json,
        latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE,
        stream=False,
    ):
        response_text = self._parse_response_text(profile, response_json)
        if (
            self.structured_output_request_contract(
                profile,
                latency_mode=latency_mode,
                stream=stream,
            )
            == CUSTOM_AI_STRUCTURED_OUTPUT_CONTRACT_JSON_SCHEMA
        ):
            return self._parse_structured_translation_output(response_text)
        return response_text

    def _load_response_json(self, response):
        content = getattr(response, "content", None)
        if isinstance(content, (bytes, bytearray)) and bytes(content).strip():
            try:
                return json.loads(bytes(content).decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
        if isinstance(content, str) and content.strip():
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass
        try:
            return response.json()
        except ValueError:
            if self._response_looks_like_sse(response):
                return self._parse_streaming_chat_response(response)
            raise

    def _response_looks_like_sse(self, response):
        content = getattr(response, "content", None)
        if isinstance(content, (bytes, bytearray)):
            return bytes(content).lstrip().startswith(b"data:")
        if isinstance(content, str):
            return content.lstrip().startswith("data:")
        text = getattr(response, "text", "")
        return isinstance(text, str) and text.lstrip().startswith("data:")

    def _iter_utf8_response_lines(self, response):
        try:
            iterator = response.iter_lines(decode_unicode=False)
        except TypeError:
            iterator = response.iter_lines()
        for raw_line in iterator:
            if isinstance(raw_line, bytes):
                yield raw_line.decode("utf-8-sig", errors="replace")
            else:
                yield str(raw_line)

    def test_profile(self, profile, latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE):
        payload = {
            "model": profile["model"],
            "messages": [{"role": "user", "content": "Reply with OK only."}],
            "temperature": 0,
            "max_tokens": 8,
        }
        response_json, duration = self._post(profile, payload, latency_mode=latency_mode)
        return self._parse_response_text(profile, response_json), duration

    def parse_models_response(self, response_json):
        if not isinstance(response_json, (dict, list)):
            raise ValueError("Invalid models response")
        error_message = self._api_error_message(response_json)
        if error_message:
            raise ValueError(error_message)
        if isinstance(response_json, list):
            data = response_json
        else:
            data = response_json.get("data")
            if data is None:
                data = response_json.get("models")
            if isinstance(data, dict):
                data = data.get("data") or data.get("models")
        if not isinstance(data, list):
            raise ValueError("Models response did not contain a data list")
        models = []
        for item in data:
            if isinstance(item, str):
                model_name = item
            elif isinstance(item, dict):
                model_name = item.get("id") or item.get("name") or item.get("model") or item.get("display_name")
            else:
                continue
            if model_name:
                model_name = str(model_name).strip()
                if model_name and model_name not in models:
                    models.append(model_name)
        if not models:
            raise ValueError("Models response did not contain model names")
        return models

    def fetch_models(self, profile, latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE):
        latency_mode = normalize_custom_ai_latency_mode(latency_mode)
        http_client = self._get_http_client(latency_mode)
        headers = {
            "Authorization": f"Bearer {profile.get('api_key', '')}",
            "Content-Type": "application/json",
        }
        errors = []
        cache_store = self._successful_responses_urls if self._uses_responses_api(profile) else self._successful_models_urls
        cache_key, urls, cached_url = self._ordered_candidates(
            profile.get("base_url"),
            cache_store,
            self.normalize_models_url_candidates,
            latency_mode,
        )
        for url in urls:
            try:
                start = time.monotonic()
                response = self._get_with_light_retry(http_client, url, headers, latency_mode)
                duration = time.monotonic() - start
                status_code = int(getattr(response, "status_code", 200) or 200)
                _log_debug(
                    "LATENCY: custom_ai models "
                    f"provider={profile.get('name', 'Custom AI')} "
                    f"url_cached={url == cached_url} status={status_code} duration={duration:.3f}s"
                )
                if status_code >= 400:
                    self._forget_successful_url(cache_store, cache_key, url)
                    try:
                        error_payload = self._load_response_json(response)
                        self.parse_models_response(error_payload)
                    except Exception as payload_error:
                        raise ValueError(str(payload_error))
                response.raise_for_status()
                self._remember_successful_url(cache_store, cache_key, url)
                return self.parse_models_response(self._load_response_json(response))
            except Exception as e:
                errors.append(f"{url}: {self._sanitize_error(str(e), profile.get('api_key', ''))}")
        if self._uses_responses_api(profile):
            configured_model = str(profile.get("model") or "").strip()
            if configured_model:
                _log_debug(
                    "LATENCY: custom_ai models fallback to configured model "
                    f"provider={profile.get('name', 'Custom AI')} model={configured_model}"
                )
                return [configured_model]
        raise ValueError("Unable to fetch model list. Tried: " + "; ".join(errors))

    def _extract_usage(self, response_json):
        usage = response_json.get("usage") if isinstance(response_json, dict) else None
        if not isinstance(usage, dict):
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_prompt_tokens": 0,
            }
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        prompt_details = (
            usage.get("prompt_tokens_details")
            or usage.get("input_tokens_details")
            or {}
        )
        completion_details = (
            usage.get("completion_tokens_details")
            or usage.get("output_tokens_details")
            or {}
        )
        cached_prompt_tokens = (
            int(prompt_details.get("cached_tokens") or 0)
            if isinstance(prompt_details, dict)
            else 0
        )
        cached_input_ratio = (
            cached_prompt_tokens / prompt_tokens
            if prompt_tokens > 0
            else 0.0
        )
        reasoning_tokens = None
        if (
            isinstance(completion_details, dict)
            and "reasoning_tokens" in completion_details
        ):
            try:
                reasoning_tokens = int(
                    completion_details.get("reasoning_tokens") or 0
                )
            except (TypeError, ValueError):
                reasoning_tokens = 0
        cost_usd = None
        try:
            cost_ticks = usage.get("cost_in_usd_ticks")
            if cost_ticks is not None:
                cost_usd = float(cost_ticks) / 10_000_000_000
        except (TypeError, ValueError):
            cost_usd = None
        normalized_usage = {
            "prompt_tokens": prompt_tokens,
            "input_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens") or (prompt_tokens + completion_tokens)),
            "cached_prompt_tokens": cached_prompt_tokens,
            "cached_input_tokens": cached_prompt_tokens,
            "cached_input_ratio": cached_input_ratio,
        }
        if reasoning_tokens is not None:
            normalized_usage["reasoning_tokens"] = reasoning_tokens
        if cost_usd is not None:
            normalized_usage["cost_usd"] = cost_usd
        return normalized_usage
