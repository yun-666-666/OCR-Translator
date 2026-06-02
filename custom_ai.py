import base64
import json
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from logger import log_debug


ACTIVE_PROFILE_KINDS = {"translation", "ocr"}
CUSTOM_AI_LATENCY_MODE_NONE = "none"
CUSTOM_AI_LATENCY_MODE_SAFE = "safe"
CUSTOM_AI_LATENCY_MODE_STREAM = "stream"
CUSTOM_AI_LATENCY_MODE_RACE = "race"
CUSTOM_AI_LATENCY_MODES = {
    CUSTOM_AI_LATENCY_MODE_NONE,
    CUSTOM_AI_LATENCY_MODE_SAFE,
    CUSTOM_AI_LATENCY_MODE_STREAM,
    CUSTOM_AI_LATENCY_MODE_RACE,
}
CUSTOM_AI_SAFE_TRANSPORT_MODES = {
    CUSTOM_AI_LATENCY_MODE_SAFE,
    CUSTOM_AI_LATENCY_MODE_STREAM,
    CUSTOM_AI_LATENCY_MODE_RACE,
}


def normalize_custom_ai_latency_mode(mode):
    mode = str(mode or "").strip().lower()
    if mode in CUSTOM_AI_LATENCY_MODES:
        return mode
    return CUSTOM_AI_LATENCY_MODE_SAFE


class CustomAIProfileManager:
    """Persist and manage user-defined OpenAI-compatible AI endpoint profiles."""

    def __init__(self, path="custom_ai_profiles.json"):
        self.path = Path(path)
        self.data = {
            "profiles": [],
            "active_translation_profile_id": None,
            "active_ocr_profile_id": None,
        }
        self.load()

    def load(self):
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8-sig") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.data.update({
                    "profiles": loaded.get("profiles", []),
                    "active_translation_profile_id": loaded.get("active_translation_profile_id"),
                    "active_ocr_profile_id": loaded.get("active_ocr_profile_id"),
                })
                self._sanitize()
        except Exception as e:
            log_debug(f"Custom AI profiles load failed: {e}")

    def save(self):
        try:
            if self.path.parent and str(self.path.parent) != ".":
                self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            log_debug(f"Custom AI profiles save failed: {e}")
            return False

    def _sanitize(self):
        profiles = []
        seen_ids = set()
        for profile in self.data.get("profiles", []):
            if not isinstance(profile, dict):
                continue
            profile_id = str(profile.get("id") or uuid.uuid4())
            if profile_id in seen_ids:
                profile_id = str(uuid.uuid4())
            seen_ids.add(profile_id)
            profiles.append({
                "id": profile_id,
                "name": str(profile.get("name") or "Custom AI").strip() or "Custom AI",
                "base_url": str(profile.get("base_url") or "").strip(),
                "api_key": str(profile.get("api_key") or ""),
                "model": str(profile.get("model") or "").strip(),
                "enabled": bool(profile.get("enabled", True)),
            })
        self.data["profiles"] = profiles
        self._repair_active_ids()

    def _first_available_profile_id(self):
        enabled = next((p for p in self.data.get("profiles", []) if p.get("enabled", True)), None)
        if enabled:
            return enabled["id"]
        first = next(iter(self.data.get("profiles", [])), None)
        return first["id"] if first else None

    def _repair_active_ids(self):
        fallback_id = self._first_available_profile_id()
        for kind in ACTIVE_PROFILE_KINDS:
            active_key = self._active_key(kind)
            active_id = self.data.get(active_key)
            if active_id and not self.get_profile(active_id):
                self.data[active_key] = fallback_id

    def _active_key(self, kind):
        self._validate_kind(kind)
        return f"active_{kind}_profile_id"

    def _validate_kind(self, kind):
        if kind not in ACTIVE_PROFILE_KINDS:
            raise ValueError(f"Invalid active profile kind: {kind}")

    def list_profiles(self, kind=None, enabled_only=False):
        profiles = list(self.data.get("profiles", []))
        if kind is not None:
            self._validate_kind(kind)
        if enabled_only:
            profiles = [p for p in profiles if p.get("enabled", True)]
        return profiles

    def get_profile(self, profile_id):
        for profile in self.data.get("profiles", []):
            if profile.get("id") == profile_id:
                return profile
        return None

    def get_active_profile(self, kind):
        active_id = self.data.get(self._active_key(kind))
        return self.get_profile(active_id) if active_id else None

    def set_active_profile(self, kind, profile_id):
        self._validate_kind(kind)
        profile = self.get_profile(profile_id)
        if not profile:
            raise ValueError(f"No profile with id {profile_id}")
        self.data[self._active_key(kind)] = profile_id
        self.save()
        return profile

    def add_profile(self, name, base_url, api_key, model, enabled=True, kind=None):
        if kind is not None:
            self._validate_kind(kind)
        profile = {
            "id": str(uuid.uuid4()),
            "name": str(name).strip(),
            "base_url": str(base_url).strip(),
            "api_key": str(api_key),
            "model": str(model).strip(),
            "enabled": bool(enabled),
        }
        self._validate_profile(profile)
        self.data["profiles"].append(profile)
        for active_kind in ACTIVE_PROFILE_KINDS:
            if not self.data.get(self._active_key(active_kind)):
                self.data[self._active_key(active_kind)] = profile["id"]
        self.save()
        return profile

    def update_profile(self, profile_id, **updates):
        profile = self.get_profile(profile_id)
        if not profile:
            raise ValueError(f"No profile with id {profile_id}")
        for key in ["name", "base_url", "api_key", "model", "enabled"]:
            if key in updates:
                profile[key] = updates[key]
        profile["name"] = str(profile.get("name") or "").strip()
        profile["base_url"] = str(profile.get("base_url") or "").strip()
        profile["api_key"] = str(profile.get("api_key") or "")
        profile["model"] = str(profile.get("model") or "").strip()
        profile["enabled"] = bool(profile.get("enabled", True))
        self._validate_profile(profile)
        self.save()
        return profile

    def delete_profile(self, profile_id):
        removed = None
        remaining = []
        for profile in self.data.get("profiles", []):
            if profile.get("id") == profile_id:
                removed = profile
            else:
                remaining.append(profile)
        if not removed:
            return False
        self.data["profiles"] = remaining
        replacement_id = self._first_available_profile_id()
        for kind in ACTIVE_PROFILE_KINDS:
            active_key = self._active_key(kind)
            if self.data.get(active_key) == profile_id:
                self.data[active_key] = replacement_id
        self.save()
        return True

    def _validate_profile(self, profile):
        if not profile.get("name"):
            raise ValueError("Profile name is required")
        if not profile.get("base_url"):
            raise ValueError("API URL is required")
        if not profile.get("api_key"):
            raise ValueError("API key is required")
        if not profile.get("model"):
            raise ValueError("Model name is required")


class CustomAIProvider:
    """OpenAI Chat Completions compatible translation and OCR provider."""

    def __init__(self, http_client=None, timeout=30):
        self.http_client = http_client
        self._owns_http_client = http_client is None
        self.timeout = timeout
        self.context_window = []
        self._client_lock = threading.Lock()
        self._url_cache_lock = threading.Lock()
        self._successful_chat_urls = {}
        self._successful_models_urls = {}

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
            log_debug("LATENCY: Custom AI HTTP session initialized with connection pooling")
            return self.http_client

    def close(self):
        if self.http_client is not None and self._owns_http_client and hasattr(self.http_client, "close"):
            try:
                self.http_client.close()
                log_debug("LATENCY: Custom AI HTTP session closed")
            except Exception as e:
                log_debug(f"Custom AI HTTP session close failed: {e}")
        self.http_client = None
        self._owns_http_client = True

    def _base_url_cache_key(self, base_url):
        return (base_url or "").strip().rstrip("/")

    def _ordered_candidates(self, base_url, cache, candidate_builder, latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE):
        candidates = candidate_builder(base_url)
        cache_key = self._base_url_cache_key(base_url)
        if normalize_custom_ai_latency_mode(latency_mode) == CUSTOM_AI_LATENCY_MODE_NONE:
            return None, candidates, None
        with self._url_cache_lock:
            cached_url = cache.get(cache_key)
        if cached_url and cached_url in candidates:
            ordered = [cached_url] + [url for url in candidates if url != cached_url]
            return cache_key, ordered, cached_url
        return cache_key, candidates, None

    def _remember_successful_url(self, cache, cache_key, url):
        if cache_key:
            with self._url_cache_lock:
                cache[cache_key] = url

    def _forget_successful_url(self, cache, cache_key, url):
        if cache_key:
            with self._url_cache_lock:
                if cache.get(cache_key) == url:
                    cache.pop(cache_key, None)

    def _is_openrouter_profile(self, profile):
        host = urlparse(profile.get("base_url", "")).netloc.lower()
        return host == "openrouter.ai" or host.endswith(".openrouter.ai")

    def _prepare_payload_for_profile(self, profile, payload, latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE):
        latency_mode = normalize_custom_ai_latency_mode(latency_mode)
        if (
            latency_mode == CUSTOM_AI_LATENCY_MODE_NONE
            or not isinstance(payload, dict)
            or not self._is_openrouter_profile(profile)
        ):
            return payload, False
        request_payload = dict(payload)
        if "provider" in request_payload:
            return request_payload, False
        request_payload["provider"] = {"sort": "latency", "allow_fallbacks": True}
        return request_payload, True

    def normalize_chat_completions_url(self, base_url):
        return self.normalize_chat_completions_url_candidates(base_url)[0]

    def normalize_chat_completions_url_candidates(self, base_url):
        url = (base_url or "").strip().rstrip("/")
        if not url:
            raise ValueError("API URL is required")
        if url.endswith("/chat/completions"):
            return [url]
        if url.endswith("/v1"):
            return [f"{url}/chat/completions"]

        candidates = [
            f"{url}/v1/chat/completions",
            f"{url}/chat/completions",
        ]
        deduped = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def normalize_models_url_candidates(self, base_url):
        url = (base_url or "").strip().rstrip("/")
        if not url:
            raise ValueError("API URL is required")
        if url.endswith("/chat/completions"):
            url = url[: -len("/chat/completions")]

        candidates = []
        if url.endswith("/v1"):
            candidates.append(f"{url}/models")
            candidates.append(f"{url[: -len('/v1')]}/models")
        else:
            candidates.append(f"{url}/v1/models")
            candidates.append(f"{url}/models")

        deduped = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def build_translation_payload(
        self,
        profile,
        text,
        source_lang,
        target_lang,
        custom_prompt="",
        context=None,
        keep_linebreaks=False,
    ):
        context = context or []
        linebreak_instruction = "Preserve line breaks using <br>." if keep_linebreaks else "Return one concise translated text."
        system_parts = [
            "You are a translation engine for on-screen game subtitles.",
            f"Translate from {source_lang or 'auto'} to {target_lang}.",
            "Return only the translation. Do not add explanations, labels, or quotes.",
            linebreak_instruction,
        ]
        if custom_prompt:
            system_parts.append(f"User custom instruction: {custom_prompt}")
        user_parts = []
        if context:
            user_parts.append("Previous subtitles:")
            user_parts.extend(str(item) for item in context if item)
        user_parts.append("Text to translate:")
        user_parts.append(text)
        return {
            "model": profile["model"],
            "messages": [
                {"role": "system", "content": "\n".join(system_parts)},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            "temperature": 0,
        }

    def build_ocr_payload(self, profile, image_data, source_lang, keep_linebreaks=False):
        prompt = (
            "Transcribe the text from the image exactly as it appears. "
            "Do not correct, translate, rephrase, or explain. "
        )
        if keep_linebreaks:
            prompt += "Keep line breaks. "
        prompt += "If there is no text in the image, return only: <EMPTY>."

        data_url = "data:image/webp;base64," + base64.b64encode(image_data).decode("ascii")
        return {
            "model": profile["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0,
        }

    def parse_chat_response(self, response_json):
        if not isinstance(response_json, dict):
            raise ValueError("Invalid API response")
        if "error" in response_json:
            error = response_json["error"]
            if isinstance(error, dict):
                raise ValueError(error.get("message") or json.dumps(error, ensure_ascii=False))
            raise ValueError(str(error))
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
    ):
        payload = self.build_translation_payload(
            profile,
            text,
            source_lang,
            target_lang,
            custom_prompt=custom_prompt,
            context=context,
            keep_linebreaks=keep_linebreaks,
        )
        latency_mode = normalize_custom_ai_latency_mode(latency_mode)
        if latency_mode == CUSTOM_AI_LATENCY_MODE_STREAM:
            response_json, duration = self._stream_post(profile, payload, stream_callback=stream_callback, latency_mode=latency_mode)
        else:
            response_json, duration = self._post(profile, payload, latency_mode=latency_mode)
        result = self.parse_chat_response(response_json)
        return result, self._extract_usage(response_json), duration

    def recognize(self, profile, image_data, source_lang, keep_linebreaks=False, latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE):
        payload = self.build_ocr_payload(profile, image_data, source_lang, keep_linebreaks=keep_linebreaks)
        response_json, duration = self._post(profile, payload, latency_mode=latency_mode)
        result = self.parse_chat_response(response_json)
        if keep_linebreaks:
            result = result.replace("\n", "<br>")
        else:
            result = result.replace("\n", " ")
        if not result or "<EMPTY>" in result:
            result = "<EMPTY>"
        return result, self._extract_usage(response_json), duration

    def test_profile(self, profile, latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE):
        payload = {
            "model": profile["model"],
            "messages": [{"role": "user", "content": "Reply with OK only."}],
            "temperature": 0,
            "max_tokens": 8,
        }
        response_json, duration = self._post(profile, payload, latency_mode=latency_mode)
        return self.parse_chat_response(response_json), duration

    def parse_models_response(self, response_json):
        if not isinstance(response_json, dict):
            raise ValueError("Invalid models response")
        if "error" in response_json:
            error = response_json["error"]
            if isinstance(error, dict):
                raise ValueError(error.get("message") or json.dumps(error, ensure_ascii=False))
            raise ValueError(str(error))
        data = response_json.get("data")
        if not isinstance(data, list):
            raise ValueError("Models response did not contain a data list")
        models = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_name = item.get("id") or item.get("name")
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
        cache_key, urls, cached_url = self._ordered_candidates(
            profile.get("base_url"),
            self._successful_models_urls,
            self.normalize_models_url_candidates,
            latency_mode,
        )
        for url in urls:
            try:
                start = time.monotonic()
                response = self._get_with_light_retry(http_client, url, headers, latency_mode)
                duration = time.monotonic() - start
                status_code = int(getattr(response, "status_code", 200) or 200)
                log_debug(
                    "LATENCY: custom_ai models "
                    f"provider={profile.get('name', 'Custom AI')} "
                    f"url_cached={url == cached_url} status={status_code} duration={duration:.3f}s"
                )
                if status_code >= 400:
                    self._forget_successful_url(self._successful_models_urls, cache_key, url)
                    try:
                        error_payload = response.json()
                        self.parse_models_response(error_payload)
                    except Exception as payload_error:
                        raise ValueError(str(payload_error))
                response.raise_for_status()
                self._remember_successful_url(self._successful_models_urls, cache_key, url)
                return self.parse_models_response(response.json())
            except Exception as e:
                errors.append(f"{url}: {self._sanitize_error(str(e), profile.get('api_key', ''))}")
        raise ValueError("Unable to fetch model list. Tried: " + "; ".join(errors))

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
                    log_debug(f"LATENCY: custom_ai models GET retry for transient error at {url}: {type(e).__name__}")
        raise last_error

    def _post(self, profile, payload, latency_mode=CUSTOM_AI_LATENCY_MODE_SAFE):
        latency_mode = normalize_custom_ai_latency_mode(latency_mode)
        http_client = self._get_http_client(latency_mode)
        headers = {
            "Authorization": f"Bearer {profile.get('api_key', '')}",
            "Content-Type": "application/json",
        }
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
                response = http_client.post(url, headers=headers, json=request_payload, timeout=self.timeout)
                duration = time.monotonic() - start
                status_code = int(getattr(response, "status_code", 200) or 200)
                log_debug(
                    "LATENCY: custom_ai post "
                    f"provider={profile.get('name', 'Custom AI')} "
                    f"url_cached={url == cached_url} "
                    f"openrouter_latency_routing={openrouter_latency_routing} "
                    f"status={status_code} duration={duration:.3f}s"
                )

                if status_code >= 400:
                    self._forget_successful_url(self._successful_chat_urls, cache_key, url)
                    errors.append(self._response_error_message(response, url, api_key))
                    continue

                try:
                    response_json = response.json()
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
                errors.append(f"{url}: {self._sanitize_error(str(e), api_key)}")

        if len(errors) == 1:
            raise ValueError(errors[0])
        raise ValueError("Unable to call chat completions. Tried: " + "; ".join(errors))

    def _stream_post(self, profile, payload, stream_callback=None, latency_mode=CUSTOM_AI_LATENCY_MODE_STREAM):
        latency_mode = normalize_custom_ai_latency_mode(latency_mode)
        http_client = self._get_http_client(latency_mode)
        headers = {
            "Authorization": f"Bearer {profile.get('api_key', '')}",
            "Content-Type": "application/json",
        }
        api_key = profile.get("api_key", "")
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        request_payload, openrouter_latency_routing = self._prepare_payload_for_profile(profile, stream_payload, latency_mode)
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
                response = http_client.post(url, headers=headers, json=request_payload, timeout=self.timeout, stream=True)
                status_code = int(getattr(response, "status_code", 200) or 200)
                log_debug(
                    "LATENCY: custom_ai stream "
                    f"provider={profile.get('name', 'Custom AI')} "
                    f"url_cached={url == cached_url} "
                    f"openrouter_latency_routing={openrouter_latency_routing} "
                    f"status={status_code}"
                )
                if status_code >= 400:
                    self._forget_successful_url(self._successful_chat_urls, cache_key, url)
                    errors.append(self._response_error_message(response, url, api_key))
                    continue
                if hasattr(response, "raise_for_status"):
                    response.raise_for_status()
                response_json = self._parse_streaming_chat_response(response, stream_callback)
                duration = time.monotonic() - start
                self._remember_successful_url(self._successful_chat_urls, cache_key, url)
                return response_json, duration
            except Exception as e:
                self._forget_successful_url(self._successful_chat_urls, cache_key, url)
                if self._discard_owned_http_client_for_transport_error(e):
                    http_client = self._get_http_client(latency_mode)
                errors.append(f"{url}: {self._sanitize_error(str(e), api_key)}")

        if len(errors) == 1:
            raise ValueError(errors[0])
        raise ValueError("Unable to call streaming chat completions. Tried: " + "; ".join(errors))

    def _parse_streaming_chat_response(self, response, stream_callback=None):
        accumulated = ""
        usage = None
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if isinstance(raw_line, bytes):
                raw_line = raw_line.decode("utf-8", errors="replace")
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
                    log_debug(f"Custom AI stream callback failed: {e}")
        if not accumulated:
            raise ValueError("Streaming API response did not contain message content")
        result = {"choices": [{"message": {"content": accumulated}}]}
        if usage:
            result["usage"] = usage
        return result

    def _response_error_message(self, response, url, api_key):
        status_code = int(getattr(response, "status_code", 0) or 0)
        detail = ""
        try:
            payload = response.json()
            if isinstance(payload, dict) and "error" in payload:
                error = payload["error"]
                if isinstance(error, dict):
                    detail = error.get("message") or json.dumps(error, ensure_ascii=False)
                else:
                    detail = str(error)
            elif payload:
                detail = json.dumps(payload, ensure_ascii=False)
        except Exception:
            detail = str(getattr(response, "text", "") or "").strip()

        detail = self._sanitize_error(detail, api_key)
        if len(detail) > 500:
            detail = detail[:500] + "..."
        if detail:
            return f"Chat completions request failed (HTTP {status_code}) at {url}: {detail}"
        return f"Chat completions request failed (HTTP {status_code}) at {url}"

    def _non_json_response_message(self, response, url, api_key):
        text = str(getattr(response, "text", "") or "").strip()
        text = self._sanitize_error(text, api_key)
        if len(text) > 300:
            text = text[:300] + "..."
        if text:
            return f"Chat completions response from {url} was non-JSON or empty. Response: {text}"
        return f"Chat completions response from {url} was non-JSON or empty."

    def _sanitize_error(self, message, api_key):
        message = str(message)
        if api_key:
            message = message.replace(str(api_key), "[redacted]")
        if self._is_tls_eof_error(message):
            return (
                "TLS/SSL connection was closed by the server or proxy before a response was received. "
                "This usually means the API URL, network route, or relay endpoint rejected the HTTPS connection. "
                f"Original error: {message}"
            )
        return message

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
        log_debug(f"LATENCY: discarded Custom AI HTTP session after transport error: {type(error).__name__}")
        return True

    def _extract_usage(self, response_json):
        usage = response_json.get("usage") if isinstance(response_json, dict) else None
        if not isinstance(usage, dict):
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens") or (prompt_tokens + completion_tokens)),
        }
