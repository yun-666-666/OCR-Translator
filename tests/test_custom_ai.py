import json
import importlib.util
import sys
import time
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import gui_builder
from custom_ai import CustomAIProfileManager, CustomAIProvider
from gui_builder import filter_model_values, run_profile_network_task_async
from language_ui import UILanguageManager
from unified_translation_cache import CACHE_SCHEMA_VERSION, UnifiedTranslationCache


translation_handler_spec = importlib.util.spec_from_file_location(
    "translation_handler_for_tests",
    Path(__file__).resolve().parents[1] / "handlers" / "translation_handler.py",
)
translation_handler_module = importlib.util.module_from_spec(translation_handler_spec)
translation_handler_spec.loader.exec_module(translation_handler_module)
TranslationHandler = translation_handler_module.TranslationHandler


class CustomAIProfileManagerTests(unittest.TestCase):
    def test_profile_manager_persists_unified_profiles_and_active_ids(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "profiles.json"
            manager = CustomAIProfileManager(path)

            profile = manager.add_profile(
                name="Local Translate",
                base_url="https://proxy.example/v1",
                api_key="secret",
                model="qwen",
            )
            manager.set_active_profile("translation", profile["id"])
            manager.set_active_profile("ocr", profile["id"])

            reloaded = CustomAIProfileManager(path)

            self.assertEqual(reloaded.get_active_profile("translation")["name"], "Local Translate")
            self.assertEqual(reloaded.get_active_profile("ocr")["model"], "qwen")
            self.assertEqual(len(reloaded.list_profiles()), 1)
            self.assertEqual(len(reloaded.list_profiles("translation")), 1)
            self.assertNotIn("kind", reloaded.list_profiles()[0])

    def test_legacy_kind_profiles_are_migrated_to_unified_list(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "profiles.json"
            legacy = {
                "profiles": [
                    {
                        "id": "translate-1",
                        "name": "Translate Proxy",
                        "kind": "translation",
                        "base_url": "https://proxy.example/v1",
                        "api_key": "secret",
                        "model": "qwen",
                        "enabled": True,
                    },
                    {
                        "id": "ocr-1",
                        "name": "Vision Proxy",
                        "kind": "ocr",
                        "base_url": "https://vision.example/v1",
                        "api_key": "vision-secret",
                        "model": "gpt-4o",
                        "enabled": True,
                    },
                ],
                "active_translation_profile_id": "translate-1",
                "active_ocr_profile_id": "ocr-1",
            }
            path.write_text(json.dumps(legacy), encoding="utf-8")

            manager = CustomAIProfileManager(path)

            self.assertEqual([p["name"] for p in manager.list_profiles()], ["Translate Proxy", "Vision Proxy"])
            self.assertEqual([p["name"] for p in manager.list_profiles("ocr")], ["Translate Proxy", "Vision Proxy"])
            self.assertEqual(manager.get_active_profile("translation")["id"], "translate-1")
            self.assertEqual(manager.get_active_profile("ocr")["id"], "ocr-1")
            self.assertNotIn("kind", manager.list_profiles()[0])

    def test_delete_active_profile_falls_back_to_remaining_profile(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CustomAIProfileManager(Path(tmp_dir) / "profiles.json")
            first = manager.add_profile(
                name="First",
                base_url="https://proxy.example/v1",
                api_key="secret",
                model="model",
            )
            second = manager.add_profile(
                name="Second",
                base_url="https://proxy2.example/v1",
                api_key="secret2",
                model="model2",
            )
            manager.set_active_profile("translation", first["id"])
            manager.set_active_profile("ocr", first["id"])

            manager.delete_profile(first["id"])

            self.assertEqual(manager.get_active_profile("translation")["id"], second["id"])
            self.assertEqual(manager.get_active_profile("ocr")["id"], second["id"])

    def test_profile_manager_persists_responses_wire_api(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "profiles.json"
            manager = CustomAIProfileManager(path)

            profile = manager.add_profile(
                name="Responses Relay",
                base_url="https://relay.example/v1",
                api_key="secret",
                model="gpt-5.5",
                wire_api="responses",
            )

            reloaded = CustomAIProfileManager(path)

            self.assertEqual(reloaded.get_profile(profile["id"])["wire_api"], "responses")

    def test_profile_manager_defaults_to_chat_completions_wire_api(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CustomAIProfileManager(Path(tmp_dir) / "profiles.json")

            profile = manager.add_profile(
                name="Chat Relay",
                base_url="https://relay.example/v1",
                api_key="secret",
                model="qwen",
            )

            self.assertEqual(profile["wire_api"], "chat_completions")


class CustomAIProviderTests(unittest.TestCase):
    def test_normalize_chat_completions_url(self):
        provider = CustomAIProvider()

        self.assertEqual(
            provider.normalize_chat_completions_url("https://host.example/v1"),
            "https://host.example/v1/chat/completions",
        )

    def test_chat_completions_url_candidates_support_v1_chat_and_bare_hosts(self):
        provider = CustomAIProvider()

        self.assertEqual(
            provider.normalize_chat_completions_url_candidates("https://host.example/v1"),
            ["https://host.example/v1/chat/completions"],
        )
        self.assertEqual(
            provider.normalize_chat_completions_url_candidates("https://host.example/v1/chat/completions"),
            ["https://host.example/v1/chat/completions"],
        )
        self.assertEqual(
            provider.normalize_chat_completions_url_candidates("https://host.example"),
            ["https://host.example/v1/chat/completions", "https://host.example/chat/completions"],
        )

    def test_model_list_url_candidates_support_v1_and_chat_completions(self):
        provider = CustomAIProvider()

        self.assertEqual(
            provider.normalize_models_url_candidates("https://host.example/v1"),
            ["https://host.example/v1/models", "https://host.example/models"],
        )
        self.assertEqual(
            provider.normalize_models_url_candidates("https://host.example/v1/"),
            ["https://host.example/v1/models", "https://host.example/models"],
        )
        self.assertEqual(
            provider.normalize_models_url_candidates("https://host.example/v1/chat/completions"),
            ["https://host.example/v1/models", "https://host.example/models"],
        )
        self.assertEqual(
            provider.normalize_models_url_candidates("https://host.example/api"),
            ["https://host.example/api/v1/models", "https://host.example/api/models"],
        )

    def test_parse_model_list_response_supports_id_and_name(self):
        provider = CustomAIProvider()

        models = provider.parse_models_response({
            "data": [
                {"id": "gpt-4o"},
                {"name": "claude-3.5"},
                {"id": "gpt-4o"},
            ]
        })

        self.assertEqual(models, ["gpt-4o", "claude-3.5"])

    def test_parse_model_list_response_supports_top_level_models_list(self):
        provider = CustomAIProvider()

        models = provider.parse_models_response({
            "models": [
                {"id": "gpt-5.5"},
                {"name": "qwen-max"},
            ]
        })

        self.assertEqual(models, ["gpt-5.5", "qwen-max"])

    def test_fetch_models_tries_fallback_without_leaking_key(self):
        class Response:
            def __init__(self, payload, status_code=200, text=""):
                self.payload = payload
                self.status_code = status_code
                self.text = text

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

            def json(self):
                return self.payload

        class Client:
            def __init__(self):
                self.urls = []

            def get(self, url, headers=None, timeout=None):
                self.urls.append((url, headers))
                if len(self.urls) == 1:
                    return Response({"error": {"message": "not here"}}, status_code=404, text="nope")
                return Response({"data": [{"id": "qwen-max"}]})

        client = Client()
        provider = CustomAIProvider(http_client=client)

        models = provider.fetch_models({"base_url": "https://host.example/v1/chat/completions", "api_key": "super-secret"})

        self.assertEqual(models, ["qwen-max"])
        self.assertEqual([url for url, _ in client.urls], ["https://host.example/v1/models", "https://host.example/models"])
        self.assertEqual(client.urls[0][1]["Authorization"], "Bearer super-secret")
        self.assertEqual(
            provider.normalize_chat_completions_url("https://host.example/v1/"),
            "https://host.example/v1/chat/completions",
        )
        self.assertEqual(
            provider.normalize_chat_completions_url("https://host.example/v1/chat/completions"),
            "https://host.example/v1/chat/completions",
        )

    def test_post_tries_chat_completion_fallback_for_bare_base_url(self):
        class Response:
            def __init__(self, payload=None, status_code=200, text=""):
                self.payload = payload
                self.status_code = status_code
                self.text = text

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

            def json(self):
                if self.payload is None:
                    raise ValueError("Expecting value")
                return self.payload

        class Client:
            def __init__(self):
                self.urls = []

            def post(self, url, headers=None, json=None, timeout=None):
                self.urls.append(url)
                if len(self.urls) == 1:
                    return Response(status_code=404, text="<html>not found</html>")
                return Response({"choices": [{"message": {"content": "OK"}}]})

        provider = CustomAIProvider(http_client=Client())

        response_json, _duration = provider._post(
            {"base_url": "https://host.example", "api_key": "super-secret"},
            {"model": "demo", "messages": []},
        )

        self.assertEqual(response_json["choices"][0]["message"]["content"], "OK")
        self.assertEqual(
            provider.http_client.urls,
            ["https://host.example/v1/chat/completions", "https://host.example/chat/completions"],
        )

    def test_post_reuses_successful_chat_completion_url_for_base_url(self):
        class Response:
            def __init__(self, payload=None, status_code=200, text=""):
                self.payload = payload
                self.status_code = status_code
                self.text = text

            def json(self):
                return self.payload

        class Client:
            def __init__(self):
                self.urls = []

            def post(self, url, headers=None, json=None, timeout=None):
                self.urls.append(url)
                if len(self.urls) == 1:
                    return Response(status_code=404, text="not found")
                return Response({"choices": [{"message": {"content": "OK"}}]})

        provider = CustomAIProvider(http_client=Client())
        profile = {"base_url": "https://host.example", "api_key": "super-secret"}
        payload = {"model": "demo", "messages": []}

        provider._post(profile, payload)
        provider._post(profile, payload)

        self.assertEqual(
            provider.http_client.urls,
            [
                "https://host.example/v1/chat/completions",
                "https://host.example/chat/completions",
                "https://host.example/chat/completions",
            ],
        )

    def test_post_clears_failed_cached_chat_url_and_falls_back(self):
        class Response:
            def __init__(self, payload=None, status_code=200, text=""):
                self.payload = payload
                self.status_code = status_code
                self.text = text

            def json(self):
                return self.payload

        class Client:
            def __init__(self):
                self.urls = []

            def post(self, url, headers=None, json=None, timeout=None):
                self.urls.append(url)
                if len(self.urls) == 1:
                    return Response(status_code=404, text="not found")
                if len(self.urls) == 2:
                    return Response({"choices": [{"message": {"content": "first OK"}}]})
                if url.endswith("/chat/completions") and not url.endswith("/v1/chat/completions"):
                    return Response(status_code=502, text="cached upstream failed")
                return Response({"choices": [{"message": {"content": "fallback OK"}}]})

        provider = CustomAIProvider(http_client=Client())
        profile = {"base_url": "https://host.example", "api_key": "super-secret"}
        payload = {"model": "demo", "messages": []}

        provider._post(profile, payload)
        response_json, _duration = provider._post(profile, payload)

        self.assertEqual(response_json["choices"][0]["message"]["content"], "fallback OK")
        self.assertEqual(
            provider.http_client.urls,
            [
                "https://host.example/v1/chat/completions",
                "https://host.example/chat/completions",
                "https://host.example/chat/completions",
                "https://host.example/v1/chat/completions",
            ],
        )

    def test_fetch_models_reuses_successful_models_url_for_base_url(self):
        class Response:
            def __init__(self, payload=None, status_code=200, text=""):
                self.payload = payload
                self.status_code = status_code
                self.text = text

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

            def json(self):
                return self.payload

        class Client:
            def __init__(self):
                self.urls = []

            def get(self, url, headers=None, timeout=None):
                self.urls.append(url)
                if len(self.urls) == 1:
                    return Response({"error": {"message": "not here"}}, status_code=404, text="nope")
                return Response({"data": [{"id": "fast-model"}]})

        provider = CustomAIProvider(http_client=Client())
        profile = {"base_url": "https://host.example/v1/chat/completions", "api_key": "super-secret"}

        provider.fetch_models(profile)
        provider.fetch_models(profile)

        self.assertEqual(
            provider.http_client.urls,
            [
                "https://host.example/v1/models",
                "https://host.example/models",
                "https://host.example/models",
            ],
        )

    def test_translate_uses_responses_api_when_profile_requests_it(self):
        class Response:
            status_code = 200

            def json(self):
                return {
                    "output_text": "\u4f60\u597d",
                    "usage": {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
                }

        class Client:
            def __init__(self):
                self.posts = []

            def post(self, url, headers=None, json=None, timeout=None):
                self.posts.append({"url": url, "payload": json, "headers": headers})
                return Response()

        provider = CustomAIProvider(http_client=Client())

        translated, usage, _duration = provider.translate(
            {
                "name": "Responses Relay",
                "base_url": "https://host.example/v1",
                "api_key": "super-secret",
                "model": "gpt-5.5",
                "wire_api": "responses",
            },
            "Hello",
            "en",
            "zh-CN",
        )

        self.assertEqual(translated, "\u4f60\u597d")
        self.assertEqual(usage["total_tokens"], 14)
        self.assertEqual(provider.http_client.posts[0]["url"], "https://host.example/v1/responses")
        payload = provider.http_client.posts[0]["payload"]
        self.assertNotIn("messages", payload)
        self.assertEqual(payload["model"], "gpt-5.5")
        self.assertFalse(payload["store"])
        self.assertEqual(payload["input"][0]["role"], "system")

    def test_translate_decodes_utf8_json_when_response_json_uses_wrong_charset(self):
        class Response:
            status_code = 200

            def __init__(self):
                payload = {
                    "choices": [{"message": {"content": "选择OCR源区域。"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                }
                self.content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.encoding = "ISO-8859-1"
                self.text = self.content.decode("latin-1")

            def json(self):
                return json.loads(self.text)

        class Client:
            def post(self, url, headers=None, json=None, timeout=None):
                return Response()

        provider = CustomAIProvider(http_client=Client())

        translated, usage, _duration = provider.translate(
            {
                "name": "Wrong Charset Relay",
                "base_url": "https://host.example/v1",
                "api_key": "super-secret",
                "model": "gpt-test",
            },
            "Select the OCR source area.",
            "en",
            "zh-CN",
        )

        self.assertEqual(translated, "选择OCR源区域。")
        self.assertEqual(usage["total_tokens"], 13)

    def test_translate_streams_responses_api_when_profile_requests_it(self):
        class Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def iter_lines(self, decode_unicode=True):
                lines = [
                    'event: response.output_text.delta',
                    'data: {"type":"response.output_text.delta","delta":"你"}',
                    'event: response.output_text.delta',
                    'data: {"type":"response.output_text.delta","delta":"好"}',
                    'event: response.completed',
                    'data: {"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":2}}}',
                ]
                for line in lines:
                    yield line if decode_unicode else line.encode("utf-8")

        class Client:
            def __init__(self):
                self.posts = []

            def post(self, url, headers=None, json=None, timeout=None, stream=False):
                self.posts.append({"url": url, "payload": json, "stream": stream})
                return Response()

        provider = CustomAIProvider(http_client=Client())
        partials = []

        translated, usage, _duration = provider.translate(
            {
                "name": "Responses Relay",
                "base_url": "https://host.example/v1",
                "api_key": "super-secret",
                "model": "gpt-5.5",
                "wire_api": "responses",
            },
            "Hello",
            "en",
            "zh-CN",
            latency_mode="stream",
            stream_callback=partials.append,
        )

        self.assertEqual(translated, "\u4f60\u597d")
        self.assertEqual(partials, ["\u4f60", "\u4f60\u597d"])
        self.assertEqual(usage["total_tokens"], 7)
        self.assertEqual(provider.http_client.posts[0]["url"], "https://host.example/v1/responses")
        self.assertTrue(provider.http_client.posts[0]["stream"])
        self.assertTrue(provider.http_client.posts[0]["payload"]["stream"])

    def test_stream_translation_decodes_utf8_lines_when_response_charset_is_wrong(self):
        class Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def iter_lines(self, decode_unicode=True):
                payloads = [
                    {"choices": [{"delta": {"content": "选"}}]},
                    {"choices": [{"delta": {"content": "择"}}], "usage": {"total_tokens": 2}},
                    "[DONE]",
                ]
                for payload in payloads:
                    if payload == "[DONE]":
                        line = "data: [DONE]"
                    else:
                        line = "data: " + json.dumps(payload, ensure_ascii=False)
                    raw = line.encode("utf-8")
                    yield raw.decode("latin-1") if decode_unicode else raw

        class Client:
            def post(self, url, headers=None, json=None, timeout=None, stream=False):
                return Response()

        provider = CustomAIProvider(http_client=Client())
        partials = []

        translated, usage, _duration = provider.translate(
            {
                "name": "Wrong Charset Stream",
                "base_url": "https://host.example/v1",
                "api_key": "super-secret",
                "model": "gpt-test",
            },
            "Select",
            "en",
            "zh-CN",
            latency_mode="stream",
            stream_callback=partials.append,
        )

        self.assertEqual(translated, "选择")
        self.assertEqual(partials, ["选", "选择"])
        self.assertEqual(usage["total_tokens"], 2)

    def test_responses_ocr_payload_uses_input_image(self):
        class Response:
            status_code = 200

            def json(self):
                return {"output_text": "screen text"}

        class Client:
            def __init__(self):
                self.posts = []

            def post(self, url, headers=None, json=None, timeout=None):
                self.posts.append(json)
                return Response()

        provider = CustomAIProvider(http_client=Client())

        result, _usage, _duration = provider.recognize(
            {
                "name": "Responses Relay",
                "base_url": "https://host.example/v1",
                "api_key": "super-secret",
                "model": "gpt-5.5",
                "wire_api": "responses",
            },
            b"webp-bytes",
            "en",
        )

        self.assertEqual(result, "screen text")
        content = provider.http_client.posts[0]["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[1]["type"], "input_image")
        self.assertIn("data:image/webp;base64,", content[1]["image_url"])

    def test_fetch_models_falls_back_to_configured_model_for_responses_profiles(self):
        class Response:
            def __init__(self, payload=None, status_code=200, text=""):
                self.payload = payload
                self.status_code = status_code
                self.text = text

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

            def json(self):
                if self.payload is None:
                    raise ValueError("Expecting value: line 1 column 1 (char 0)")
                return self.payload

        class Client:
            def __init__(self):
                self.urls = []

            def get(self, url, headers=None, timeout=None):
                self.urls.append(url)
                if len(self.urls) == 1:
                    return Response({"object": "list"})
                return Response(None, text="")

        provider = CustomAIProvider(http_client=Client())

        models = provider.fetch_models(
            {
                "base_url": "https://host.example/v1",
                "api_key": "super-secret",
                "model": "gpt-5.5",
                "wire_api": "responses",
            }
        )

        self.assertEqual(models, ["gpt-5.5"])
        self.assertEqual(provider.http_client.urls, ["https://host.example/v1/models", "https://host.example/models"])

    def test_fetch_models_reports_api_error_message_for_error_payloads(self):
        class Response:
            def __init__(self, payload=None, status_code=401, text=""):
                self.payload = payload
                self.status_code = status_code
                self.text = text

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

            def json(self):
                return self.payload

        class Client:
            def __init__(self):
                self.urls = []

            def get(self, url, headers=None, timeout=None):
                self.urls.append(url)
                return Response({"code": "INVALID_API_KEY", "message": "Invalid API key"}, status_code=401, text="nope")

        provider = CustomAIProvider(http_client=Client())

        with self.assertRaises(ValueError) as ctx:
            provider.fetch_models(
                {
                    "base_url": "https://host.example/v1",
                    "api_key": "super-secret",
                    "model": "",
                    "wire_api": "responses",
                }
            )

        self.assertIn("Invalid API key", str(ctx.exception))
        self.assertNotIn("Models response did not contain a data list", str(ctx.exception))

    def test_openrouter_profile_adds_latency_provider_routing_without_mutating_input(self):
        class Response:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "OK"}}]}

        class Client:
            def __init__(self):
                self.payloads = []

            def post(self, url, headers=None, json=None, timeout=None):
                self.payloads.append(json)
                return Response()

        payload = {"model": "openai/gpt-5.5", "messages": []}
        provider = CustomAIProvider(http_client=Client())

        provider._post(
            {"base_url": "https://openrouter.ai/api/v1", "api_key": "super-secret"},
            payload,
        )

        self.assertEqual(
            provider.http_client.payloads[0]["provider"],
            {"sort": "latency", "allow_fallbacks": True},
        )
        self.assertNotIn("provider", payload)

    def test_openrouter_profile_preserves_existing_provider_routing(self):
        class Response:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "OK"}}]}

        class Client:
            def __init__(self):
                self.payloads = []

            def post(self, url, headers=None, json=None, timeout=None):
                self.payloads.append(json)
                return Response()

        payload = {
            "model": "openai/gpt-5.5",
            "messages": [],
            "provider": {"sort": "price"},
        }
        provider = CustomAIProvider(http_client=Client())

        provider._post(
            {"base_url": "https://openrouter.ai/api/v1", "api_key": "super-secret"},
            payload,
        )

        self.assertEqual(provider.http_client.payloads[0]["provider"], {"sort": "price"})

    def test_none_latency_mode_skips_openrouter_provider_routing(self):
        payload = {"model": "openai/gpt-5.5", "messages": []}
        provider = CustomAIProvider()

        request_payload, routed = provider._prepare_payload_for_profile(
            {"base_url": "https://openrouter.ai/api/v1"},
            payload,
            latency_mode="none",
        )

        self.assertFalse(routed)
        self.assertIs(request_payload, payload)
        self.assertNotIn("provider", request_payload)

    def test_default_requests_session_is_reused_and_closed(self):
        class Response:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "OK"}}]}

        class FakeAdapter:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeSession:
            instances = []

            def __init__(self):
                self.mounted = []
                self.closed = False
                self.posts = []
                FakeSession.instances.append(self)

            def mount(self, prefix, adapter):
                self.mounted.append((prefix, adapter))

            def post(self, url, headers=None, json=None, timeout=None):
                self.posts.append(url)
                return Response()

            def close(self):
                self.closed = True

        fake_requests = types.SimpleNamespace(
            Session=FakeSession,
            adapters=types.SimpleNamespace(HTTPAdapter=FakeAdapter),
        )
        provider = CustomAIProvider()

        with unittest.mock.patch.dict(sys.modules, {"requests": fake_requests}):
            provider._post(
                {"base_url": "https://host.example/v1", "api_key": "super-secret"},
                {"model": "demo", "messages": []},
            )
            provider._post(
                {"base_url": "https://host.example/v1", "api_key": "super-secret"},
                {"model": "demo", "messages": []},
            )

        self.assertEqual(len(FakeSession.instances), 1)
        self.assertEqual(FakeSession.instances[0].posts, ["https://host.example/v1/chat/completions"] * 2)

        provider.close()

        self.assertTrue(FakeSession.instances[0].closed)
        self.assertIsNone(provider.http_client)

    def test_stream_translation_emits_partial_updates_and_returns_final_text(self):
        class Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def iter_lines(self, decode_unicode=False):
                lines = [
                    'data: {"choices":[{"delta":{"content":"Hel"}}]}',
                    'data: {"choices":[{"delta":{"content":"lo"}}]}',
                    'data: [DONE]',
                ]
                return iter(lines)

        class Client:
            def __init__(self):
                self.payloads = []
                self.stream_flags = []

            def post(self, url, headers=None, json=None, timeout=None, stream=False):
                self.payloads.append(json)
                self.stream_flags.append(stream)
                return Response()

        partials = []
        provider = CustomAIProvider(http_client=Client())

        result, usage, duration = provider.translate(
            {"base_url": "https://host.example/v1", "api_key": "super-secret", "model": "demo"},
            "Bonjour",
            "fr",
            "en",
            latency_mode="stream",
            stream_callback=partials.append,
        )

        self.assertEqual(result, "Hello")
        self.assertEqual(partials, ["Hel", "Hello"])
        self.assertEqual(usage["total_tokens"], 0)
        self.assertTrue(provider.http_client.stream_flags[0])
        self.assertTrue(provider.http_client.payloads[0]["stream"])

    def test_stream_post_preserves_original_payload_while_adding_stream_flag(self):
        class Response:
            status_code = 200

            def raise_for_status(self):
                return None

            def iter_lines(self, decode_unicode=False):
                return iter([
                    'data: {"choices":[{"delta":{"content":"OK"}}]}',
                    'data: [DONE]',
                ])

        class Client:
            def __init__(self):
                self.payloads = []

            def post(self, url, headers=None, json=None, timeout=None, stream=False):
                self.payloads.append(json)
                return Response()

        payload = {"model": "demo", "messages": []}
        provider = CustomAIProvider(http_client=Client())

        provider._stream_post(
            {"base_url": "https://host.example/v1", "api_key": "super-secret"},
            payload,
        )

        self.assertNotIn("stream", payload)
        self.assertTrue(provider.http_client.payloads[0]["stream"])

    def test_post_reports_non_json_response_without_jsondecode_or_key_leak(self):
        class Response:
            status_code = 200
            text = "<html>super-secret upstream page</html>"

            def raise_for_status(self):
                return None

            def json(self):
                raise ValueError("Expecting value: line 1 column 1 (char 0) super-secret")

        class Client:
            def post(self, url, headers=None, json=None, timeout=None):
                return Response()

        provider = CustomAIProvider(http_client=Client())

        with self.assertRaises(ValueError) as ctx:
            provider._post(
                {"base_url": "https://host.example/v1", "api_key": "super-secret"},
                {"model": "demo", "messages": []},
            )

        message = str(ctx.exception)
        self.assertIn("non-JSON", message)
        self.assertNotIn("JSONDecodeError", message)
        self.assertNotIn("super-secret", message)

    def test_post_reports_json_error_body(self):
        class Response:
            status_code = 400
            text = '{"error":{"message":"bad model"}}'

            def raise_for_status(self):
                raise RuntimeError("400 Client Error")

            def json(self):
                return {"error": {"message": "bad model"}}

        class Client:
            def post(self, url, headers=None, json=None, timeout=None):
                return Response()

        provider = CustomAIProvider(http_client=Client())

        with self.assertRaises(ValueError) as ctx:
            provider._post(
                {"base_url": "https://host.example/v1", "api_key": "super-secret"},
                {"model": "demo", "messages": []},
            )

        self.assertIn("bad model", str(ctx.exception))
        self.assertNotIn("super-secret", str(ctx.exception))

    def test_post_enters_rate_limit_cooldown_after_retry_after_response(self):
        class Response:
            status_code = 429
            text = '{"error":{"message":"Rate limit exceeded"}}'
            headers = {"Retry-After": "7"}

            def raise_for_status(self):
                raise RuntimeError("429 Client Error")

            def json(self):
                return {"error": {"message": "Rate limit exceeded"}}

        class Client:
            def __init__(self):
                self.calls = 0

            def post(self, url, headers=None, json=None, timeout=None):
                self.calls += 1
                return Response()

        provider = CustomAIProvider(http_client=Client())
        profile = {"base_url": "https://host.example/v1", "api_key": "super-secret"}

        with self.assertRaises(ValueError) as ctx:
            provider._post(profile, {"model": "demo", "messages": []})

        self.assertIn("Rate limit exceeded", str(ctx.exception))
        self.assertEqual(provider.http_client.calls, 1)
        self.assertGreater(provider.get_cooldown_remaining(profile), 0)

        with self.assertRaises(ValueError) as cooldown_ctx:
            provider._post(profile, {"model": "demo", "messages": []})

        self.assertIn("cooldown", str(cooldown_ctx.exception).lower())
        self.assertEqual(provider.http_client.calls, 1)

    def test_post_enters_cooldown_after_service_unavailable_retry_after_response(self):
        class Response:
            status_code = 503
            text = '{"error":{"message":"Service temporarily unavailable"}}'
            headers = {"Retry-After": "11"}

            def raise_for_status(self):
                raise RuntimeError("503 Client Error")

            def json(self):
                return {"error": {"message": "Service temporarily unavailable"}}

        class Client:
            def __init__(self):
                self.calls = 0

            def post(self, url, headers=None, json=None, timeout=None):
                self.calls += 1
                return Response()

        provider = CustomAIProvider(http_client=Client())
        profile = {"base_url": "https://host.example/v1", "api_key": "super-secret"}

        with self.assertRaises(ValueError) as ctx:
            provider._post(profile, {"model": "demo", "messages": []})

        self.assertIn("temporarily unavailable", str(ctx.exception).lower())
        self.assertEqual(provider.http_client.calls, 1)
        self.assertGreater(provider.get_cooldown_remaining(profile), 0)

        with self.assertRaises(ValueError) as cooldown_ctx:
            provider._post(profile, {"model": "demo", "messages": []})

        self.assertIn("cooldown", str(cooldown_ctx.exception).lower())
        self.assertEqual(provider.http_client.calls, 1)

    def test_rate_limit_cooldown_grows_after_consecutive_retry_after_failures(self):
        class Response:
            status_code = 429
            text = '{"error":{"message":"Rate limit exceeded"}}'
            headers = {}

            def raise_for_status(self):
                raise RuntimeError("429 Client Error")

            def json(self):
                return {"error": {"message": "Rate limit exceeded"}}

        provider = CustomAIProvider(http_client=object())
        profile = {"base_url": "https://host.example/v1", "name": "Custom AI"}

        with patch("custom_ai.time.monotonic", return_value=100.0):
            first = provider._activate_rate_limit_cooldown(profile, Response(), "Rate limit exceeded")

        with patch("custom_ai.time.monotonic", return_value=101.0):
            second = provider._activate_rate_limit_cooldown(profile, Response(), "Rate limit exceeded")

        self.assertGreater(second, first)

    def test_successful_response_resets_rate_limit_backoff_counter(self):
        class Response:
            status_code = 429
            text = '{"error":{"message":"Rate limit exceeded"}}'
            headers = {}

            def raise_for_status(self):
                raise RuntimeError("429 Client Error")

            def json(self):
                return {"error": {"message": "Rate limit exceeded"}}

        provider = CustomAIProvider(http_client=object())
        profile = {"base_url": "https://host.example/v1", "name": "Custom AI"}

        with patch("custom_ai.time.monotonic", return_value=100.0):
            _ = provider._activate_rate_limit_cooldown(profile, Response(), "Rate limit exceeded")
            first_counter = provider._rate_limit_backoff_counts[provider._rate_limit_cache_key(profile)]

        provider._note_rate_limit_success(profile)

        self.assertNotIn(provider._rate_limit_cache_key(profile), provider._rate_limit_backoff_counts)
        self.assertEqual(first_counter, 1)

    def test_ssl_eof_transport_errors_are_explained_without_key_leak(self):
        provider = CustomAIProvider()

        message = provider._sanitize_error(
            "HTTPSConnectionPool(host='api.e2ez.com', port=443): Max retries exceeded "
            "with url: /v1/chat/completions (Caused by SSLError(SSLEOFError(8, "
            "'[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol "
            "super-secret')))",
            "super-secret",
        )

        self.assertIn("TLS/SSL connection was closed by the server or proxy", message)
        self.assertNotIn("super-secret", message)

    def test_build_translation_payload_contains_prompt_context_and_text(self):
        provider = CustomAIProvider()
        profile = {"model": "qwen-translator"}

        payload = provider.build_translation_payload(
            profile=profile,
            text="Bonjour",
            source_lang="fr",
            target_lang="en",
            custom_prompt="Use game subtitle style.",
            context=["Salut", "Ca va?"],
            keep_linebreaks=False,
        )

        self.assertEqual(payload["model"], "qwen-translator")
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertIn("Use game subtitle style.", serialized)
        self.assertIn("fr", serialized)
        self.assertIn("en", serialized)
        self.assertIn("Salut", serialized)
        self.assertIn("Bonjour", serialized)

    def test_build_ocr_payload_contains_webp_data_url(self):
        provider = CustomAIProvider()
        profile = {"model": "vision-model"}

        payload = provider.build_ocr_payload(
            profile=profile,
            image_data=b"webp-bytes",
            source_lang="ja",
            keep_linebreaks=True,
        )

        serialized = json.dumps(payload)
        self.assertEqual(payload["model"], "vision-model")
        self.assertIn("data:image/webp;base64,", serialized)
        self.assertIn("Transcribe", serialized)

    def test_build_ocr_payload_includes_non_auto_source_language_hint(self):
        provider = CustomAIProvider()
        payload = provider.build_ocr_payload(
            profile={"model": "vision-model"},
            image_data=b"webp-bytes",
            source_lang="ja",
            keep_linebreaks=False,
        )
        prompt = payload["messages"][0]["content"][0]["text"]

        self.assertIn("Source language", prompt)
        self.assertIn("ja", prompt)

    def test_build_ocr_payload_omits_source_language_hint_for_auto(self):
        provider = CustomAIProvider()
        payload = provider.build_ocr_payload(
            profile={"model": "vision-model"},
            image_data=b"webp-bytes",
            source_lang="auto",
            keep_linebreaks=False,
        )
        prompt = payload["messages"][0]["content"][0]["text"]

        self.assertNotIn("Source language", prompt)

    def test_parse_response_handles_success_and_errors(self):
        provider = CustomAIProvider()

        self.assertEqual(
            provider.parse_chat_response({"choices": [{"message": {"content": "Hello"}}]}),
            "Hello",
        )
        with self.assertRaises(ValueError):
            provider.parse_chat_response({"choices": []})
        with self.assertRaises(ValueError):
            provider.parse_chat_response({"error": {"message": "bad key"}})

    def test_parse_responses_response_ignores_null_error_field(self):
        provider = CustomAIProvider()

        result = provider.parse_responses_response({
            "error": None,
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "OK"},
                    ]
                }
            ],
        })

        self.assertEqual(result, "OK")

    def test_cache_key_is_isolated_by_profile_url_and_model(self):
        cache = UnifiedTranslationCache(max_size=10)

        cache.store(
            "Bonjour",
            "fr",
            "en",
            "custom_ai",
            "Hello from A",
            profile_id="a",
            base_url="https://a.example/v1",
            model="model-a",
        )

        self.assertEqual(
            cache.get(
                "Bonjour",
                "fr",
                "en",
                "custom_ai",
                profile_id="a",
                base_url="https://a.example/v1",
                model="model-a",
            ),
            "Hello from A",
        )
        self.assertIsNone(
            cache.get(
                "Bonjour",
                "fr",
                "en",
                "custom_ai",
                profile_id="b",
                base_url="https://b.example/v1",
                model="model-a",
            )
        )

    def test_cache_key_is_isolated_by_custom_prompt_context_and_linebreak_mode(self):
        cache = UnifiedTranslationCache(max_size=10)

        cache.store(
            "Bonjour",
            "fr",
            "en",
            "custom_ai",
            "Hello with context",
            profile_id="a",
            base_url="https://a.example/v1",
            model="model-a",
            custom_prompt="Use subtitle tone",
            keep_linebreaks=True,
            context=("one", "two"),
        )

        self.assertEqual(
            cache.get(
                "Bonjour",
                "fr",
                "en",
                "custom_ai",
                profile_id="a",
                base_url="https://a.example/v1",
                model="model-a",
                custom_prompt="Use subtitle tone",
                keep_linebreaks=True,
                context=("one", "two"),
            ),
            "Hello with context",
        )
        self.assertIsNone(
            cache.get(
                "Bonjour",
                "fr",
                "en",
                "custom_ai",
                profile_id="a",
                base_url="https://a.example/v1",
                model="model-a",
                custom_prompt="Use narration tone",
                keep_linebreaks=True,
                context=("one", "two"),
            )
        )
        self.assertIsNone(
            cache.get(
                "Bonjour",
                "fr",
                "en",
                "custom_ai",
                profile_id="a",
                base_url="https://a.example/v1",
                model="model-a",
                custom_prompt="Use subtitle tone",
                keep_linebreaks=False,
                context=("one", "two"),
            )
        )
        self.assertIsNone(
            cache.get(
                "Bonjour",
                "fr",
                "en",
                "custom_ai",
                profile_id="a",
                base_url="https://a.example/v1",
                model="model-a",
                custom_prompt="Use subtitle tone",
                keep_linebreaks=True,
                context=("three",),
            )
        )

    def test_cache_key_is_isolated_by_wire_api_and_reasoning_effort(self):
        cache = UnifiedTranslationCache(max_size=10)
        params = {
            "profile_id": "profile-1",
            "base_url": "https://host.example/v1/",
            "model": "gpt-5.5",
            "wire_api": "responses",
            "reasoning_effort": "xhigh",
        }
        cache.store("Hello", "en", "zh-CN", "custom_ai", "你好", **params)

        self.assertEqual(
            cache.get("Hello", "en", "zh-CN", "custom_ai", **params),
            "你好",
        )
        self.assertIsNone(
            cache.get(
                "Hello",
                "en",
                "zh-CN",
                "custom_ai",
                **{**params, "wire_api": "chat_completions"},
            )
        )
        self.assertIsNone(
            cache.get(
                "Hello",
                "en",
                "zh-CN",
                "custom_ai",
                **{**params, "reasoning_effort": "low"},
            )
        )

    def test_full_cache_update_does_not_evict_another_entry(self):
        cache = UnifiedTranslationCache(max_size=3)
        for key in ("one", "two", "three"):
            cache.store(key, "en", "zh-CN", "custom_ai", key)

        cache.store("three", "en", "zh-CN", "custom_ai", "updated")

        self.assertEqual(cache.get_stats()["total_entries"], 3)
        self.assertEqual(cache.get("one", "en", "zh-CN", "custom_ai"), "one")
        self.assertEqual(cache.get("three", "en", "zh-CN", "custom_ai"), "updated")

    def test_cache_hit_uses_single_dictionary_lookup(self):
        class CountingDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.contains_calls = 0
                self.getitem_calls = 0

            def __contains__(self, key):
                self.contains_calls += 1
                return super().__contains__(key)

            def __getitem__(self, key):
                self.getitem_calls += 1
                return super().__getitem__(key)

        cache = UnifiedTranslationCache(max_size=10)
        cache.store("Hello", "en", "zh-CN", "custom_ai", "你好")
        cache._cache = CountingDict(cache._cache)

        self.assertEqual(cache.get("Hello", "en", "zh-CN", "custom_ai"), "你好")
        self.assertEqual(cache._cache.getitem_calls, 1)
        self.assertEqual(cache._cache.contains_calls, 0)

    def test_get_stats_uses_provider_index_without_scanning_cache_keys(self):
        class NoKeysDict(dict):
            def keys(self):
                raise AssertionError("full cache key scan should not run")

        cache = UnifiedTranslationCache(max_size=10)
        cache.store("one", "en", "zh-CN", "custom_ai", "one")
        cache.store("two", "en", "zh-CN", "custom_ai", "two")
        cache.store("three", "en", "zh-CN", "deepl_api", "three")
        cache._cache = NoKeysDict(cache._cache)

        stats = cache.get_stats()

        self.assertEqual(stats["total_entries"], 3)
        self.assertEqual(
            stats["provider_breakdown"],
            {"custom_ai": 2, "deepl_api": 1},
        )

    def test_clear_provider_uses_provider_index_without_scanning_cache_keys(self):
        class NoKeysDict(dict):
            def keys(self):
                raise AssertionError("full cache key scan should not run")

        cache = UnifiedTranslationCache(max_size=10)
        cache.store("one", "en", "zh-CN", "custom_ai", "one")
        cache.store("two", "en", "zh-CN", "custom_ai", "two")
        cache.store("three", "en", "zh-CN", "deepl_api", "three")
        cache._cache = NoKeysDict(cache._cache)

        cache.clear_provider("custom_ai")

        self.assertEqual(cache.get_stats()["total_entries"], 1)
        self.assertIsNone(cache.get("one", "en", "zh-CN", "custom_ai"))
        self.assertIsNone(cache.get("two", "en", "zh-CN", "custom_ai"))
        self.assertEqual(cache.get("three", "en", "zh-CN", "deepl_api"), "three")

    def test_persistent_load_trims_exactly_to_max_size(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "custom_ai_cache.json"
            key_builder = UnifiedTranslationCache(max_size=20)
            entries = []
            for index in range(12):
                cache_key = key_builder._generate_cache_key(
                    f"text-{index}",
                    "en",
                    "zh-CN",
                    "custom_ai",
                )
                entries.append(
                    {
                        "key": list(cache_key),
                        "translation": f"value-{index}",
                        "access_time": float(index),
                    }
                )
            cache_path.write_text(
                json.dumps(
                    {
                        "schema_version": CACHE_SCHEMA_VERSION,
                        "entries": entries,
                    }
                ),
                encoding="utf-8",
            )

            cache = UnifiedTranslationCache(max_size=3, persistence_path=cache_path)

            self.assertEqual(cache.get_stats()["total_entries"], 3)
            self.assertIsNone(cache.get("text-8", "en", "zh-CN", "custom_ai"))
            self.assertEqual(
                cache.get("text-11", "en", "zh-CN", "custom_ai"),
                "value-11",
            )
            cache.close()

    def test_lru_eviction_avoids_full_cache_sort(self):
        import unified_translation_cache as cache_module

        cache = UnifiedTranslationCache(max_size=10)
        cache_keys = []
        for index in range(10):
            text = f"text-{index}"
            cache.store(text, "en", "zh-CN", "custom_ai", f"value-{index}")
            cache_keys.append(
                cache._generate_cache_key(text, "en", "zh-CN", "custom_ai")
            )

        with cache.lock:
            for index, cache_key in enumerate(cache_keys):
                cache._access_times[cache_key] = float(index)

        with patch.object(
            cache_module,
            "sorted",
            side_effect=AssertionError("full cache sort should not run"),
            create=True,
        ):
            cache.store("new", "en", "zh-CN", "custom_ai", "new-value")

        self.assertEqual(cache.get_stats()["total_entries"], 10)
        self.assertIsNone(cache.get("text-0", "en", "zh-CN", "custom_ai"))
        self.assertEqual(
            cache.get("text-1", "en", "zh-CN", "custom_ai"),
            "value-1",
        )
        self.assertEqual(
            cache.get("new", "en", "zh-CN", "custom_ai"),
            "new-value",
        )

    def test_persistent_cache_restores_entries_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "custom_ai_cache.json"
            cache = UnifiedTranslationCache(max_size=10, persistence_path=cache_path)
            cache.store(
                "Bonjour",
                "fr",
                "en",
                "custom_ai",
                "Hello again",
                profile_id="profile-1",
                base_url="https://host.example/v1",
                model="demo-model",
            )
            cache.flush()

            restored = UnifiedTranslationCache(max_size=10, persistence_path=cache_path)

            self.assertEqual(
                restored.get(
                    "Bonjour",
                    "fr",
                    "en",
                    "custom_ai",
                    profile_id="profile-1",
                    base_url="https://host.example/v1",
                    model="demo-model",
                ),
                "Hello again",
            )

    def test_persistent_cache_clear_removes_saved_entries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "custom_ai_cache.json"
            cache = UnifiedTranslationCache(max_size=10, persistence_path=cache_path)
            cache.store(
                "Bonjour",
                "fr",
                "en",
                "custom_ai",
                "Hello once",
                profile_id="profile-1",
                base_url="https://host.example/v1",
                model="demo-model",
            )

            cache.clear_all()
            restored = UnifiedTranslationCache(max_size=10, persistence_path=cache_path)

            self.assertIsNone(
                restored.get(
                    "Bonjour",
                    "fr",
                    "en",
                    "custom_ai",
                    profile_id="profile-1",
                    base_url="https://host.example/v1",
                    model="demo-model",
                )
            )

    def test_persistent_cache_clear_missing_provider_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "custom_ai_cache.json"
            cache = UnifiedTranslationCache(
                max_size=10,
                persistence_path=cache_path,
                persistence_delay_seconds=60.0,
            )

            cache.store("Bonjour", "fr", "en", "custom_ai", "Hello")
            self.assertTrue(cache.flush())
            persisted_generation = cache._persisted_generation

            cache.clear_provider("deepl_api")

            self.assertEqual(cache._persistence_generation, persisted_generation)
            self.assertIsNone(cache._persistence_timer)
            self.assertTrue(cache_path.exists())
            self.assertEqual(
                cache.get("Bonjour", "fr", "en", "custom_ai"),
                "Hello",
            )
            cache.close()

    def test_persistent_cache_store_is_deferred_until_flush(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "custom_ai_cache.json"
            cache = UnifiedTranslationCache(
                max_size=10,
                persistence_path=cache_path,
                persistence_delay_seconds=60.0,
            )

            cache.store("Bonjour", "fr", "en", "custom_ai", "Hello")

            self.assertFalse(cache_path.exists())
            self.assertTrue(cache.flush())
            self.assertTrue(cache_path.exists())
            cache.close()

    def test_persistent_cache_repeated_same_store_only_refreshes_lru(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "custom_ai_cache.json"
            cache = UnifiedTranslationCache(
                max_size=10,
                persistence_path=cache_path,
                persistence_delay_seconds=60.0,
            )

            try:
                cache.store("Hello", "en", "zh-CN", "custom_ai", "你好")
                self.assertTrue(cache.flush())
                cache_key = cache._generate_cache_key(
                    "Hello",
                    "en",
                    "zh-CN",
                    "custom_ai",
                )
                first_access_time = cache._access_times[cache_key]
                persisted_generation = cache._persisted_generation

                with patch("unified_translation_cache.time.time", return_value=first_access_time + 10.0):
                    cache.store("Hello", "en", "zh-CN", "custom_ai", "你好")

                self.assertEqual(cache._access_times[cache_key], first_access_time + 10.0)
                self.assertEqual(cache._persistence_generation, persisted_generation)
                self.assertIsNone(cache._persistence_timer)
            finally:
                cache.close()

    def test_persistent_cache_coalesces_multiple_stores(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "custom_ai_cache.json"
            cache = UnifiedTranslationCache(
                max_size=10,
                persistence_path=cache_path,
                persistence_delay_seconds=60.0,
            )

            cache.store("one", "en", "zh-CN", "custom_ai", "一")
            first_timer = cache._persistence_timer
            cache.store("two", "en", "zh-CN", "custom_ai", "二")

            self.assertIs(cache._persistence_timer, first_timer)
            self.assertTrue(cache.flush())
            restored = UnifiedTranslationCache(max_size=10, persistence_path=cache_path)
            self.assertEqual(restored.get("one", "en", "zh-CN", "custom_ai"), "一")
            self.assertEqual(restored.get("two", "en", "zh-CN", "custom_ai"), "二")
            cache.close()
            restored.close()

    def test_persistent_cache_close_flushes_pending_entries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "custom_ai_cache.json"
            cache = UnifiedTranslationCache(
                max_size=10,
                persistence_path=cache_path,
                persistence_delay_seconds=60.0,
            )

            cache.store("Bonjour", "fr", "en", "custom_ai", "Hello")
            cache.close()

            restored = UnifiedTranslationCache(max_size=10, persistence_path=cache_path)
            self.assertEqual(
                restored.get("Bonjour", "fr", "en", "custom_ai"),
                "Hello",
            )
            restored.close()

    def test_persistent_cache_store_after_close_is_synchronously_persisted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "custom_ai_cache.json"
            cache = UnifiedTranslationCache(
                max_size=10,
                persistence_path=cache_path,
                persistence_delay_seconds=60.0,
            )
            cache.close()

            cache.store("late", "en", "zh-CN", "custom_ai", "迟到")

            restored = UnifiedTranslationCache(max_size=10, persistence_path=cache_path)
            self.assertEqual(
                restored.get("late", "en", "zh-CN", "custom_ai"),
                "迟到",
            )
            restored.close()

    def test_persistent_cache_clear_cannot_be_resurrected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "custom_ai_cache.json"
            cache = UnifiedTranslationCache(
                max_size=10,
                persistence_path=cache_path,
                persistence_delay_seconds=60.0,
            )

            cache.store("Bonjour", "fr", "en", "custom_ai", "Hello")
            cache.clear_all()
            cache.close()

            self.assertFalse(cache_path.exists())
            restored = UnifiedTranslationCache(max_size=10, persistence_path=cache_path)
            self.assertIsNone(restored.get("Bonjour", "fr", "en", "custom_ai"))
            restored.close()

    def test_persistent_cache_ignores_legacy_schema(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "custom_ai_cache.json"
            legacy_cache = UnifiedTranslationCache(max_size=10)
            legacy_key = legacy_cache._generate_cache_key(
                "Bonjour",
                "fr",
                "en",
                "custom_ai",
            )
            cache_path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {
                                "key": list(legacy_key),
                                "translation": "Legacy value",
                                "access_time": time.time(),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            cache = UnifiedTranslationCache(max_size=10, persistence_path=cache_path)

            self.assertIsNone(cache.get("Bonjour", "fr", "en", "custom_ai"))
            cache.close()


class DummyVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class TranslationHandlerCustomAITests(unittest.TestCase):
    def test_inflight_key_isolated_by_wire_api_and_reasoning_effort(self):
        profile = {
            "id": "profile-1",
            "name": "Translator",
            "base_url": "https://host.example/v1/",
            "api_key": "super-secret",
            "model": "gpt-5.5",
            "wire_api": "responses",
            "reasoning_effort": "xhigh",
        }

        class Profiles:
            def get_active_profile(self, kind):
                return profile

        class App:
            custom_ai_profiles = Profiles()
            keep_linebreaks_var = DummyVar(False)
            source_lang_var = DummyVar("en")
            target_lang_var = DummyVar("zh-CN")
            custom_context_window_var = DummyVar(0)
            custom_prompt_text = ""

        handler = TranslationHandler(App())

        responses_xhigh_key = handler.get_inflight_translation_key("Hello")
        profile["wire_api"] = "chat_completions"
        chat_xhigh_key = handler.get_inflight_translation_key("Hello")
        profile["reasoning_effort"] = "low"
        chat_low_key = handler.get_inflight_translation_key("Hello")

        self.assertNotEqual(responses_xhigh_key, chat_xhigh_key)
        self.assertNotEqual(chat_xhigh_key, chat_low_key)
        handler.close()

    def test_close_flushes_cache_and_closes_provider(self):
        handler = TranslationHandler(object())
        events = []
        handler.unified_cache = Mock()
        handler.unified_cache.close.side_effect = lambda: events.append("cache")
        handler.custom_ai_provider = Mock()
        handler.custom_ai_provider.close.side_effect = lambda: events.append("provider")

        handler.close()

        self.assertEqual(events, ["cache", "provider"])

    def test_custom_ai_ocr_provider_errors_are_returned_visibly(self):
        class Profiles:
            def get_active_profile(self, kind):
                return {
                    "id": "profile-1",
                    "name": "Vision",
                    "base_url": "https://host.example/v1",
                    "api_key": "super-secret",
                    "model": "vision-model",
                }

        class App:
            custom_ai_profiles = Profiles()
            keep_linebreaks_var = DummyVar(False)

        handler = TranslationHandler(App())
        handler.custom_ai_provider.recognize = Mock(side_effect=ValueError("vision failed"))

        result = handler.perform_ocr(b"image", "en")

        self.assertTrue(result.startswith("<e>: Custom AI OCR error: ValueError - vision failed"))

    def test_custom_ai_translation_uses_configured_context_window_size(self):
        class Profiles:
            def get_active_profile(self, kind):
                return {
                    "id": "profile-1",
                    "name": "Translator",
                    "base_url": "https://host.example/v1",
                    "api_key": "super-secret",
                    "model": "translation-model",
                }

        class App:
            custom_ai_profiles = Profiles()
            keep_linebreaks_var = DummyVar(False)
            source_lang_var = DummyVar("en")
            target_lang_var = DummyVar("zh-CN")
            custom_prompt_text = ""

            def __init__(self, context_size):
                self.custom_context_window_var = DummyVar(context_size)

        for context_size, expected_context in [
            (0, []),
            (3, ["two", "three", "four"]),
            (5, ["one", "two", "three", "four"]),
        ]:
            with self.subTest(context_size=context_size):
                handler = TranslationHandler(App(context_size))
                handler.custom_context_window = ["one", "two", "three", "four"]
                handler.custom_ai_provider.translate = Mock(return_value=("translated", {}, 0.01))

                handler._custom_ai_translate("current", 0.0)

                passed_context = handler.custom_ai_provider.translate.call_args.kwargs["context"]
                self.assertEqual(passed_context, expected_context)

    def test_custom_ai_context_window_keeps_configured_history_limit(self):
        class App:
            custom_context_window_var = DummyVar(3)

        handler = TranslationHandler(App())

        for subtitle in ["one", "two", "three", "four", "five"]:
            handler._update_custom_context(subtitle)

        self.assertEqual(handler.custom_context_window, ["three", "four", "five"])

    def test_custom_ai_race_mode_uses_fastest_same_model_profile(self):
        slow = {
            "id": "slow",
            "name": "Slow Relay",
            "base_url": "https://slow.example/v1",
            "api_key": "slow-key",
            "model": "same-model",
        }
        fast = {
            "id": "fast",
            "name": "Fast Relay",
            "base_url": "https://fast.example/v1",
            "api_key": "fast-key",
            "model": "same-model",
        }
        other_model = {
            "id": "other",
            "name": "Other Model",
            "base_url": "https://other.example/v1",
            "api_key": "other-key",
            "model": "different-model",
        }

        class Profiles:
            def get_active_profile(self, kind):
                return slow

            def list_profiles(self, kind=None, enabled_only=False):
                return [slow, fast, other_model]

        class App:
            translation_model_var = DummyVar("custom_ai")
            custom_ai_profiles = Profiles()
            custom_source_lang = "ja"
            custom_target_lang = "en"
            source_lang_var = DummyVar("ja")
            target_lang_var = DummyVar("en")
            keep_linebreaks_var = DummyVar(False)
            custom_context_window_var = DummyVar(0)
            custom_ai_latency_mode_var = DummyVar("race")
            custom_prompt_text = ""

        handler = TranslationHandler(App())
        called_profile_ids = []

        def fake_translate(profile, *args, **kwargs):
            called_profile_ids.append(profile["id"])
            if profile["id"] == "slow":
                time.sleep(0.05)
                return "slow result", {}, 0.05
            return "fast result", {}, 0.01

        handler.custom_ai_provider.translate = Mock(side_effect=fake_translate)

        result = handler._custom_ai_translate("こんにちは", time.monotonic())

        self.assertEqual(result, "fast result")
        time.sleep(0.08)
        self.assertIn("slow", called_profile_ids)
        self.assertIn("fast", called_profile_ids)
        self.assertNotIn("other", called_profile_ids)

    def test_custom_ai_translation_uses_persistent_cache_between_handler_instances(self):
        profile = {
            "id": "profile-1",
            "name": "Translator",
            "base_url": "https://host.example/v1",
            "api_key": "super-secret",
            "model": "translation-model",
        }

        class Profiles:
            def get_active_profile(self, kind):
                return profile

        class App:
            translation_model_var = DummyVar("custom_ai")
            custom_ai_profiles = Profiles()
            custom_source_lang = "en"
            custom_target_lang = "zh-CN"
            source_lang_var = DummyVar("en")
            target_lang_var = DummyVar("zh-CN")
            keep_linebreaks_var = DummyVar(False)
            custom_context_window_var = DummyVar(0)
            custom_ai_latency_mode_var = DummyVar("safe")
            custom_prompt_text = ""

            def __init__(self, cache_path):
                self.custom_ai_translation_cache_file = cache_path

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "custom_ai_cache.json"

            first_handler = TranslationHandler(App(cache_path))
            first_handler.custom_ai_provider.translate = Mock(return_value=("translated", {}, 0.01))

            self.assertEqual(first_handler._custom_ai_translate("current", 0.0), "translated")
            first_handler.close()

            second_handler = TranslationHandler(App(cache_path))
            second_handler.custom_ai_provider.translate = Mock(side_effect=AssertionError("provider should not be called"))

            self.assertEqual(second_handler._custom_ai_translate("current", 0.0), "translated")
            second_handler.close()

    def test_custom_ai_translation_errors_are_not_persisted(self):
        profile = {
            "id": "profile-1",
            "name": "Translator",
            "base_url": "https://host.example/v1",
            "api_key": "super-secret",
            "model": "translation-model",
        }

        class Profiles:
            def get_active_profile(self, kind):
                return profile

        class App:
            translation_model_var = DummyVar("custom_ai")
            custom_ai_profiles = Profiles()
            custom_source_lang = "en"
            custom_target_lang = "zh-CN"
            source_lang_var = DummyVar("en")
            target_lang_var = DummyVar("zh-CN")
            keep_linebreaks_var = DummyVar(False)
            custom_context_window_var = DummyVar(0)
            custom_ai_latency_mode_var = DummyVar("safe")
            custom_prompt_text = ""

            def __init__(self, cache_path):
                self.custom_ai_translation_cache_file = cache_path

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "custom_ai_cache.json"

            first_handler = TranslationHandler(App(cache_path))
            first_handler.custom_ai_provider.translate = Mock(side_effect=ValueError("upstream busy"))

            first_result = first_handler._custom_ai_translate("current", 0.0)
            self.assertIn("Custom AI translation error", first_result)

            second_handler = TranslationHandler(App(cache_path))
            second_handler.custom_ai_provider.translate = Mock(return_value=("translated", {}, 0.01))

            self.assertEqual(second_handler._custom_ai_translate("current", 0.0), "translated")


class ModelFilterTests(unittest.TestCase):
    def test_filter_model_values_matches_case_insensitive_substrings(self):
        models = ["openai/gpt-5.5", "deepseek/deepseek-v4", "Google/Gemini-3.5"]

        self.assertEqual(filter_model_values(models, "5.5"), ["openai/gpt-5.5"])
        self.assertEqual(filter_model_values(models, "GEMINI"), ["Google/Gemini-3.5"])
        self.assertEqual(filter_model_values(models, ""), models)


class ProfileFormValuesTests(unittest.TestCase):
    def test_profile_form_values_preserve_selected_responses_metadata_when_name_is_edited(self):
        selected_profile = {
            "id": "profile-1",
            "name": "\u6d4b\u8bd51",
            "base_url": "https://api.e2ez.com",
            "api_key": "old-secret",
            "model": "gpt-5.4",
            "enabled": True,
            "wire_api": "responses",
            "reasoning_effort": "xhigh",
        }

        class Profiles:
            def get_profile(self, profile_id):
                return selected_profile if profile_id == selected_profile["id"] else None

            def list_profiles(self):
                return [selected_profile]

        app = types.SimpleNamespace(
            custom_ai_profiles=Profiles(),
            ai_profile_selected_id="profile-1",
            ai_profile_name_var=DummyVar("222"),
            ai_profile_url_var=DummyVar("https://ebbandflow.online/v1"),
            ai_profile_key_var=DummyVar("new-secret"),
            ai_profile_model_var=DummyVar("gpt-5.4"),
        )

        values = gui_builder.build_custom_ai_profile_values_from_form(app)

        self.assertEqual(values["name"], "222")
        self.assertEqual(values["base_url"], "https://ebbandflow.online/v1")
        self.assertEqual(values["api_key"], "new-secret")
        self.assertEqual(values["model"], "gpt-5.4")
        self.assertEqual(values["wire_api"], "responses")
        self.assertEqual(values["reasoning_effort"], "xhigh")


class UILanguageManagerTests(unittest.TestCase):
    def test_chinese_display_name_is_not_mojibake_and_legacy_name_still_loads(self):
        manager = UILanguageManager()

        self.assertEqual(manager.get_available_languages()["zh"], "\u4e2d\u6587")
        self.assertEqual(manager.get_language_code_from_name("\u4e2d\u6587"), "zh")
        self.assertEqual(manager.get_language_code_from_name("\u6d93\ue15f\u6783"), "zh")
        self.assertEqual(manager.normalize_display_name("\u6d93\ue15f\u6783"), "\u4e2d\u6587")


class ProfileNetworkTaskTests(unittest.TestCase):
    def test_profile_network_task_starts_background_thread_without_running_task_on_ui_thread(self):
        class FakeThread:
            instances = []

            def __init__(self, target=None, name=None, daemon=None):
                self.target = target
                self.name = name
                self.daemon = daemon
                self.started = False
                FakeThread.instances.append(self)

            def start(self):
                self.started = True

        app = types.SimpleNamespace(
            root=types.SimpleNamespace(after=lambda *args: None),
        )
        button = types.SimpleNamespace(config=Mock())
        task = Mock(return_value="OK")

        with unittest.mock.patch("gui_builder.threading.Thread", FakeThread):
            run_profile_network_task_async(
                app,
                button,
                task,
                on_success=Mock(),
                failure_title="Connection Test Failed",
            )

        task.assert_not_called()
        self.assertEqual(len(FakeThread.instances), 1)
        self.assertTrue(FakeThread.instances[0].started)
        self.assertTrue(FakeThread.instances[0].daemon)
        button.config.assert_called_with(state="disabled")

    def test_profile_network_task_schedules_success_back_on_ui_thread(self):
        scheduled = []

        class FakeThread:
            instance = None

            def __init__(self, target=None, name=None, daemon=None):
                self.target = target
                FakeThread.instance = self

            def start(self):
                return None

        app = types.SimpleNamespace(
            root=types.SimpleNamespace(after=lambda delay, callback, *args: scheduled.append((delay, callback, args))),
        )
        button = types.SimpleNamespace(config=Mock())
        on_success = Mock()

        with unittest.mock.patch("gui_builder.threading.Thread", FakeThread):
            run_profile_network_task_async(
                app,
                button,
                task=lambda: ("OK", 0.1),
                on_success=on_success,
                failure_title="Connection Test Failed",
            )

        FakeThread.instance.target()
        delay, callback, args = scheduled[0]
        self.assertEqual(delay, 0)

        callback(*args)

        on_success.assert_called_once_with(("OK", 0.1))
        button.config.assert_any_call(state="normal")


if __name__ == "__main__":
    unittest.main()
