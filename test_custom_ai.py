import json
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

from custom_ai import CustomAIProfileManager, CustomAIProvider
from gui_builder import filter_model_values
from unified_translation_cache import UnifiedTranslationCache


translation_handler_spec = importlib.util.spec_from_file_location(
    "translation_handler_for_tests",
    Path(__file__).resolve().parent / "handlers" / "translation_handler.py",
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

    def test_post_reports_content_free_json_error(self):
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

        message = str(ctx.exception)
        self.assertIn("HTTP 400", message)
        self.assertIn("https://host.example/v1/chat/completions", message)
        self.assertNotIn("bad model", message)
        self.assertNotIn('{"error"', message)
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


class DummyVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class TranslationHandlerCustomAITests(unittest.TestCase):
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


class ModelFilterTests(unittest.TestCase):
    def test_filter_model_values_matches_case_insensitive_substrings(self):
        models = ["openai/gpt-5.5", "deepseek/deepseek-v4", "Google/Gemini-3.5"]

        self.assertEqual(filter_model_values(models, "5.5"), ["openai/gpt-5.5"])
        self.assertEqual(filter_model_values(models, "GEMINI"), ["Google/Gemini-3.5"])
        self.assertEqual(filter_model_values(models, ""), models)


if __name__ == "__main__":
    unittest.main()
