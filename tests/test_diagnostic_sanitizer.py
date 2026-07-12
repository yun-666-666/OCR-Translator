import unittest

from custom_ai import CustomAIProvider
from diagnostic_sanitizer import sanitize_error_text, sanitize_url


class RequestsLikeFakeResponse:
    def __init__(self, status_code=200, headers=None, text="", json_payload=None, json_error=None):
        self.status_code = status_code
        self.headers = dict(headers or {})
        self.text = text
        self._json_payload = json_payload
        self._json_error = json_error
        self.closed = False

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._json_payload

    def close(self):
        self.closed = True


class SanitizeUrlTests(unittest.TestCase):
    def test_removes_credentials_query_and_fragment(self):
        self.assertEqual(
            sanitize_url(
                "https://user:pass@example.com:8443/v1/chat?token=query-secret#frag"
            ),
            "https://example.com:8443/v1/chat",
        )

    def test_normalizes_default_port_and_ipv6_host(self):
        self.assertEqual(
            sanitize_url("HTTP://Example.COM:80/status?verbose=true"),
            "http://example.com/status",
        )
        self.assertEqual(
            sanitize_url("https://[2001:DB8::1]:444/v1#details"),
            "https://[2001:db8::1]:444/v1",
        )

    def test_malformed_url_is_replaced_without_leaking_input(self):
        malformed = "https://user:pass@[::1/v1?token=unique-leak"
        result = sanitize_url(malformed)

        self.assertEqual(result, "[redacted-url]")
        self.assertNotIn("unique-leak", result)


class SanitizeErrorTextTests(unittest.TestCase):
    def test_redacts_secrets_named_fields_urls_and_response_content(self):
        marker = "UNIQUE_RESPONSE_CONTENT_91F3"
        message = "\n".join(
            (
                "request failed using known-value",
                "Authorization: Bearer authorization-value",
                "Cookie: session=cookie-value",
                'api_key="api-key-value"',
                "token: token-value",
                "secret=secret-value",
                "password: password-value",
                f"response_body: {marker}",
                "content='content-value'",
                "endpoint https://url-user:url-pass@Example.COM:443/v1/chat?token=url-secret#frag",
            )
        )

        result = sanitize_error_text(message, known_secrets=("known-value",))

        for secret_value in (
            "known-value",
            "authorization-value",
            "cookie-value",
            "api-key-value",
            "token-value",
            "secret-value",
            "password-value",
            marker,
            "content-value",
            "url-user",
            "url-pass",
            "url-secret",
        ):
            self.assertNotIn(secret_value, result)
        for field_name in (
            "Authorization",
            "Cookie",
            "api_key",
            "token",
            "secret",
            "password",
            "response_body",
            "content",
        ):
            self.assertIn(f"{field_name}: [redacted]", result.replace("=", ": ").replace(":  ", ": "))
        self.assertIn("https://example.com/v1/chat", result)
        self.assertNotIn("\n", result)

    def test_output_never_exceeds_positive_max_length(self):
        for max_length in (1, 2, 3, 4, 25):
            with self.subTest(max_length=max_length):
                result = sanitize_error_text("x" * 100, max_length=max_length)
                self.assertLessEqual(len(result), max_length)
        self.assertTrue(sanitize_error_text("x" * 100, max_length=25).endswith("..."))

    def test_redacts_escaped_quotes_inside_quoted_named_fields(self):
        result = sanitize_error_text(
            "\n".join(
                (
                    'content="UNIQUE_DOUBLE_\\"DOUBLE_TAIL_MARKER"',
                    "secret='UNIQUE_SINGLE_\\'SINGLE_TAIL_MARKER'",
                )
            )
        )

        self.assertNotIn("DOUBLE_TAIL_MARKER", result)
        self.assertNotIn("SINGLE_TAIL_MARKER", result)
        self.assertIn("content=[redacted]", result)
        self.assertIn("secret=[redacted]", result)

    def test_redacts_entire_cookie_remainder(self):
        result = sanitize_error_text(
            "Cookie: session=abc; csrftoken=PRIVATE_COOKIE_TAIL"
        )

        self.assertEqual(result, "Cookie: [redacted]")
        self.assertNotIn("PRIVATE_COOKIE_TAIL", result)

    def test_redacts_entire_content_remainder(self):
        result = sanitize_error_text(
            'content={"first":"safe","second":"PRIVATE_CONTENT_TAIL"}'
        )

        self.assertEqual(result, "content=[redacted]")
        self.assertNotIn("PRIVATE_CONTENT_TAIL", result)

    def test_redacts_entire_response_body_remainder(self):
        result = sanitize_error_text(
            'response_body={"message":"PRIVATE_BODY_TAIL","code":500}'
        )

        self.assertEqual(result, "response_body=[redacted]")
        self.assertNotIn("PRIVATE_BODY_TAIL", result)

    def test_redacts_api_key_field_name_variants(self):
        cases = (
            ("api_key", "PRIVATE_UNDERSCORE_KEY"),
            ("api-key", "PRIVATE_DASH_KEY"),
            ("apikey", "PRIVATE_COMPACT_KEY"),
        )
        for field_name, marker in cases:
            with self.subTest(field_name=field_name):
                result = sanitize_error_text(f"{field_name}={marker}")
                self.assertEqual(result, f"{field_name}=[redacted]")
                self.assertNotIn(marker, result)

    def test_none_known_secrets_is_treated_as_empty(self):
        self.assertEqual(
            sanitize_error_text("token=PRIVATE_TOKEN", known_secrets=None),
            "token=[redacted]",
        )

    def test_non_positive_max_length_disables_truncation(self):
        self.assertEqual(sanitize_error_text("a   b", max_length=0), "a b")


class CustomAITransportSanitizationTests(unittest.TestCase):
    def setUp(self):
        self.provider = CustomAIProvider(http_client=object())
        self.api_key = "UNIQUE_API_KEY_629B"
        self.unsafe_url = (
            "https://url-user:url-pass@Example.COM:443/v1/chat"
            "?token=UNIQUE_QUERY_TOKEN_1D6C#UNIQUE_FRAGMENT_8A42"
        )

    def assert_no_leaks(self, text, *markers):
        for marker in markers:
            self.assertNotIn(marker, text)

    def test_sanitize_error_redacts_url_credentials_and_named_secret_fields(self):
        message = "\n".join(
            (
                f"endpoint={self.unsafe_url}",
                f"api_key={self.api_key}",
                "Authorization: Bearer UNIQUE_AUTHORIZATION_38E1",
                "Cookie: session=UNIQUE_COOKIE_C94A",
                "token=UNIQUE_TOKEN_FIELD_592D",
                "password=UNIQUE_PASSWORD_A50E",
            )
        )

        result = self.provider._sanitize_error(message, self.api_key)

        self.assertIn("https://example.com/v1/chat", result)
        self.assert_no_leaks(
            result,
            "url-user",
            "url-pass",
            "UNIQUE_QUERY_TOKEN_1D6C",
            "UNIQUE_FRAGMENT_8A42",
            self.api_key,
            "UNIQUE_AUTHORIZATION_38E1",
            "UNIQUE_COOKIE_C94A",
            "UNIQUE_TOKEN_FIELD_592D",
            "UNIQUE_PASSWORD_A50E",
        )

    def test_structured_http_error_exposes_only_safe_type_and_code(self):
        message_marker = "UNIQUE_STRUCTURED_MESSAGE_F117"
        response = RequestsLikeFakeResponse(
            status_code=400,
            headers={"Content-Type": "application/json"},
            text=f'{{"error":{{"message":"{message_marker}"}}}}',
            json_payload={
                "error": {
                    "message": message_marker,
                    "type": "invalid_request_error",
                    "code": "unsupported_model",
                    "password": "UNIQUE_STRUCTURED_PASSWORD_2704",
                }
            },
        )

        result = self.provider._response_error_message(
            response,
            self.unsafe_url,
            self.api_key,
        )

        self.assertIn("HTTP 400", result)
        self.assertIn("https://example.com/v1/chat", result)
        self.assertIn("type=invalid_request_error", result)
        self.assertIn("code=unsupported_model", result)
        self.assert_no_leaks(
            result,
            message_marker,
            "UNIQUE_STRUCTURED_PASSWORD_2704",
            "url-user",
            "url-pass",
            "UNIQUE_QUERY_TOKEN_1D6C",
            "UNIQUE_FRAGMENT_8A42",
        )

    def test_structured_http_error_omits_provider_controlled_identifiers(self):
        unsafe_cases = (
            ("subtitle-private-marker", "response-content-marker"),
            (
                "private marker https://url-user:url-pass@example.com/v1"
                "?token=UNIQUE_TYPE_QUERY_3C91#UNIQUE_TYPE_FRAGMENT_957A",
                "password=UNIQUE_CODE_PASSWORD_538F super-secret",
            ),
        )
        for error_type, error_code in unsafe_cases:
            with self.subTest(error_type=error_type, error_code=error_code):
                response = RequestsLikeFakeResponse(
                    status_code=400,
                    headers={"Content-Type": "application/json"},
                    text="",
                    json_payload={
                        "error": {
                            "message": "UNIQUE_PRIVATE_MESSAGE_433C",
                            "type": error_type,
                            "code": error_code,
                        }
                    },
                )

                result = self.provider._response_error_message(
                    response,
                    self.unsafe_url,
                    "super-secret",
                )

                self.assertNotIn("type=", result)
                self.assertNotIn("code=", result)
                for marker in (
                    "subtitle-private-marker",
                    "response-content-marker",
                    "private marker",
                    "url-user",
                    "url-pass",
                    "UNIQUE_TYPE_QUERY_3C91",
                    "UNIQUE_TYPE_FRAGMENT_957A",
                    "UNIQUE_CODE_PASSWORD_538F",
                    "super-secret",
                ):
                    self.assertNotIn(marker, result)

        safe_response = RequestsLikeFakeResponse(
            status_code=429,
            headers={"Content-Type": "application/json"},
            text="",
            json_payload={
                "error": {
                    "message": "UNIQUE_PRIVATE_MESSAGE_CE77",
                    "type": "rate_limit_error",
                    "code": 429,
                }
            },
        )

        safe_result = self.provider._response_error_message(
            safe_response,
            self.unsafe_url,
            self.api_key,
        )

        self.assertIn("type=rate_limit_error", safe_result)
        self.assertIn("code=429", safe_result)
        self.assertNotIn("UNIQUE_PRIVATE_MESSAGE_CE77", safe_result)

    def test_non_json_response_never_exposes_body_or_html_title(self):
        cases = (
            (
                "text",
                {"Content-Type": "text/plain"},
                "UNIQUE_NON_JSON_BODY_6FCE",
            ),
            (
                "html",
                {"Content-Type": "text/html; charset=UTF-8"},
                "<html><head><title>UNIQUE_HTML_TITLE_858E</title></head>"
                "<body>UNIQUE_HTML_BODY_9D8C</body></html>",
            ),
        )
        for label, headers, body in cases:
            with self.subTest(label=label):
                response = RequestsLikeFakeResponse(
                    status_code=200,
                    headers=headers,
                    text=body,
                    json_error=ValueError("not json"),
                )

                result = self.provider._non_json_response_message(
                    response,
                    self.unsafe_url,
                    self.api_key,
                )

                self.assertIn("https://example.com/v1/chat", result)
                self.assert_no_leaks(
                    result,
                    "UNIQUE_NON_JSON_BODY_6FCE",
                    "UNIQUE_HTML_TITLE_858E",
                    "UNIQUE_HTML_BODY_9D8C",
                    "url-user",
                    "url-pass",
                    "UNIQUE_QUERY_TOKEN_1D6C",
                    "UNIQUE_FRAGMENT_8A42",
                )

    def test_endpoint_aggregation_sanitizes_urls_and_exceptions(self):
        exception_marker = "UNIQUE_ENDPOINT_EXCEPTION_491A"

        class RaisingClient:
            def post(self, url, headers=None, json=None, timeout=None, stream=False):
                raise RuntimeError(
                    f"response_body={exception_marker}\n"
                    "Authorization: Bearer UNIQUE_ENDPOINT_AUTH_6B49"
                )

        provider = CustomAIProvider(http_client=RaisingClient())
        base_profile = {
            "base_url": self.unsafe_url,
            "api_key": self.api_key,
            "model": "demo",
        }
        payload = {"model": "demo", "messages": []}
        calls = (
            ("chat", lambda: provider._post(base_profile, payload)),
            (
                "responses",
                lambda: provider._responses_post(
                    dict(base_profile, wire_api="responses"),
                    payload,
                    provider.http_client,
                    {},
                    "safe",
                ),
            ),
            ("chat_stream", lambda: provider._stream_post(base_profile, payload)),
            (
                "responses_stream",
                lambda: provider._stream_responses_post(
                    dict(base_profile, wire_api="responses"),
                    payload,
                    http_client=provider.http_client,
                    headers={},
                ),
            ),
        )

        for label, call in calls:
            with self.subTest(label=label), self.assertRaises(ValueError) as ctx:
                call()
            result = str(ctx.exception)
            self.assertIn("https://example.com/v1/chat", result)
            self.assert_no_leaks(
                result,
                exception_marker,
                "UNIQUE_ENDPOINT_AUTH_6B49",
                "url-user",
                "url-pass",
                "UNIQUE_QUERY_TOKEN_1D6C",
                "UNIQUE_FRAGMENT_8A42",
            )


if __name__ == "__main__":
    unittest.main()
