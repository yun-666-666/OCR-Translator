import unittest

from diagnostic_sanitizer import sanitize_error_text, sanitize_url


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


if __name__ == "__main__":
    unittest.main()
