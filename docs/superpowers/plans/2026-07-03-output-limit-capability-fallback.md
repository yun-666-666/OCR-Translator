# Output-Limit Capability Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retry once without an unsupported output-limit field and remember that capability for later translations.

**Architecture:** `CustomAIProvider` will maintain a process-local set keyed by wire API, base URL, and model. A shared post helper will apply capability memory and perform the single compatibility retry for all request paths.

**Tech Stack:** Python, requests-compatible clients, `unittest`

---

### Task 1: Chat Completions capability memory

**Files:**
- Modify: `custom_ai.py`
- Test: `tests/test_custom_ai.py`

- [x] **Step 1: Write the failing Chat test**

```python
def test_chat_output_limit_rejection_retries_once_and_remembers_capability(self):
    class Response:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self.payload = payload
            self.text = json.dumps(payload)
            self.headers = {}

        def json(self):
            return self.payload

    class Client:
        def __init__(self):
            self.payloads = []

        def post(self, url, headers=None, json=None, timeout=None):
            self.payloads.append(dict(json))
            if "max_tokens" in json:
                return Response(
                    400,
                    {"error": {"message": "Unsupported parameter: max_tokens"}},
                )
            return Response(
                200,
                {"choices": [{"message": {"content": "OK"}}]},
            )

    client = Client()
    provider = CustomAIProvider(http_client=client)
    profile = {
        "base_url": "https://host.example/v1",
        "api_key": "secret",
        "model": "demo",
    }
    payload = {"model": "demo", "messages": [], "max_tokens": 64}

    first, _ = provider._post(profile, payload)
    second, _ = provider._post(profile, payload)

    self.assertEqual(first["choices"][0]["message"]["content"], "OK")
    self.assertEqual(second["choices"][0]["message"]["content"], "OK")
    self.assertEqual(len(client.payloads), 3)
    self.assertIn("max_tokens", client.payloads[0])
    self.assertNotIn("max_tokens", client.payloads[1])
    self.assertNotIn("max_tokens", client.payloads[2])
```

- [x] **Step 2: Verify RED**

Run:
`python -m unittest tests.test_custom_ai.CustomAIProviderTests.test_chat_output_limit_rejection_retries_once_and_remembers_capability -v`

Expected: FAIL because the first 400 response is returned without a same-URL
compatibility retry.

- [x] **Step 3: Implement capability helpers**

```python
def _output_limit_capability_key(self, profile):
    return (
        normalize_custom_ai_wire_api(profile.get("wire_api")),
        self._base_url_cache_key(profile.get("base_url")),
        str(profile.get("model") or "").strip(),
    )

def _output_limit_field(self, profile):
    if self._uses_responses_api(profile):
        return "max_output_tokens"
    return "max_tokens"
```

Add a lock, an unsupported-capability set, explicit rejection detection, payload
copy/removal, and a shared `_post_with_output_limit_fallback(...)` helper.

- [x] **Step 4: Route Chat through the helper and verify GREEN**

Replace the direct Chat `http_client.post(...)` call with the helper. Run the
Step 2 command and expect PASS.

### Task 2: Streaming Responses and false-positive protection

**Files:**
- Modify: `custom_ai.py`
- Test: `tests/test_custom_ai.py`

- [x] **Step 1: Write the failing streaming Responses test**

```python
def test_streaming_responses_output_limit_rejection_retries_and_remembers(self):
    class Response:
        def __init__(self, rejected):
            self.status_code = 400 if rejected else 200
            self.text = json.dumps({
                "error": {"message": "Unknown parameter: max_output_tokens"}
            })
            self.headers = {}
            self.rejected = rejected

        def json(self):
            return json.loads(self.text)

        def iter_lines(self, decode_unicode=False):
            return iter([
                'data: {"type":"response.output_text.delta","delta":"OK"}',
                'data: [DONE]',
            ])

    class Client:
        def __init__(self):
            self.payloads = []
            self.stream_flags = []

        def post(
            self, url, headers=None, json=None, timeout=None, stream=False
        ):
            self.payloads.append(dict(json))
            self.stream_flags.append(stream)
            return Response("max_output_tokens" in json)

    client = Client()
    provider = CustomAIProvider(http_client=client)
    profile = {
        "base_url": "https://host.example/v1",
        "api_key": "secret",
        "model": "demo",
        "wire_api": "responses",
    }
    payload = {"model": "demo", "messages": [], "max_tokens": 64}

    first, _ = provider._stream_post(profile, payload)
    second, _ = provider._stream_post(profile, payload)

    self.assertEqual(first["output_text"], "OK")
    self.assertEqual(second["output_text"], "OK")
    self.assertEqual(client.stream_flags, [True, True, True])
    self.assertIn("max_output_tokens", client.payloads[0])
    self.assertNotIn("max_output_tokens", client.payloads[1])
    self.assertNotIn("max_output_tokens", client.payloads[2])
```

- [x] **Step 2: Verify RED**

Run the named streaming Responses test. Expected: the request does not recover.

- [x] **Step 3: Route remaining paths through the helper**

Use `_post_with_output_limit_fallback(...)` in `_responses_post`,
`_stream_post`, and `_stream_responses_post`, passing `stream=True` for both
streaming paths.

- [x] **Step 4: Verify focused compatibility behavior**

First add this false-positive regression:

```python
def test_output_limit_value_error_does_not_disable_capability(self):
    class Response:
        status_code = 400
        text = json.dumps({
            "error": {"message": "max_tokens must be less than or equal to 32"}
        })
        headers = {}

        def json(self):
            return json.loads(self.text)

    class Client:
        def __init__(self):
            self.calls = 0

        def post(self, url, headers=None, json=None, timeout=None):
            self.calls += 1
            return Response()

    client = Client()
    provider = CustomAIProvider(http_client=client)
    with self.assertRaises(ValueError):
        provider._post(
            {
                "base_url": "https://host.example/v1",
                "api_key": "secret",
                "model": "demo",
            },
            {"model": "demo", "messages": [], "max_tokens": 64},
        )
    self.assertEqual(client.calls, 1)
    self.assertFalse(provider._unsupported_output_limit_keys)
```

Run the three new capability tests plus existing Chat/Responses streaming,
deterministic fail-fast, 404 fallback, and 502 recovery tests. Expect all PASS.

### Task 3: Explicit truncation recovery

**Files:**
- Modify: `custom_ai.py`
- Test: `tests/test_custom_ai.py`

- [x] **Step 1: Write failing Chat and Responses truncation tests**

```python
def test_translation_retries_without_output_limit_after_explicit_truncation(self):
    cases = [
        (
            "chat_completions",
            {"choices": [{
                "message": {"content": "partial"},
                "finish_reason": "length",
            }]},
        ),
        (
            "responses",
            {
                "output_text": "partial",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        ),
    ]
    # For each wire API, the first payload contains its output-limit field,
    # the second omits it, and translate() returns "complete".
```

- [x] **Step 2: Verify RED**

Run:
`python -m unittest tests.test_custom_ai.CustomAIProviderTests.test_translation_retries_without_output_limit_after_explicit_truncation -v`

Expected: FAIL because `translate()` returns `partial` after one request.

- [x] **Step 3: Implement one-shot truncation recovery**

```python
def _response_was_output_limited(self, profile, response_json):
    if self._uses_responses_api(profile):
        details = response_json.get("incomplete_details") or {}
        return (
            response_json.get("status") == "incomplete"
            and details.get("reason") == "max_output_tokens"
        )
    choices = response_json.get("choices") or []
    return bool(choices and choices[0].get("finish_reason") == "length")
```

In `translate()`, when this predicate is true and the original payload contains
`max_tokens`, copy the payload, remove `max_tokens`, retry through the same
streaming or non-streaming path once, and add both request durations.

- [x] **Step 4: Verify GREEN**

Run the named truncation test plus the output-limit capability tests and expect all
PASS.

### Task 4: Full verification and handoff

**Files:**
- Create: `.codex/handoffs/2026-07-03_17-50-39.md`

- [x] Run `python -m unittest tests.test_custom_ai -q`.
- [x] Run `python -m unittest tests.test_latency_optimization -q`.
- [x] Run `python -m unittest discover -s tests -q`.
- [x] Run `python -m unittest discover -q`.
- [x] Run `python -m compileall -q custom_ai.py handlers worker_threads.py tests`.
- [x] Run `git diff --check`.
- [x] Record RED/GREEN evidence, backup path, verification counts, and remaining
  real-network limitations in `.codex/handoffs/2026-07-03_17-50-39.md`.
