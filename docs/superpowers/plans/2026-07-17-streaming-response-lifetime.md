# Streaming Response Lifetime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every Custom AI HTTP response as soon as the layer that owns it stops using it, without changing request, retry, parsing, or provider behavior.

**Architecture:** `_post_with_output_limit_fallback` owns only compatibility-rejected responses and closes each one before replacement; its final response remains caller-owned. `_stream_post` and `_stream_responses_post` own the final streamed response and close it in a per-endpoint `finally` block. Existing `_close_response_quietly` preserves the original result or exception if cleanup itself fails.

**Tech Stack:** Python 3, `requests`-compatible response/session objects, `unittest`, existing `CustomAIProvider` test fakes.

---

### Task 1: Prove compatibility-fallback response ownership

**Files:**
- Modify: `tests/test_custom_ai.py`
- Modify: `custom_ai_transport.py:81-231`

- [ ] **Step 1: Write the failing abandoned-response test**

Add a focused test whose two fake responses expose `close_calls`. The first response rejects `prompt_cache_key`; the second succeeds. Call `_post_with_output_limit_fallback(..., stream=True)` and assert the rejected response is closed once while the returned final response is still open:

```python
def test_output_limit_fallback_closes_abandoned_response_only(self):
    class Response:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self.payload = payload
            self.text = json.dumps(payload)
            self.close_calls = 0

        def json(self):
            return self.payload

        def close(self):
            self.close_calls += 1

    rejected = Response(
        400,
        {"error": {"message": "Unknown parameter: prompt_cache_key"}},
    )
    accepted = Response(200, {"choices": [{"message": {"content": "OK"}}]})

    class Client:
        def __init__(self):
            self.responses = [rejected, accepted]

        def post(self, _url, **_kwargs):
            return self.responses.pop(0)

    provider = CustomAIProvider(http_client=Client())
    profile = {
        "base_url": "https://host.example/v1",
        "api_key": "super-secret",
        "model": "demo",
    }

    response = provider._post_with_output_limit_fallback(
        provider.http_client,
        "https://host.example/v1/chat/completions",
        {},
        {
            "model": "demo",
            "messages": [],
            "prompt_cache_key": "cache-key",
        },
        profile,
        profile["api_key"],
        stream=True,
        request_kind="translation",
    )

    self.assertIs(response, accepted)
    self.assertEqual(rejected.close_calls, 1)
    self.assertEqual(accepted.close_calls, 0)
```

- [ ] **Step 2: Run the test and confirm RED**

Run:

```powershell
python -m unittest tests.test_custom_ai.CustomAIProviderTests.test_output_limit_fallback_closes_abandoned_response_only -v
```

Expected: `FAIL` because `rejected.close_calls` is `0`.

- [ ] **Step 3: Implement minimal compatibility cleanup**

In `_post_with_output_limit_fallback`, initialize `response` defensively and add a small local replacement helper so all four retry branches share exactly one cleanup rule:

```python
response = None

def replace_response(current_payload):
    nonlocal response
    self._close_response_quietly(response)
    response = None
    response = send(current_payload)

try:
    response = send(request_payload)
    # Keep the existing four compatibility checks and capability memory.
    # In each retry branch call replace_response(request_payload).
    return response
except Exception:
    self._close_response_quietly(response)
    raise
```

Move the existing compatibility loop and the post-loop reasoning-memory update inside the shown `try`. Replace each of the four compatibility `response = send(request_payload)` reassignments with `replace_response(request_payload)`. The normal `return response` stays inside the `try`, so it leaves the final response open; only an exception takes the cleanup branch.

- [ ] **Step 4: Run the focused test and confirm GREEN**

Run the command from Step 2. Expected: `OK`.

### Task 2: Prove streaming-wrapper cleanup

**Files:**
- Modify: `tests/test_custom_ai.py`
- Modify: `custom_ai_transport.py:1025-1221`

- [ ] **Step 1: Write failing Chat and Responses streaming tests**

Add two tests with fake streaming responses that implement `close()` and return valid SSE lines. The Chat test returns a delta plus `[DONE]`. The Responses test returns a text delta plus `response.completed`. After `_stream_post`, assert parsed text and exactly one close call for the produced response.

```python
def test_stream_post_closes_successful_chat_response(self):
    class Response:
        status_code = 200

        def __init__(self):
            self.close_calls = 0

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=False):
            return iter([
                'data: {"choices":[{"delta":{"content":"OK"}}]}',
                "data: [DONE]",
            ])

        def close(self):
            self.close_calls += 1

    response = Response()

    class Client:
        def post(self, _url, **_kwargs):
            return response

    provider = CustomAIProvider(http_client=Client())
    result, _duration = provider._stream_post(
        {"base_url": "https://host.example/v1", "api_key": "super-secret"},
        {"model": "demo", "messages": []},
    )

    self.assertEqual(result["choices"][0]["message"]["content"], "OK")
    self.assertEqual(response.close_calls, 1)
```

Add the complete Responses variant:

```python
def test_stream_post_closes_successful_responses_response(self):
    class Response:
        status_code = 200

        def __init__(self):
            self.close_calls = 0

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=False):
            return iter([
                'data: {"type":"response.output_text.delta","delta":"OK"}',
                (
                    'data: {"type":"response.completed",'
                    '"response":{"status":"completed"}}'
                ),
            ])

        def close(self):
            self.close_calls += 1

    response = Response()

    class Client:
        def post(self, _url, **_kwargs):
            return response

    provider = CustomAIProvider(http_client=Client())
    result, _duration = provider._stream_post(
        {
            "base_url": "https://host.example/v1",
            "api_key": "super-secret",
            "wire_api": "responses",
        },
        {"model": "demo", "messages": []},
    )

    self.assertEqual(result["output_text"], "OK")
    self.assertEqual(response.close_calls, 1)
```

- [ ] **Step 2: Run both tests and confirm RED**

Run:

```powershell
python -m unittest tests.test_custom_ai.CustomAIProviderTests.test_stream_post_closes_successful_chat_response tests.test_custom_ai.CustomAIProviderTests.test_stream_post_closes_successful_responses_response -v
```

Expected: both fail because each `close_calls` value is `0`.

- [ ] **Step 3: Implement per-endpoint `finally` cleanup**

In each streaming URL loop, initialize `response = None` before `try` and add:

```python
finally:
    self._close_response_quietly(response)
```

Keep parsing, error aggregation, cooldown, URL memory, transport recovery, and return values unchanged. Python executes `finally` before a `return`, `continue`, `break`, or propagated exception, covering every exit without duplicating cleanup branches.

- [ ] **Step 4: Run all three response-lifetime tests**

Run:

```powershell
python -m unittest tests.test_custom_ai.CustomAIProviderTests.test_output_limit_fallback_closes_abandoned_response_only tests.test_custom_ai.CustomAIProviderTests.test_stream_post_closes_successful_chat_response tests.test_custom_ai.CustomAIProviderTests.test_stream_post_closes_successful_responses_response -v
```

Expected: `Ran 3 tests ... OK`.

### Task 3: Validate, document, and commit round 1

**Files:**
- Create: `.codex/handoffs/2026-07-17_22-25-38.md`
- Verify: `custom_ai_transport.py`
- Verify: `tests/test_custom_ai.py`

- [ ] **Step 1: Run related and full tests**

```powershell
python -m unittest tests.test_custom_ai.CustomAIProviderTests -q
python -m unittest discover -s tests -q
```

Expected: both commands exit `0` with no failures.

- [ ] **Step 2: Run compile and diff hygiene**

```powershell
python -m py_compile custom_ai_transport.py tests/test_custom_ai.py
git diff --check
```

Expected: both commands exit `0`.

- [ ] **Step 3: Write the handoff**

Record the task summary, exact files, backup directory, RED/GREEN commands and results, full verification results, the response-ownership decision, Grok round-1 report path, and deferred findings.

- [ ] **Step 4: Stage only round-1 files and commit**

```powershell
git add -- custom_ai_transport.py tests/test_custom_ai.py docs/superpowers/plans/2026-07-17-streaming-response-lifetime.md .codex/handoffs/2026-07-17_22-25-38.md
git diff --cached --check
git commit -m "fix: close Custom AI streaming responses"
```

Expected: one focused local commit; no push or release.

- [ ] **Step 5: Re-submit changed transport code to Grok 4.5**

Ask Grok to verify the resource-lifetime invariant, identify regressions or double-close risks with proof, and rank the next candidate. Save the read-only result as `docs/cleanup/grok-4.5-project-audit-round-2-2026-07-17.md` before deciding whether to begin another code increment.
