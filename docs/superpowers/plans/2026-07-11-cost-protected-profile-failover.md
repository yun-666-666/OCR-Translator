# Cost-Protected Custom AI Profile Failover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use enabled Custom AI profiles sequentially after a failed profile, and prevent additional API calls while every profile is cooling down.

**Architecture:** `CustomAIProvider` owns a short-lived, profile-specific circuit in the existing transport layer. `TranslationRequestsMixin` orders the active and enabled fallback profiles, checks each profile's cache, then performs at most one sequential call per profile. The worker uses that same eligible set when deciding whether to defer work.

**Tech Stack:** Python 3, `requests`-compatible HTTP clients, `unittest`, existing Custom AI cache and cooldown infrastructure.

---

### Task 1: Add provider-level forced-SSE recovery and profile circuit state

**Files:**
- Modify: `custom_ai.py:116-123`
- Modify: `custom_ai_requests.py:420-432`
- Modify: `custom_ai_transport.py:397-423`
- Test: `tests/test_custom_ai.py:CustomAIProviderTests`

- [ ] **Step 1: Write failing provider tests**

```python
def test_non_stream_chat_response_accepts_forced_sse_with_text(self):
    provider = CustomAIProvider(http_client=object())
    response = ResponseWithSse([
        'data: {"choices":[{"delta":{"content":"译文"}}]}',
        'data: [DONE]',
    ])
    self.assertEqual(
        provider._load_response_json(response)["choices"][0]["message"]["content"],
        "译文",
    )

def test_profile_unavailable_cooldown_blocks_only_failed_profile(self):
    provider = CustomAIProvider(http_client=object())
    with patch("custom_ai.time.monotonic", return_value=100.0):
        provider.mark_profile_unavailable(FAILED_PROFILE, "empty forced SSE")
    with patch("custom_ai.time.monotonic", return_value=101.0):
        self.assertGreater(provider.get_cooldown_remaining(FAILED_PROFILE), 0.0)
        self.assertEqual(provider.get_cooldown_remaining(HEALTHY_PROFILE), 0.0)
```

- [ ] **Step 2: Run the two tests and verify RED**

Run: `py -m unittest tests.test_custom_ai.CustomAIProviderTests.test_non_stream_chat_response_accepts_forced_sse_with_text tests.test_custom_ai.CustomAIProviderTests.test_profile_unavailable_cooldown_blocks_only_failed_profile -v`

Expected: failure because forced SSE is treated as non-JSON and `mark_profile_unavailable` does not exist.

- [ ] **Step 3: Implement the smallest provider changes**

```python
# custom_ai.py
self._profile_unavailable_cooldowns = {}

# custom_ai_transport.py
def mark_profile_unavailable(self, profile, detail, seconds=60.0):
    key = self._profile_unavailable_key(profile)
    self._profile_unavailable_cooldowns[key] = time.monotonic() + seconds

# custom_ai_requests.py
try:
    return response.json()
except ValueError:
    if self._response_looks_like_sse(response):
        return self._parse_streaming_chat_response(response)
    raise
```

Use a non-secret profile id plus endpoint/model fallback key, merge profile and rate-limit remaining time in `get_cooldown_remaining`, and remove only the successful profile's unavailable state after a valid response.  A forced stream without text must still raise its existing parse error.

- [ ] **Step 4: Run the two tests and verify GREEN**

Run: same command as Step 2.

Expected: both pass; the failed profile has about 59 seconds remaining while the healthy profile remains callable.

### Task 2: Add ordered handler failover with per-profile cache checks

**Files:**
- Modify: `handlers/translation_requests.py:301-359,710-902`
- Test: `tests/test_custom_ai.py:TranslationHandlerCustomAITests`

- [ ] **Step 1: Write failing handler tests**

```python
def test_custom_ai_translation_falls_through_once_to_next_enabled_profile(self):
    handler = handler_with_profiles([PRIMARY, FALLBACK])
    handler.custom_ai_provider.translate = Mock(side_effect=[
        ValueError("forced SSE was empty"),
        ("译文", {}, 0.01),
    ])
    self.assertEqual(handler._custom_ai_translate("source", 0.0), "译文")
    self.assertEqual(
        [call.args[0]["id"] for call in handler.custom_ai_provider.translate.call_args_list],
        ["primary", "fallback"],
    )

def test_custom_ai_translation_uses_fallback_cache_without_network_call(self):
    handler = handler_with_profiles([PRIMARY, FALLBACK])
    store_translation_in_profile_cache(handler, FALLBACK, "source", "缓存译文")
    self.assertEqual(handler._custom_ai_translate("source", 0.0), "缓存译文")
    handler.custom_ai_provider.translate.assert_not_called()
```

- [ ] **Step 2: Run the two tests and verify RED**

Run: `py -m unittest tests.test_custom_ai.TranslationHandlerCustomAITests.test_custom_ai_translation_falls_through_once_to_next_enabled_profile tests.test_custom_ai.TranslationHandlerCustomAITests.test_custom_ai_translation_uses_fallback_cache_without_network_call -v`

Expected: failure because the current handler returns after the primary error and checks only the selected profile cache.

- [ ] **Step 3: Implement sequential candidate selection**

```python
def _get_custom_ai_failover_profiles(self, active_profile):
    profiles = [active_profile]
    profiles.extend(self.app.custom_ai_profiles.list_profiles(enabled_only=True))
    return deduplicate_by_profile_id(profiles)
```

In `_custom_ai_translate`, leave explicit `race` behavior unchanged.  For other modes, loop over the candidate list, skip a profile whose `get_cooldown_remaining` is positive, build that profile's existing cache parameters, check its cache, and invoke `translate` once.  Call `mark_profile_unavailable` only after a candidate call raises, record the failure against that candidate, and return on the first valid cached or remote translation.  If none succeeds, return one redacted all-profiles-unavailable error rather than a per-frame retry.

- [ ] **Step 4: Run the two tests and verify GREEN**

Run: same command as Step 2.

Expected: the provider is called in `[primary, fallback]` order exactly once each; a fallback cache hit makes zero provider calls.

### Task 3: Defer the scheduler only when all profiles are cooling down

**Files:**
- Modify: `handlers/translation_requests.py:666-708`
- Test: `tests/test_custom_ai.py:TranslationHandlerCustomAITests`

- [ ] **Step 1: Write failing cooldown-selection tests**

```python
def test_translation_cooldown_is_zero_when_enabled_fallback_is_healthy(self):
    handler = handler_with_profiles([PRIMARY, FALLBACK])
    handler.custom_ai_provider.get_cooldown_remaining = Mock(
        side_effect=lambda profile: 60.0 if profile["id"] == "primary" else 0.0
    )
    self.assertEqual(handler.get_translation_provider_cooldown_seconds(), 0.0)

def test_translation_cooldown_uses_earliest_enabled_profile_expiry(self):
    handler = handler_with_profiles([PRIMARY, FALLBACK])
    handler.custom_ai_provider.get_cooldown_remaining = Mock(
        side_effect=lambda profile: {"primary": 60.0, "fallback": 15.0}[profile["id"]]
    )
    self.assertEqual(handler.get_translation_provider_cooldown_seconds(), 15.0)
```

- [ ] **Step 2: Run the two tests and verify RED**

Run: `py -m unittest tests.test_custom_ai.TranslationHandlerCustomAITests.test_translation_cooldown_is_zero_when_enabled_fallback_is_healthy tests.test_custom_ai.TranslationHandlerCustomAITests.test_translation_cooldown_uses_earliest_enabled_profile_expiry -v`

Expected: failure because the existing query reads only the selected profile cooldown.

- [ ] **Step 3: Implement candidate-aware cooldown selection**

```python
profiles = self._get_custom_ai_failover_profiles(active_profile)
remaining_values = [max(0.0, float(cooldown_getter(profile))) for profile in profiles]
remaining = min(remaining_values) if remaining_values else 0.0
```

Use the selected profile only for explicit `race` behavior; use the sequential candidate list for ordinary modes.  Preserve the existing runtime metric name and error handling.

- [ ] **Step 4: Run the two tests and verify GREEN**

Run: same command as Step 2.

Expected: a healthy fallback returns `0.0`; all cooling profiles return the nearest expiry.

### Task 4: Verify the complete regression surface

**Files:**
- Modify: `tests/test_custom_ai.py`

- [ ] **Step 1: Run focused Custom AI tests**

Run: `py -m unittest tests.test_custom_ai -q`

Expected: all Custom AI tests pass, including rate-limit and race-mode coverage.

- [ ] **Step 2: Run the project test suite and static checks**

Run: `py -m unittest discover -s tests`

Expected: all tests pass.

Run: `py -B -m py_compile custom_ai.py custom_ai_requests.py custom_ai_transport.py handlers\\translation_requests.py tests\\test_custom_ai.py`

Expected: no output and exit code 0.

Run: `git diff --check`

Expected: no output and exit code 0.

- [ ] **Step 3: Record a handoff**

Create `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md` with the diagnosis, changed files, backup path, RED/GREEN evidence, final test results, and the distinction between sequential failover and explicit race mode.
