# Custom AI End-to-End Deadlines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each Custom AI translation request consume one frozen monotonic deadline across local queueing, HTTP attempts, retries, endpoint/profile fallback, streaming, and parallel race candidates.

**Architecture:** Add a small dependency-free deadline value object that owns an absolute `deadline_monotonic`, reports the remaining HTTP budget, and raises one content-free expiration exception. The existing translation request snapshot records this deadline at its initial arrival and survives pending-queue coalescing; all later provider operations receive the same absolute value. Transport code recomputes the remaining budget immediately before every HTTP request and stream chunk, while handler-level failover/race code prevents a new provider call or an unbounded coordinator wait after expiry.

**Tech Stack:** Python 3.9-3.12, `time.monotonic`, `concurrent.futures`, Requests-compatible fake clients, `unittest` and `unittest.mock`. No real-provider requests.

---

## File structure

- `custom_ai_deadline.py` (new): frozen deadline validation, remaining-budget calculation, and safe `CustomAIRequestDeadlineExceeded` error.
- `custom_ai_transport.py`: applies one deadline to compatibility retries, transient retries, endpoint candidates, stream parsing, and the Responses API equivalents.
- `handlers/translation_requests.py`: freezes the deadline in the existing request snapshot and propagates it through failover and race orchestration.
- `worker_translation.py`: preserves the original request snapshot while it waits in the newest-only pending queue.
- `worker_threads.py`: builds a snapshot from the original request arrival, reuses a queued snapshot, and preserves compatibility with older test doubles.
- `tests/test_custom_ai.py`: fake-clock/provider tests for transport retries, fallback paths, stream chunks, failover, and race.
- `tests/test_latency_optimization.py`: scheduler tests proving queue delay consumes the same frozen budget.

## Task 0: Back up every existing target

**Files:**

- Back up: `custom_ai_transport.py`
- Back up: `handlers/translation_requests.py`
- Back up: `worker_translation.py`
- Back up: `worker_threads.py`
- Back up: `tests/test_custom_ai.py`
- Back up: `tests/test_latency_optimization.py`

- [ ] **Step 1: Create and verify backups.**

  Create `.codex/backups/YYYY-MM-DD_HH-mm-ss/` with the six workspace-relative paths above. Compare every source/backup SHA-256 pair and stop unless all pairs match. Confirm `git status --short` is clean. `custom_ai_deadline.py` is new and requires no pre-edit backup.

## Task 1: Define the immutable deadline contract and enforce it in direct HTTP attempts

**Files:**

- Create: `custom_ai_deadline.py`
- Modify: `custom_ai_transport.py`
- Test: `tests/test_custom_ai.py`

- [ ] **Step 1: Write failing unit tests for the deadline value object.**

  Add tests that patch `custom_ai_deadline.time.monotonic` and prove a deadline created at `100.0` with a `4.0` second budget reports `4.0`, then `1.5`, then raises the exact content-free exception at `104.0`. Include invalid/non-finite/non-positive input fallback coverage so caller-supplied metadata cannot create an infinite deadline.

  ```python
  deadline = CustomAIRequestDeadline.from_timeout(4.0, now=100.0)
  self.assertEqual(deadline.remaining_seconds(now=102.5), 1.5)
  with self.assertRaisesRegex(
      CustomAIRequestDeadlineExceeded,
      "Custom AI request deadline expired",
  ):
      deadline.http_timeout(now=104.0)
  ```

- [ ] **Step 2: Run the new deadline tests and verify RED.**

  Run:

  ```powershell
  py -m unittest tests.test_custom_ai.CustomAIDeadlineTests -v
  ```

  Expected: import/attribute failure because neither the deadline module nor the transport propagation exists.

- [ ] **Step 3: Implement `custom_ai_deadline.py`.**

  Add this complete dependency-free value object. Transport owns the decision of which configured timeout is the fallback; the class only validates the supplied value.

  ```python
  from dataclasses import dataclass
  import math
  import time


  class CustomAIRequestDeadlineExceeded(TimeoutError):
      """Safe terminal error: no Custom AI request budget remains."""


  def _finite(value):
      try:
          normalized = float(value)
      except (TypeError, ValueError):
          return None
      return normalized if math.isfinite(normalized) else None


  def _positive_finite(value):
      normalized = _finite(value)
      return normalized if normalized is not None and normalized > 0.0 else None


  @dataclass(frozen=True)
  class CustomAIRequestDeadline:
      deadline_monotonic: float

      @classmethod
      def from_timeout(cls, timeout_seconds, now=None, fallback_seconds=30.0):
          started_at = _finite(now)
          if started_at is None:
              started_at = time.monotonic()
          budget = _positive_finite(timeout_seconds)
          if budget is None:
              budget = _positive_finite(fallback_seconds)
          if budget is None:
              raise ValueError("Custom AI request timeout must be positive")
          return cls(started_at + budget)

      @classmethod
      def from_absolute(cls, deadline_monotonic):
          absolute = _finite(deadline_monotonic)
          if absolute is None:
              raise ValueError("Custom AI request deadline must be finite")
          return cls(absolute)

      def remaining_seconds(self, now=None):
          current = _finite(now)
          if current is None:
              current = time.monotonic()
          return max(0.0, self.deadline_monotonic - current)

      def http_timeout(self, now=None):
          remaining = self.remaining_seconds(now)
          if remaining <= 0.0:
              raise CustomAIRequestDeadlineExceeded(
                  "Custom AI request deadline expired"
              )
          return remaining
  ```

  `from_timeout` uses `time.monotonic()` when `now` is omitted. `http_timeout` returns only the positive remaining duration and otherwise raises `CustomAIRequestDeadlineExceeded("Custom AI request deadline expired")`; it must never include URL, profile, API key, response body, or exception text.

- [ ] **Step 4: Thread one resolved deadline through the provider's direct transport path.**

  Extend `CustomAIProvider.translate`, `recognize`, `_post`, `_responses_post`, `_stream_post`, `_stream_responses_post`, `_post_with_transient_recovery`, and `_post_with_output_limit_fallback` with a private `deadline` parameter. At the public boundary, resolve either an already-provided `deadline_monotonic` or one deadline from `timeout_seconds`; do not rebuild it in child calls.

  The nested `send` function in `_post_with_output_limit_fallback` must call `deadline.http_timeout()` immediately before every `http_client.post`:

  ```python
  def send(current_payload):
      kwargs = {
          "headers": headers,
          "json": current_payload,
          "timeout": deadline.http_timeout(),
      }
      if stream:
          kwargs["stream"] = True
      return http_client.post(url, **kwargs)
  ```

  Preserve legacy direct callers by accepting absent `deadline_monotonic`; preserve the configured `self.timeout` fallback only when neither a valid timeout nor an absolute deadline is supplied.

- [ ] **Step 5: Write direct retry/endpoint RED tests.**

  Use a fake post client that advances the patched monotonic clock. Prove the first endpoint receives the full budget, a failed endpoint consumes part of it, and the next endpoint receives only the remaining budget. Add a transient `502` and a compatibility-retry case; after expiry, assert the client call count does not increase and the raised message is exactly the safe deadline error.

  ```python
  self.assertEqual(client.timeouts, [4.0, 1.5])
  self.assertEqual(client.urls, ["https://relay/v1/chat/completions"])
  with self.assertRaises(CustomAIRequestDeadlineExceeded):
      provider.translate(profile, "text", "en", "zh", timeout_seconds=4.0)
  self.assertEqual(client.call_count, 1)
  ```

- [ ] **Step 6: Run focused GREEN verification and commit.**

  Run:

  ```powershell
  py -m unittest tests.test_custom_ai.CustomAIDeadlineTests -v
  py -m unittest tests.test_custom_ai -q
  ```

  Expected: deadline tests pass, all existing Custom AI tests pass, and every fake client's observed timeout is no greater than the frozen remaining budget.

  Commit only `custom_ai_deadline.py`, `custom_ai_transport.py`, and the focused test changes:

  ```powershell
  git add custom_ai_deadline.py custom_ai_transport.py tests/test_custom_ai.py
  git commit -m "feat: bound Custom AI HTTP attempts by one deadline"
  ```

## Task 2: Apply the deadline to all transport fallback and streaming paths

**Files:**

- Modify: `custom_ai_transport.py`
- Test: `tests/test_custom_ai.py`

- [ ] **Step 1: Write failing stream and structured-fallback tests.**

  Add fake streaming chat and Responses responses whose `iter_lines()` advances a patched monotonic clock. Assert that a chunk arriving after expiry raises the safe deadline exception, no later chunk invokes `stream_callback`, and the stream response is closed if it exposes `close`. Add tests that structured-output fallback, stream-to-non-stream fallback, output-limit fallback, and Responses empty-output recovery use the original absolute deadline rather than resetting a `timeout_seconds` value.

  ```python
  response = FakeStream(["data: first", "data: second"], clock)
  with self.assertRaises(CustomAIRequestDeadlineExceeded):
      provider.translate(profile, "text", "en", "zh", latency_mode="stream",
                         timeout_seconds=1.0, stream_callback=partials.append)
  self.assertEqual(partials, ["first"])
  self.assertTrue(response.closed)
  ```

- [ ] **Step 2: Run the focused stream tests and verify RED.**

  Run:

  ```powershell
  py -m unittest tests.test_custom_ai.CustomAITransportDeadlineFallbackTests -v
  ```

  Expected: stream/fallback code either calls a second request with a new full timeout or delivers a late chunk.

- [ ] **Step 3: Implement deadline-aware fallback and parser behavior.**

  Pass the same deadline into every `active_request` closure in `translate`, including stream-to-non-stream, output-limit, structured/plain, and Responses empty-output recovery. In each outer endpoint loop, immediately re-raise `CustomAIRequestDeadlineExceeded` rather than adding it to an endpoint error list. Update both `_parse_streaming_chat_response` and `_parse_streaming_responses_response` to call `deadline.http_timeout()` after each received raw line and before invoking `stream_callback`; close the response in a `finally` block without masking a parse or deadline exception.

  Do not add sleeps, retries, new cooldown rules, or body logging. The existing transient/capability eligibility rules remain intact; they merely lose authority to start another attempt once `deadline.http_timeout()` fails.

- [ ] **Step 4: Run focused GREEN verification and commit.**

  Run:

  ```powershell
  py -m unittest tests.test_custom_ai.CustomAITransportDeadlineFallbackTests -v
  py -m unittest tests.test_custom_ai -q
  ```

  Expected: the new stream/recovery tests and the existing stream, Responses, structured-output, cooldown, and retry tests pass.

  Commit:

  ```powershell
  git add custom_ai_transport.py tests/test_custom_ai.py
  git commit -m "fix: share Custom AI deadlines across fallback paths"
  ```

## Task 3: Freeze the deadline before queueing and propagate it through failover and race

**Files:**

- Modify: `handlers/translation_requests.py`
- Modify: `worker_translation.py`
- Modify: `worker_threads.py`
- Test: `tests/test_custom_ai.py`
- Test: `tests/test_latency_optimization.py`

- [ ] **Step 1: Write failing queue-snapshot tests.**

  Extend `get_custom_ai_translation_request_snapshot` expectations with `deadline_monotonic`. Given `requested_at_monotonic=100.0` and a resolved safe timeout of `4.0`, the snapshot must contain `104.0`. Queue that snapshot, advance the clock to `103.0`, flush it, and assert the worker receives the original snapshot and absolute deadline rather than a newly resolved `107.0` deadline. Advance to `104.0` before provider entry and assert no HTTP client call occurs.

  ```python
  snapshot = handler.get_custom_ai_translation_request_snapshot(
      "Hello", requested_at_monotonic=100.0,
  )
  self.assertEqual(snapshot["deadline_monotonic"], 104.0)
  worker_threads._queue_pending_translation_request(
      app, "Hello", 1, 1.0, "submit interval", request_snapshot=snapshot,
  )
  ```

- [ ] **Step 2: Run queue tests and verify RED.**

  Run:

  ```powershell
  py -m unittest tests.test_latency_optimization.TranslationDeadlineQueueTests -v
  ```

  Expected: pending entries omit the snapshot and flushing rebuilds a deadline from the later clock value.

- [ ] **Step 3: Preserve the frozen snapshot through scheduler boundaries.**

  Change `get_custom_ai_translation_request_snapshot(text_content, commit=False, requested_at_monotonic=None)` to validate an internal monotonic arrival time and add `deadline_monotonic = arrival + timeout_seconds` to the returned dict. In `start_async_translation`, build this snapshot once after a cache miss and before queue/capacity decisions; when a queued snapshot is supplied, reuse it. Keep a `TypeError` compatibility fallback for older fake/extension snapshot getters that only accept `(text, commit=False)`.

  Add an optional `request_snapshot` argument to `_queue_pending_translation_request`, `_flush_pending_translation_request`, and `_expedite_pending_translation_request`; retain it in `pending_translation_request` and pass it to the later `_start_async_translation` call. Coalescing the same pending source text retains its original snapshot/deadline. Explicit profile refresh continues to invalidate the pending entry and builds a new request snapshot, preserving its existing intended behavior.

- [ ] **Step 4: Write failing failover/race tests.**

  Add fake provider tests where profile A consumes the remaining deadline and fails, then prove profile B is never called. For race mode, make all candidates block beyond the absolute deadline; assert the coordinator stops waiting using the remaining budget, cancels unstarted futures, releases race identities, and returns the one safe deadline error rather than an aggregated profile/body string. A healthy first candidate must still win with unchanged cache/latency behavior.

  ```python
  with self.assertRaisesRegex(ValueError, "Custom AI request deadline expired"):
      handler._custom_ai_translate_with_failover(
          primary_profile, "Hello", "en", "zh-CN", [], False,
          deadline_monotonic=101.0,
      )
  self.assertEqual(provider.called_profiles, ["A"])
  ```

- [ ] **Step 5: Propagate the absolute deadline through handler orchestration.**

  Resolve `deadline_monotonic` from `request_snapshot` in `_custom_ai_translate`; direct callers create one at that handler boundary from their `timeout_seconds`. Pass that same value to every `custom_ai_provider.translate` invocation in `_custom_ai_translate_with_failover` and `_custom_ai_translate_race`. Before each uncached failover candidate, require positive remaining budget and re-raise deadline expiry instead of appending a candidate error.

  In `_custom_ai_translate_race`, create one `CustomAIRequestDeadline` from the passed absolute value, submit every immediate candidate with that same value, and call `concurrent.futures.as_completed(future_to_profile, timeout=deadline.http_timeout())`. On `TimeoutError`, cancel unstarted futures, preserve existing non-blocking executor shutdown, and raise `CustomAIRequestDeadlineExceeded` after normal race-identity cleanup. Never permit a late result to win after the coordinator deadline.

- [ ] **Step 6: Run focused GREEN verification and commit.**

  Run:

  ```powershell
  py -m unittest tests.test_latency_optimization.TranslationDeadlineQueueTests -v
  py -m unittest tests.test_custom_ai.CustomAIDeadlineTests tests.test_custom_ai.CustomAITransportDeadlineFallbackTests -v
  py -m unittest tests.test_latency_optimization -q
  py -m unittest tests.test_custom_ai -q
  ```

  Expected: queued work retains one deadline, expiry prevents all later provider calls, race cleanup is intact, and existing adaptive timeout assertions still pass.

  Commit:

  ```powershell
  git add handlers/translation_requests.py worker_translation.py worker_threads.py tests/test_custom_ai.py tests/test_latency_optimization.py
  git commit -m "fix: propagate Custom AI deadlines through queue and failover"
  ```

## Task 4: Increment verification and handoff

**Files:**

- Create: `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md`

- [ ] **Step 1: Run complete offline verification.**

  ```powershell
  py -m unittest tests.test_latency_optimization -q
  py -m unittest tests.test_custom_ai -q
  py -m unittest discover -s tests -q
  py -m unittest test_custom_ai test_custom_ai_startup -q
  $files = @(Get-ChildItem -LiteralPath . -File -Filter '*.py') + @(Get-ChildItem -LiteralPath handlers -File -Filter '*.py') + @(Get-ChildItem -LiteralPath tests -File -Filter '*.py')
  py -m py_compile $files.FullName
  git diff --check 046c462..HEAD
  ```

  Confirm the worktree is clean after the implementation commits. Do not claim success from prior run output; record new counts and exit codes.

- [ ] **Step 2: Audit requirements against the approved design.**

  Check each deadline path explicitly: initial route decision, pending queue, endpoint candidates, transient retry, capability retry, output-limit retry, structured/plain retry, empty-output retry, stream-to-non-stream fallback, stream chunks, profile failover, and race coordinator/candidates. Confirm each reaches the same absolute deadline and that no deadline error can contain secret or provider body data.

- [ ] **Step 3: Create the handoff and commit it.**

  Write a timestamped handoff containing changed/created files, exact backup directory, RED/GREEN evidence, final commands and counts, design decisions, review conclusions, and any non-blocking external-only validation. Then commit it:

  ```powershell
  git add .codex/handoffs/YYYY-MM-DD_HH-mm-ss.md
  git commit -m "docs: hand off Custom AI deadline increment"
  ```

## Required review gates

For Tasks 1-3, use a fresh implementer and then perform two independent reviews before moving on:

1. Specification review against this plan and the approved Increment 3 design.
2. Code-quality review focusing on monotonic-clock correctness, exception hygiene, no post-expiry HTTP calls, race cleanup, stream response closing, fake-client compatibility, and no secret/body leakage.

Fix every Critical or Important finding and rerun both reviews after the fix.

## Non-goals

- Do not change the adaptive P90 route-selection policy or its stream/race exclusions.
- Do not add sleeps, network calls, provider retries, new UI controls, or new cooldown behavior.
- Do not alter local PaddleOCR, API OCR session isolation, cache schema, or the excluded `Dango-Translator-Ver6.3.1/` directory.
- Do not log profile dictionaries, endpoint queries, credentials, request/response bodies, or deadline exception internals.
