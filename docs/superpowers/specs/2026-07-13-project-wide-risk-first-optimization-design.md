# Project-Wide Risk-First Optimization Design

## Objective

Harden the tracked OCR Translator project against the highest-value security,
runtime-consistency, latency, verification, and packaging risks found in the
2026-07-13 read-only audit, then publish the verified result to the current
GitHub branch.

The untracked `Dango-Translator-Ver6.3.1/` directory is outside this project.
It must not be read as product code, changed, staged, committed, or pushed.

## Current Evidence

- The current branch is `codex/publish-project-updates-20260706` and is aligned
  with its remote tracking branch at `046c462` before this design change.
- `py -m unittest discover -s tests` passes 504 tests.
- `py -m unittest test_custom_ai test_custom_ai_startup -q` passes 35 tests.
- Default `py -m unittest -q` runs only the 35 root tests and silently misses
  the 504-test main suite.
- All 76 tracked Python source/test files parsed successfully.
- The latest runtime log shows healthy sub-second `xai` translations, so the
  existing translation snapshot, adaptive route timeout, queue coalescing,
  cache, and metrics work should be preserved rather than redesigned.
- The same log contains 1,626 no-op clear-timeout reset messages and 725
  repeated cooling-profile skip messages in the sampled window, showing a
  remaining diagnostic-I/O hotspot.

## Scope and Ordering

Work is divided into four independently testable increments. Each increment
gets its own implementation plan, backup set, TDD cycle, focused verification,
and commit. A full-project verification and secret scan follow the fourth
increment before the branch is pushed.

1. Secret-safe diagnostics and credential persistence.
2. Immutable API OCR request identity and session isolation.
3. True end-to-end Custom AI request deadlines.
4. Test, configuration, packaging, documentation, and log-noise hardening.

Security goes first because the current behavior can persist user content or
credentials in plaintext. OCR consistency goes before latency work because a
fast request is still incorrect if it belongs to a previous profile or
session. Engineering hygiene goes last so its tests and documentation describe
the final runtime behavior.

## Increment 1: Secret-Safe Diagnostics and Credentials

### Diagnostic content policy

Custom AI short logs will be metadata-only by default. Their normal blocks may
contain provider name, model name, duration, token counts, cost, success/error
classification, and content shape (`chars` and `lines`), but not OCR text,
translated text, prompts, response bodies, credentials, or endpoint query
parameters.

A new explicit `custom_ai_log_content_enabled` setting will default to `False`.
When enabled, it may add the existing result body to the short log. The setting
will be placed in the existing diagnostics/debugging area without changing the
overall UI layout. The label must state that recognized/translated content is
written to disk. Disabling the setting affects future entries; the existing
clear-log action will be extended to clear the debug log and both Custom AI
short logs and their rotated backups.

`debug_logging_enabled=False` will stop both the debug log and Custom AI short
logs. Runtime latency observations used for adaptive behavior must still be
updated in memory even when disk logging is disabled.

### Error sanitation

A small shared sanitizer module will own these rules:

- URLs retain only normalized scheme, host, optional non-default port, and
  path. User info, query, and fragment are removed.
- Known credential-bearing fields such as `Authorization`, `Cookie`,
  `api_key`, `token`, `secret`, and `password` are redacted case-insensitively.
- The active API key and any known credential values are replaced even when
  embedded in a larger message.
- Upstream errors expose the HTTP status, safe provider error code/type, and a
  bounded sanitized message. Raw arbitrary response objects/bodies are not
  copied into UI or ordinary logs.
- TLS/reset guidance may remain descriptive but cannot append an unsanitized
  original exception.

Transport exceptions, failover summaries, UI-visible errors, and logs must all
pass through the same sanitizer so one safe representation is reused end to
end.

### Credential failure policy

Credential Manager failure will be fail-closed for disk persistence:

- A newly entered API key may remain in the in-memory profile for the current
  process, but it is never serialized into a JSON or INI file as fallback.
- Profile metadata may be saved without the key only when doing so cannot
  destroy a previously stored credential reference.
- The save operation returns an explicit failure result that the existing UI
  surfaces as a concise error; it must not report success.
- An edit uses the current transactional/rollback behavior so a failed new
  credential write does not delete a previously valid credential.
- Legacy plaintext keys may be read only for migration. After successful
  migration they are removed atomically; failed migration leaves the source
  file unchanged and reports the problem.

Profile/config paths remain unchanged in this optimization cycle to avoid an
unrelated user-data migration. Repository ignore rules will add defensive
patterns for supported local profile/config variants and their temporary
files, while explicitly retaining the publishable example configuration.
Ignore rules are only a secondary guard; fail-closed credential persistence is
the primary secret-control mechanism.

### Increment 1 tests

- A unique secret marker returned by a fake provider never appears in default
  debug or short logs.
- Content logging occurs only after the explicit setting is enabled.
- Disabling debug logging prevents all runtime disk logs while in-memory
  metrics still update.
- URL user info, query tokens, headers, cookie values, response-body markers,
  and API keys are absent from raised errors, UI strings, and log blocks.
- A failing fake credential store produces no plaintext key on disk and an
  explicit failed-save result.
- A failed credential update preserves the prior credential reference and
  secret.
- Clearing logs removes active and rotated debug/OCR/translation log files.

## Increment 2: Immutable API OCR Identity and Session Isolation

### OCR request snapshot

Every API OCR submission will create one immutable snapshot before image
conversion/cache lookup. It freezes:

- OCR session generation and batch sequence.
- A copy of the selected OCR profile, including its in-memory credential.
- Provider/model/wire API identity.
- Source language.
- Line-break, reasoning, image-detail, image-format, and MIME contract.
- Frame-cache identity derived from exactly those frozen fields.

`TranslationHandler.perform_ocr` will consume the snapshot instead of reading
the active profile or Tk variables in the worker thread. The profile object is
never logged and later UI edits cannot mutate it.

### Session generation

The app owns a monotonically increasing `ocr_session_generation`. Starting,
stopping, resetting, or closing an OCR session advances it. An API OCR call is
identified by `(generation, sequence)` rather than sequence alone.

Generation checks occur:

1. Before an asynchronous provider call begins.
2. After the provider returns and before scheduling the UI callback.
3. In the UI callback before cache writes, display updates, or translation
   submission.

Stale callbacks perform no cache, display, context, or scheduler mutation.
Their final cleanup removes only their own generation-qualified active-call
token, so an old `sequence=1` cannot remove a new session's `sequence=1`.

Cache hits follow the same snapshot and generation checks. Local PaddleOCR
behavior remains unchanged except where it shares session-reset bookkeeping.

### Increment 2 tests

- Block profile A after submission, switch the UI to profile B, then release
  the worker. Only A is called and only A's cache key is populated.
- Change line-break/image/reasoning settings while a call is blocked; the
  provider and cache retain the submitted contract.
- Complete an old API OCR call after stop/start. It does not display, cache,
  start translation, or alter the new active-call set.
- A current-generation cache hit still displays normally.
- Error and `<EMPTY>` results from stale generations are also discarded.

## Increment 3: End-to-End Custom AI Deadlines

### Deadline model

`timeout_seconds` will become a total request budget, not a fresh timeout for
every attempt. A monotonic absolute deadline is created from the submission
time and frozen timeout decision. Queue delay, endpoint fallback, transient
retry, structured/plain fallback, Responses empty-output recovery, profile
failover, stream fallback, and race candidates all consume the same budget.

Before each network attempt, the transport computes the remaining budget. If
no useful budget remains, it raises one sanitized deadline-expired error and
does not start another URL, retry, or profile. Each HTTP call receives at most
the remaining duration. Streaming code also checks the absolute deadline
between chunks so repeated reads cannot extend total wall time indefinitely.

Parallel race candidates share the same deadline. Finishing/cancelling race
losers preserves the existing suppression semantics. Adaptive route decisions
still choose the initial budget; this increment changes budget consumption,
not the existing P90 policy or its exclusions.

The implementation will not add arbitrary sleeps. Retry eligibility remains
based on the existing status/error policy, but eligibility cannot override the
remaining deadline.

### Increment 3 tests

- A fake clock/client advances time for each endpoint and retry; total work
  stops at the configured deadline plus a small deterministic epsilon.
- Budget exhaustion prevents the next URL, profile, structured fallback, or
  empty-output retry from being sent.
- A healthy first attempt receives the expected budget and behaves unchanged.
- Stream chunks and race candidates cannot outlive the shared deadline.
- Existing cooldown, rate-limit, route-adaptive, and failover tests remain
  green.

## Increment 4: Engineering and Release Hygiene

### Test discovery and isolation

The repository will have one documented test command that runs root and
`tests/` suites together. Test package discovery will be fixed so the command
cannot silently report only 35 tests. Subprocess-based import tests will set a
temporary `OCR_TRANSLATOR_LOG_DIR`; test runs must not append to the user's
runtime log.

A Windows GitHub Actions workflow will run the unified offline suite on Python
3.9, 3.10, 3.11, and 3.12. It will not call real providers or require secrets.

### Atomic user-file writes

INI configuration and custom-prompt writes will use a shared same-directory
temporary-file helper with flush, `fsync`, and `os.replace`. A parse failure
will preserve the original file and create a timestamped corrupt-file copy
before defaults can be saved. Temporary files are cleaned safely without
masking the original error.

### Packaging and documentation

- `resources/__init__.py` will make the resource directory an installable data
  package, and `setup.py` will declare `resources/*.csv` as that package's
  data. The existing flat runtime modules remain unchanged.
- Package/resource tests will build a wheel in an isolated temporary output
  directory and inspect its contents.
- Installation documentation will stop claiming that `requirements.txt`
  installs PaddleOCR when it does not. Paddle installation will be documented
  as an explicit optional backend step based on the official supported
  procedure.
- Version references will match `3.10.2` unless a later release is deliberately
  created.
- The developer guide will describe the actual setuptools/PyInstaller paths
  and only commands/files that exist.
- `compile_app.py` will use argument lists and `sys.executable -m pip`, refuse
  to mutate an unmanaged global environment without explicit confirmation,
  and document an isolated build environment. It will not silently upgrade or
  replace the developer's PyTorch installation.

### Bounded log-noise cleanup

- `reset_clear_timeout` logs only a real active-to-inactive timer transition.
- Repeated cooling-profile skip messages use the existing content-free
  coalescing gate keyed by non-secret profile identity.
- No broader per-translation lifecycle logs are removed without new runtime
  evidence; diagnostic sequence visibility remains available.

### Increment 4 tests

- The unified test command executes at least the current 539 tests and leaves
  the real runtime log unchanged.
- Interrupted/failed atomic writes preserve the old config/prompt bytes.
- A corrupt config is preserved before default recovery.
- A built wheel contains runtime Python modules and required CSV resources.
- Build commands are passed as argument arrays through the active interpreter
  and are testable without invoking pip/PyInstaller.
- Repeated no-op timeout resets produce no log; real transitions still do.
- Cooling-profile repetitions are summarized by the existing coalescer.

## Error Handling and Rollback

- Each increment begins with timestamped backups of every existing file it may
  change, preserving paths relative to the workspace root.
- New files do not require pre-edit backups, but they are listed in the handoff.
- Security failures are explicit and fail closed; runtime stale/deadline events
  are content-free and non-fatal.
- Existing unrelated work and the excluded Dango directory are never staged.
- Every increment is committed separately, so a regression can be reverted
  without losing later unrelated work.

## Verification and Publication Gates

Before pushing, all of the following must be true:

1. Focused RED/GREEN tests exist for every changed behavior.
2. The unified offline suite passes and proves it includes both prior suites.
3. Python syntax/import checks pass without writing into runtime logs.
4. Wheel/resource build and archive inspection pass.
5. `git diff --check` and staged-diff review pass.
6. A staged-file secret scan finds no credentials, runtime logs, local config,
   caches, backups, or Dango paths.
7. The final handoff records backups, commands, results, decisions, and any
   remaining external-only validation.
8. Only intended files are committed.
9. The current branch is pushed to `origin` and `git ls-remote` confirms the
   remote branch points to the final local commit.

No paid API request or GitHub Release is part of this design. Real-provider
validation requires separate explicit authorization. The requested publication
deliverable is a verified branch push.

## Explicit Non-Goals

- No desktop UI redesign or modernization.
- No replacement of the working translation request snapshot, adaptive P90
  policy, unified cache, runtime metrics, or queue coalescing architecture.
- No broad exception-style cleanup, mass reformatting, or large facade/module
  rewrite.
- No blanket dependency upgrade, especially not PySide6 beyond its pinned
  compatibility boundary.
- No work on the excluded Dango directory or generated/dependency folders.
