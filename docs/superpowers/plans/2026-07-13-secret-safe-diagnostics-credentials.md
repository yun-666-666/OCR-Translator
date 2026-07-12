# Secret-Safe Diagnostics and Credentials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent ordinary diagnostics, UI errors, profile persistence, and config persistence from writing user content or credentials to disk without explicit consent.

**Architecture:** Add one dependency-light sanitizer used by the Custom AI transport and handler boundaries. Keep short-log metrics in the existing asynchronous writer, but make result bodies opt-in and honor the global debug-log switch. Make credential writes transactional and fail closed before profile/config files are replaced.

**Tech Stack:** Python 3.9-3.12, unittest, tkinter, requests-compatible fake clients, Windows Credential Manager abstraction, existing rotating logger.

---

### Task 0: Back Up Existing Targets

**Files:**
- Back up: `custom_ai_transport.py`
- Back up: `handlers/translation_requests.py`
- Back up: `handlers/translation_results.py`
- Back up: `handlers/translation_handler.py`
- Back up: `custom_ai_profiles.py`
- Back up: `config_manager.py`
- Back up: `logger.py`
- Back up: `app_logic.py`
- Back up: `gui_diagnostics_builder.py`
- Back up: `handlers/ui_interaction_handler.py`
- Back up: `.gitignore`
- Back up: `tests/test_custom_ai.py`
- Back up: `tests/test_runtime_logging.py`

- [ ] **Step 1: Create one timestamped backup tree**

Run:

```powershell
$stamp = Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'
$backup = Join-Path '.codex/backups' $stamp
$files = @(
  'custom_ai_transport.py',
  'handlers/translation_requests.py',
  'handlers/translation_results.py',
  'handlers/translation_handler.py',
  'custom_ai_profiles.py',
  'config_manager.py',
  'logger.py',
  'app_logic.py',
  'gui_diagnostics_builder.py',
  'handlers/ui_interaction_handler.py',
  '.gitignore',
  'tests/test_custom_ai.py',
  'tests/test_runtime_logging.py'
)
foreach ($file in $files) {
  $destination = Join-Path $backup $file
  New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
  Copy-Item -LiteralPath $file -Destination $destination
}
$backup
```

Expected: one `.codex/backups/YYYY-MM-DD_HH-mm-ss/` path containing every listed file at its project-relative path.

- [ ] **Step 2: Verify every backup hash**

Run:

```powershell
$files | ForEach-Object {
  $sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_).Hash
  $backupHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $backup $_)).Hash
  [pscustomobject]@{ File = $_; Match = ($sourceHash -eq $backupHash) }
}
```

Expected: every `Match` value is `True`. Do not edit until this passes.

### Task 1: Add a Shared Content-Free Error Sanitizer

**Files:**
- Create: `diagnostic_sanitizer.py`
- Create: `tests/test_diagnostic_sanitizer.py`

- [ ] **Step 1: Write failing sanitizer tests**

Create tests covering URL sanitation, named-field sanitation, known secret replacement, and bounded output:

```python
import unittest

from diagnostic_sanitizer import sanitize_error_text, sanitize_url


class DiagnosticSanitizerTests(unittest.TestCase):
    def test_url_drops_userinfo_query_and_fragment(self):
        value = "https://user:pass@example.com:8443/v1/chat?token=query-secret#frag"
        self.assertEqual(sanitize_url(value), "https://example.com:8443/v1/chat")

    def test_error_redacts_credentials_and_response_content(self):
        marker = "subtitle-private-marker"
        value = (
            "Authorization: Bearer header-secret; Cookie=session=cookie-secret; "
            "api_key=field-secret; url=https://u:p@example.com/v1?token=query-secret; "
            f"response_body={marker}"
        )
        result = sanitize_error_text(
            value,
            known_secrets=["header-secret", "field-secret"],
            max_length=240,
        )
        for secret in [
            "header-secret",
            "cookie-secret",
            "field-secret",
            "query-secret",
            marker,
            "user:pass",
        ]:
            self.assertNotIn(secret, result)
        self.assertIn("[redacted]", result)

    def test_error_is_bounded_after_sanitation(self):
        self.assertLessEqual(len(sanitize_error_text("x" * 1000, max_length=80)), 80)
```

- [ ] **Step 2: Run the tests to verify RED**

Run: `py -m unittest tests.test_diagnostic_sanitizer -v`

Expected: import failure because `diagnostic_sanitizer.py` does not exist.

- [ ] **Step 3: Implement the sanitizer**

Create `diagnostic_sanitizer.py` with these public functions and no project imports:

```python
import re
from urllib.parse import urlsplit, urlunsplit


_URL_PATTERN = re.compile(r"https?://[^\s;,)]+", re.IGNORECASE)
_NAMED_SECRET_PATTERN = re.compile(
    r"(?i)\b(authorization|cookie|api[_-]?key|token|secret|password|response_body|content)"
    r"\s*[:=]\s*(?:bearer\s+)?[^\s;,]+"
)


def sanitize_url(value):
    raw = str(value or "").strip()
    try:
        parts = urlsplit(raw)
        host = str(parts.hostname or "").lower()
        if not parts.scheme or not host:
            return "[redacted-url]"
        if ":" in host:
            host = f"[{host}]"
        port = parts.port
        default_port = (
            (parts.scheme.lower() == "https" and port == 443)
            or (parts.scheme.lower() == "http" and port == 80)
        )
        netloc = host if port is None or default_port else f"{host}:{port}"
        return urlunsplit((parts.scheme.lower(), netloc, parts.path, "", ""))
    except (TypeError, ValueError):
        return "[redacted-url]"


def sanitize_error_text(message, known_secrets=(), max_length=500):
    text = str(message or "")
    for secret in known_secrets or ():
        secret = str(secret or "")
        if secret:
            text = text.replace(secret, "[redacted]")
    text = _URL_PATTERN.sub(lambda match: sanitize_url(match.group(0)), text)
    text = _NAMED_SECRET_PATTERN.sub(
        lambda match: f"{match.group(1)}=[redacted]",
        text,
    )
    text = " ".join(text.split())
    limit = max(0, int(max_length))
    if limit and len(text) > limit:
        suffix = "..."
        text = text[: max(0, limit - len(suffix))] + suffix
    return text
```

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run: `py -m unittest tests.test_diagnostic_sanitizer -v`

Expected: all sanitizer tests pass.

- [ ] **Step 5: Commit the isolated sanitizer**

Run:

```powershell
git add -- diagnostic_sanitizer.py tests/test_diagnostic_sanitizer.py
git commit -m "security: add content-free diagnostic sanitizer"
```

Expected: one commit containing only the sanitizer and its tests.

### Task 2: Sanitize Transport, Failover, Log, and UI Error Boundaries

**Files:**
- Modify: `custom_ai_transport.py`
- Modify: `handlers/translation_requests.py`
- Modify: `tests/test_custom_ai.py`
- Test: `tests/test_diagnostic_sanitizer.py`

- [ ] **Step 1: Add failing end-to-end redaction tests**

Add a new `CustomAITransportSanitizationTests` class to
`tests/test_diagnostic_sanitizer.py`. Its tests construct fake error responses
with URL user info/query values and secret-bearing JSON/body fields. Assert
that `_response_error_message`, `_non_json_response_message`,
`_sanitize_error`, `_sanitize_custom_ai_profile_error`, failover summaries,
and returned UI strings omit every marker.

Use this common assertion helper in `tests/test_diagnostic_sanitizer.py`:

```python
def assert_markers_absent(test_case, value, markers):
    rendered = str(value)
    for marker in markers:
        test_case.assertNotIn(marker, rendered)
```

The fake response must expose `status_code`, `headers`, `text`, `json()`, and `close()` so it exercises the real transport branches without network access.

- [ ] **Step 2: Run the focused tests to verify RED**

Run:

```powershell
py -m unittest tests.test_diagnostic_sanitizer.CustomAITransportSanitizationTests -v
```

Expected: current transport output still contains at least the endpoint query or injected response marker.

- [ ] **Step 3: Route transport errors through the sanitizer**

In `custom_ai_transport.py`:

```python
from diagnostic_sanitizer import sanitize_error_text, sanitize_url
```

Change `_response_error_message` and `_non_json_response_message` to render `sanitize_url(url)`, accept only safe error `type`/`code` plus a sanitized bounded message from JSON, and never interpolate raw response bodies. Change `_sanitize_error` to call:

```python
return sanitize_error_text(
    message,
    known_secrets=[api_key],
    max_length=500,
)
```

TLS/reset guidance may prepend its fixed explanation to this already-sanitized result; it must not append the original exception separately.

- [ ] **Step 4: Re-sanitize handler-level aggregation**

In `handlers/translation_requests.py`, import `sanitize_error_text`. Make `_sanitize_custom_ai_profile_error` sanitize the provider result again with the profile API key, and sanitize each failover failure before it is logged or returned. The final multi-profile message must contain provider display names plus safe error classes only.

- [ ] **Step 5: Run focused transport and failover suites**

Run:

```powershell
py -m unittest tests.test_diagnostic_sanitizer -v
py -m unittest tests.test_custom_ai.CostProtectedProfileFailoverHandlerTests -v
```

Expected: all tests pass and no fake marker is present in exception, log, or returned UI strings.

- [ ] **Step 6: Commit the boundary integration**

Run:

```powershell
git add -- custom_ai_transport.py handlers/translation_requests.py tests/test_custom_ai.py tests/test_diagnostic_sanitizer.py
git commit -m "security: sanitize custom AI error boundaries"
```

### Task 3: Make Custom AI Short Logs Metadata-Only by Default

**Files:**
- Modify: `config_manager.py`
- Modify: `app_logic.py`
- Modify: `gui_diagnostics_builder.py`
- Modify: `handlers/ui_interaction_handler.py`
- Modify: `handlers/translation_results.py`
- Modify: `handlers/translation_handler.py`
- Modify: `tests/test_runtime_logging.py`

- [ ] **Step 1: Write failing short-log policy tests**

Update the existing short-log test so the default block contains:

```text
Result: chars=10 lines=1
```

and does not contain the fake result body. Add cases where `custom_ai_log_content_enabled_var.get()` is `True` and where global debug logging is disabled. The enabled case must include the body; the disabled case must not call the writer but must still update prompt-cache and latency observations.

- [ ] **Step 2: Run the focused tests to verify RED**

Run: `py -m unittest tests.test_runtime_logging.RotatingTextWriterTests.test_custom_ai_short_log_uses_shared_rotating_writer -v`

Expected: the default current block contains the fake translated text.

- [ ] **Step 3: Add the config variable and existing-tab checkbox**

Add this default in `config_manager.DEFAULT_CONFIG_SETTINGS`:

```python
'custom_ai_log_content_enabled': 'False',
```

Initialize in `app_logic.py`:

```python
self.custom_ai_log_content_enabled_var = tk.BooleanVar(
    value=self.config.getboolean(
        'Settings',
        'custom_ai_log_content_enabled',
        fallback=False,
    )
)
```

Persist it beside `debug_logging_enabled` in `handlers/ui_interaction_handler.py`.

In `gui_diagnostics_builder.create_debug_tab`, add one `ttk.Checkbutton` to the existing `button_frame`:

```python
ttk.Checkbutton(
    button_frame,
    text=app.ui_lang.get_label(
        "custom_ai_log_content_enabled_label",
        "Write recognized/translated content to diagnostic logs",
    ),
    variable=app.custom_ai_log_content_enabled_var,
    command=app.save_settings,
).pack(side=tk.LEFT, padx=5)
```

- [ ] **Step 4: Separate metrics from disk persistence**

In `_log_custom_short_call`, keep `_record_custom_prompt_cache_usage` and `_record_custom_ai_latency_observation` before the disk-logging guard. Then return without building/scheduling a block when `logger.is_debug_logging_enabled()` is false.

Render the result line as:

```python
result_summary = summarize_text_for_log(result_text)
result_section = f"Result: {result_summary}\n"
if content_logging_enabled:
    result_section += f"--------------------\n{result_text}\n--------------------\n"
```

Import/re-export `is_debug_logging_enabled` through `handlers.translation_handler` so the mixin can be patched in tests without a circular import.

- [ ] **Step 5: Run focused and handler tests**

Run:

```powershell
py -m unittest tests.test_runtime_logging.RotatingTextWriterTests -q
py -m unittest tests.test_custom_ai.TranslationHandlerCustomAITests -q
```

Expected: all tests pass; ordinary short-log blocks are content-free.

- [ ] **Step 6: Commit the short-log policy**

Run:

```powershell
git add -- config_manager.py app_logic.py gui_diagnostics_builder.py handlers/ui_interaction_handler.py handlers/translation_results.py handlers/translation_handler.py tests/test_runtime_logging.py
git commit -m "security: make custom AI content logs opt-in"
```

### Task 4: Make Credential Persistence Fail Closed

**Files:**
- Modify: `custom_ai_profiles.py`
- Modify: `config_manager.py`
- Modify: `tests/test_custom_ai.py`

- [ ] **Step 1: Replace plaintext-fallback expectations with failing tests**

Replace
`CustomAIProfileManagerTests.test_unavailable_profile_credential_store_keeps_plaintext_and_logs_safely`
with a fail-closed migration test. Add adjacent
`test_new_profile_credential_failure_does_not_create_profile_file`,
`test_profile_update_credential_failure_preserves_previous_reference`, and
`test_provider_config_credential_failure_preserves_original_file` methods.
Use the existing `FakeCredentialStore(fail_writes=True)` and assert:

```python
self.assertNotIn("api_key", json.loads(profile_path.read_text(encoding="utf-8"))["profiles"][0])
self.assertNotIn(secret_marker, profile_path.read_text(encoding="utf-8"))
self.assertEqual(manager.get_profile(existing_id)["api_key_ref"], previous_ref)
```

For legacy plaintext migration, snapshot the file bytes before manager/config load and assert they remain byte-identical after the failed migration.

- [ ] **Step 2: Run the credential tests to verify RED**

Run:

```powershell
py -m unittest tests.test_custom_ai.CustomAIProfileManagerTests -v
```

Expected: current code serializes the plaintext fallback or reports success.

- [ ] **Step 3: Add an explicit persistence exception**

In `custom_ai_profiles.py`:

```python
class CredentialPersistenceError(RuntimeError):
    """Raised when a secret cannot be stored without plaintext fallback."""
```

Change `_store_profile_api_key` so credential-store failure logs only the action, reference, and exception class, removes plaintext-fallback state, and raises `CredentialPersistenceError("API key could not be stored securely")`.

Remove all serialization of `_api_key_plaintext_fallback`; `serialize_for_disk` must never emit an `api_key` field.

- [ ] **Step 4: Preserve migration source on failure**

During `_sanitize`, catch `CredentialPersistenceError` only around legacy migration, retain the key in the in-memory profile, mark the profile with a private `_credential_persistence_error=True`, and do not request an automatic save. `_save_staged_data` must reject snapshots containing that marker so later metadata changes cannot erase the only remaining legacy key.

In `config_manager.migrate_provider_api_keys_to_credentials`, collect write failures and raise `CredentialPersistenceError` after leaving the plaintext config object unchanged. `load_app_config` catches it, logs a content-free warning, and skips its automatic rewrite. `save_app_config` lets it propagate so the existing UI reports save failure.

- [ ] **Step 5: Preserve transactional update behavior**

For `add_profile`, store the credential before appending/saving staged data. For `update_profile`, write the versioned credential before saving metadata; if that write or metadata save fails, delete only the newly created reference and keep the previous published data/reference. Delete the prior reference only after the new data is published.

- [ ] **Step 6: Run credential and profile suites**

Run:

```powershell
py -m unittest tests.test_custom_ai.CustomAIProfileManagerTests -v
```

Expected: all fail-closed and rollback tests pass.

- [ ] **Step 7: Commit fail-closed persistence**

Run:

```powershell
git add -- custom_ai_profiles.py config_manager.py tests/test_custom_ai.py
git commit -m "security: fail closed on credential persistence errors"
```

### Task 5: Clear All Diagnostic Log Families and Add Ignore Guards

**Files:**
- Modify: `logger.py`
- Modify: `handlers/ui_interaction_handler.py`
- Modify: `.gitignore`
- Modify: `tests/test_runtime_logging.py`

- [ ] **Step 1: Write failing log-family clearing tests**

In a temporary `OCR_TRANSLATOR_LOG_DIR`, create active plus `.1`/`.2` files for:

```python
[
    "translator_debug.log",
    "CustomAI_OCR_Short_Log.txt",
    "CustomAI_Translation_Short_Log.txt",
]
```

Call the UI clear action. Assert the debug file contains only its clear marker, both short logs are empty or absent, and all rotated backups are absent.

- [ ] **Step 2: Run the focused tests to verify RED**

Run: `py -m unittest tests.test_runtime_logging.RotatingTextWriterTests.test_ui_clear_debug_log_uses_shared_writer -v`

Expected: only the debug log is cleared.

- [ ] **Step 3: Implement shared family clearing**

Add `clear_rotating_log_family(filename, max_bytes, backup_count, marker="")` in `logger.py`. It must acquire the registered writer lock, clear the active file through the writer, and delete `filename.1` through `filename.N` with `Path.unlink(missing_ok=True)` semantics compatible with Python 3.9.

Add `clear_runtime_diagnostic_logs()` that clears:

```python
clear_debug_log()
clear_rotating_log_family("CustomAI_OCR_Short_Log.txt", 2 * 1024 * 1024, 2)
clear_rotating_log_family("CustomAI_Translation_Short_Log.txt", 2 * 1024 * 1024, 2)
```

Update `handlers/ui_interaction_handler.py` to call this aggregate function.

- [ ] **Step 4: Add defensive root ignore patterns**

Replace the narrow root entries with:

```gitignore
/ocr_translator_config*.ini
!/ocr_translator_config.example.ini
/custom_ai_profiles*.json
/.custom_ai_profiles*.tmp
/custom_ai_translation_cache*.json
/custom_ai_translation_cache*.sqlite3
```

Do not add or ignore `Dango-Translator-Ver6.3.1/`; it remains an unrelated user directory outside commit scope.

- [ ] **Step 5: Run logger and ignore checks**

Run:

```powershell
py -m unittest tests.test_runtime_logging -q
git check-ignore ocr_translator_config.local.ini custom_ai_profiles.local.json .custom_ai_profiles.local.tmp
git check-ignore -q ocr_translator_config.example.ini; if ($LASTEXITCODE -eq 0) { throw 'example config must remain publishable' }
```

Expected: logger tests pass; local variants are ignored; example config is not ignored.

- [ ] **Step 6: Commit clearing and ignore guards**

Run:

```powershell
git add -- logger.py handlers/ui_interaction_handler.py .gitignore tests/test_runtime_logging.py
git commit -m "security: clear and ignore runtime diagnostic files"
```

### Task 6: Increment Verification and Handoff

**Files:**
- Create: `.codex/handoffs/YYYY-MM-DD_HH-mm-ss.md`

- [ ] **Step 1: Run focused security suites**

Run:

```powershell
py -m unittest tests.test_diagnostic_sanitizer tests.test_runtime_logging -q
py -m unittest tests.test_custom_ai -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run both existing project suites**

Run:

```powershell
py -m unittest discover -s tests
py -m unittest test_custom_ai test_custom_ai_startup -q
```

Expected: both suites pass; counts are at least the 504 and 35 baselines plus new tests.

- [ ] **Step 3: Run syntax and diff checks**

Run:

```powershell
@'
import pathlib
files = list(pathlib.Path('.').glob('*.py')) + list(pathlib.Path('handlers').glob('*.py')) + list(pathlib.Path('tests').glob('*.py'))
for path in sorted(set(files)):
    compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
print(f'compiled={len(set(files))}')
'@ | py -
git diff --check
```

Expected: compilation reports no errors and `git diff --check` is silent.

- [ ] **Step 4: Review scope and secret safety**

Run:

```powershell
git status --short
git diff --stat
git diff --name-only
```

Expected: no runtime log, config, cache, backup, credential file, generated directory, or `Dango-Translator-Ver6.3.1/` path is part of the diff.

- [ ] **Step 5: Create the increment handoff**

Create a timestamped handoff containing the task summary, exact changed files, backup path, RED/GREEN evidence, all verification results, security decisions, and any remaining work. Do not include API keys, response bodies, local config values, or runtime subtitle content.

- [ ] **Step 6: Commit the handoff if no code changes remain unstaged**

Run:

```powershell
git add -- .codex/handoffs/<timestamp>.md
git commit -m "docs: hand off security hardening increment"
```

Expected: the working tree contains only the excluded untracked Dango directory before moving to Increment 2.
