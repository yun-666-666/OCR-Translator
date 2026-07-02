# Runtime Logging and Cache Boundary Optimization Design

## Context

The current debug logger opens and closes `translator_debug.log` for every message. The application has hundreds of call sites, and the adaptive OCR load path can emit two messages every two seconds while nothing changes. The main debug log is already about 1.7 MB and has no size limit.

Custom AI short logs also open the same file twice for the first call in a session and grow without rotation. Unit tests exercise these paths and currently append synthetic provider results to the same files used for real runtime diagnosis.

The unified translation cache has two independent capacity defects:

- updating an existing key while the cache is full evicts an unrelated entry first;
- loading a persisted file larger than a reduced `max_size` evicts only one 10% batch and can remain far above the configured limit.

## Goals

- Keep runtime logs bounded without removing diagnostics.
- Reuse open log streams safely across threads instead of reopening the file for every message.
- Preserve UTF-8 BOM compatibility and the existing timestamped debug-log format.
- Keep unit-test logs out of real runtime logs automatically.
- Preserve the existing UI action that clears and immediately refreshes the debug log.
- Reduce steady-state adaptive OCR log volume while retaining transitions and periodic health evidence.
- Make cache updates and persisted-load capacity enforcement correct.

## Non-goals

- No UI redesign or new settings.
- No changes to OCR capture cadence, provider payloads, rate limits, translation scheduling, or displayed text.
- No deletion of current logs during migration.
- No external logging dependency, database, telemetry service, or network reporting.

## Logging Architecture

`logger.py` will own a process-local registry of `_RotatingTextWriter` instances keyed by absolute path and rotation policy. Each writer will:

- serialize access with an `RLock`;
- keep one line-buffered UTF-8-SIG stream open;
- flush every write so crash diagnostics remain current;
- rotate before a write would exceed `max_bytes`;
- retain a fixed number of numbered backups;
- support safe clear and close operations.

The main debug log will default to 5 MiB with three backups. Custom AI OCR and translation short logs will default to 2 MiB with two backups.

`append_rotating_text()` will support raw multi-line API log blocks. `log_debug()` will retain the existing `YYYY-MM-DD HH:MM:SS: message` format. `clear_debug_log()` will truncate through the same writer lock and write the existing clear marker so the UI does not race an active writer.

## Test Log Isolation

`resolve_runtime_log_path()` will recognize Python `unittest` and pytest entry points. Test processes will write to a process-specific directory under the system temporary directory. Normal source and packaged application processes will continue using the current working directory unless an explicit `OCR_TRANSLATOR_LOG_DIR` environment override is set.

Custom AI short logs will use the same resolver, so automated provider tests no longer pollute real API result logs.

## Adaptive OCR Log Throttling

`update_adaptive_scan_interval()` will keep its existing two-second load-check cadence and all interval decisions. Logging will change only:

- overload, normal, or moderate state transitions log immediately;
- an unchanged state logs one health message every 30 seconds;
- the unconditional per-check line and duplicate steady-state line are removed.

The helper will store only the last state and last log timestamp on the app instance.

## Cache Capacity Semantics

`UnifiedTranslationCache.store()` will evict only when inserting a new key at capacity. Updating an existing key changes its value and recency without removing another entry.

Persisted loading will calculate the exact excess count and remove enough least-recently-used entries to satisfy `len(cache) <= max_size` in one pass. Normal insertion can retain the existing 10% batch eviction policy to avoid sorting on every new entry near capacity.

## Error Handling

- Log write or rotation failures remain non-fatal and fall back to a concise console message.
- Rotation uses `os.replace()` and operates only on the configured log family.
- Test-path detection has an environment override and never changes application data paths.
- Cache load continues to skip malformed entries and fail open on unreadable JSON.

## Testing

TDD will cover:

- existing-key updates at full cache capacity without eviction;
- exact trimming of oversized persisted caches;
- writer stream reuse;
- bounded rotation and backup count;
- thread-safe concurrent log writes without lost lines;
- safe clear while the writer remains reusable;
- test-process path redirection;
- Custom AI short logs using the shared rotating writer;
- adaptive logging suppression for unchanged state;
- immediate logging on adaptive state transitions.

Final verification will include focused tests, full discovery suites, compile checks, a log-write performance comparison against the backed-up logger, a runtime-log hash check across unit tests, and visible application startup/interaction/shutdown through Computer Use.
