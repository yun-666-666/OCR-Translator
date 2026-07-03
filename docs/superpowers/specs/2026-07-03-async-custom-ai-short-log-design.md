# Async Custom AI Short Log Design

## Goal

Remove Custom AI short-log disk I/O from OCR and translation critical paths while
preserving ordered, bounded logs and flushing accepted entries during shutdown.

## Root cause

`TranslationHandler._log_custom_short_call()` calls `append_rotating_text()`
synchronously before returning the OCR or translation result. The shared rotating
writer flushes every write. A slow disk, antivirus scan, or rotation therefore
adds directly to subtitle latency.

A blocking-write probe confirmed that `_log_custom_short_call()` cannot return
while the file write is blocked.

## Design

Each `TranslationHandler` owns:

- one lazily-started `ThreadPoolExecutor(max_workers=1)` for ordered writes
- one state lock protecting executor availability and session-header state
- one `_write_custom_short_log()` worker that catches background I/O exceptions

`_log_custom_short_call()` continues to construct the small log block
synchronously, then submits only the file write. One worker preserves call order
without changing the global logger.

## Shutdown

`close()` atomically detaches the executor under the state lock, then calls
`shutdown(wait=True)` outside the lock. This guarantees:

- every task accepted before close is flushed
- no task can be submitted after detachment
- no deadlock between submission and shutdown
- repeated `close()` calls remain harmless

Cache and provider shutdown keep their existing behavior.

## Alternatives rejected

- Making the global logger asynchronous would alter every diagnostic path and
  broaden failure semantics beyond translation performance.
- Disabling short logs would remove cached-token and provider-duration evidence.
- Buffered writes without an explicit shutdown barrier could lose the most recent
  translations during normal application exit.

## Verification

A deterministic blocked-writer test proves the logging call returns while I/O is
blocked and `close()` waits until release. Existing short-log content tests are
updated to close the handler before inspecting the asynchronous write.
