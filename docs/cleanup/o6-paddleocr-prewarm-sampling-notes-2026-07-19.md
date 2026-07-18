# O6 PaddleOCR prewarm sampling notes (2026-07-19)

Observation-only metrics were added so cold/hot prewarm cost can be separated from Start and first OCR waits. No model, prewarm, daemon, or 20s wait-policy changes were made.

## Where to read metrics

1. Debug log lines:
   - `PaddleOCR prewarm started ... generation=N`
   - `PaddleOCR prewarm phase=text_recognition ... kind=...`
   - `PaddleOCR prewarm phase=full_engine ... kind=...`
   - `PaddleOCR prewarm completed ...`
   - `PaddleOCR worker prewarm wait completed=... ready=... duration=...`
2. Diagnostics panel / Copy summary:
   - timings: `paddleocr_prewarm_*_duration`, `paddleocr_first_ocr_wait_duration`
   - gauges: generation / active / ready / phase seconds / first OCR wait
   - labels: reason, trigger, settings summary, host, build kind, wait path
3. Programmatic: `app.get_paddleocr_prewarm_metrics_snapshot()`

## Safe fields only

Recorded settings summary includes version/size/device/lang/numeric limits plus only boolean source-directory configured/present flags. It never stores OCR text, API keys, full environment variables, or any user-path fragment.

Phase kinds that can be inferred without Paddle internals:

| `build_kind` | Meaning |
|---|---|
| `cache_hit` | Engine already in process cache |
| `model_download_and_build` | Model files absent before construct, present after |
| `model_build_cached_files` | Model files already present; construct only |
| `model_construct_files_absent` | Construct finished but files still absent/unknown |
| `model_construct` | Construct only; file presence unknown |

True in-constructor download vs antivirus file IO still cannot be split further than import / files-probe / construct.

## Cold-start sample procedure

1. Quit the app completely.
2. Optional deeper cold: clear process-local engine cache is automatic on restart; OS model cache remains unless you intentionally remove PaddleX official model cache outside the app.
3. Launch app with OCR model = PaddleOCR.
4. Do **not** press Start until Diagnostics shows prewarm `status=completed` or log shows prewarm completed.
5. Copy metrics summary once after prewarm completes.
6. Press Start and capture one subtitle frame.
7. Copy metrics summary again.

Expected cold signals:

- `paddleocr_prewarm_text_recognition_kind` / `full_engine_kind` not `cache_hit`
- non-trivial `paddleocr_prewarm_total_duration`
- `paddleocr_first_ocr_wait_duration` near 0 if prewarm finished before Start, or >0 if Start raced prewarm

## Hot-start sample procedure

1. Keep the same process running after a successful prewarm.
2. Stop capture if running, then Start again without changing PaddleOCR settings.
3. Copy metrics summary.

Expected hot signals:

- second engine acquisition would be `cache_hit` if prewarm/get runs again for same settings
- `already_ready` outcome if prewarm is re-requested for identical settings
- first OCR wait remains the one-shot value from the current generation

## Settings-change sample

1. Change model size or device.
2. Save settings.
3. Observe generation bump, new prewarm run, and new first-OCR wait generation after next Start.

## Current environment sampling result

Automated unit coverage uses fake engines and mocked monotonic clocks; it does **not** exercise the real multi-tens-of-seconds Paddle construct path.

Real cold/hot wall-clock sampling was **not** produced in this change session because:

- this task is metrics/tests only and must not alter the 20s wait or prewarm strategy
- a full GUI cold start with real PaddleOCR model construct is environment-dependent and can take ~30–60s
- no interactive Start/OCR desktop capture was driven after the instrumentation landed

To collect real samples on this machine after installing/using PaddleOCR models, follow the cold/hot procedures above and archive the Copy summary text (no OCR content is included).
