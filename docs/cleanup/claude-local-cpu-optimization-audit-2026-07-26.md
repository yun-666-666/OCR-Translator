# Claude Local-CPU / Redundant-Work Optimization Audit — 2026-07-26

| Item | Value |
|---|---|
| Date | 2026-07-26 |
| Project | OCR-Translator-3.9.6-experiment |
| Branch | `main` |
| HEAD | `cbf0a1ed` — `feat: integrate optimized OCR translator and bilingual docs` |
| Working tree | `app_logic.py` + `tests/test_latency_optimization.py` modified (user WIP: `on_ocr_model_change` keeps warmed prewarm) — **preserved, not reopened** |
| Scope | **Local** CPU / memory / redundant-work optimization on the live path (PaddleOCR + Custom AI). Complementary to the ~28 prior `docs/cleanup/` audits, which exhausted **paid remote-API request-waste**. |
| Method | 4 parallel subsystem review agents + independent line-level verification + in-repo micro-benchmarks |
| Reads only | Static source + existing tests + real debug-log line **counts** (no secrets/config body/cache body read) |
| Git | No stage / commit / push in this audit doc |

---

## 0. Verdict

The remote-API-$ hot path is genuinely exhausted (prior audits). This audit deliberately targeted the axis those audits **deprioritized**: local CPU, redundant per-frame/per-request work, and lock-hold time. It found **12 real, evidence-backed items**, two of which are **independently corroborated by two agents each**.

Nothing here is a correctness bug. These are efficiency wins on the continuously-running capture→OCR→translate→display loop. Two items also fix a **user-visible multi-second stall** (F4) and a **UI-thread lock-convoy tail risk** (F6).

Impact is honest: most items are **microseconds/frame** individually, but they run **2–10×/second forever**, several land **on the Tk UI thread**, and several inflate a **shared lock's** hold time. The two MED-impact standouts (F1 prewarm republish, F4 keystroke engine-destroy) are worth fixing on their own.

---

## 1. Method

Four independent reviewers were each scoped to one subsystem and required to back every finding with `file:line`, run-frequency, evidence, honest impact, and risk. They were explicitly told the remote-API-call-count path is exhausted and to hunt **local** inefficiency only. Findings were then re-verified against the live tree (worktree/backup copies excluded) and cross-checked for duplication.

**Calibration facts established independently:**
- No regexes are compiled inside functions anywhere in the source tree (all module-level) — already optimized.
- LRU eviction already uses `heapq.nsmallest` (worktree copies still show the old `sorted()` — confirming prior optimization).
- Logs/cache/sqlite are correctly git-ignored; repo hygiene is clean.
- `debug_logging_enabled` default is `False` in the shipped example but **`True`** in the developer's live `ocr_translator_config.ini` — so both "string built then discarded" (end users) and "regex-sanitize + flush per line" (developer) costs are real.

---

## 2. Findings summary (ranked by impact ÷ risk)

| # | Finding | Impact | Risk | Corroboration | Plan |
|---|---|---|---|---|---|
| **F1** | Prewarm metrics republished (~20 gauges/labels, ~50 regex, ~20 lock acquires) on **every** local OCR frame | **MED** | Low | **Agents 1 + 4** | **Fix now** |
| **F2** | Debug preview image (`convert→upscale→convert→cvtColor`) built every OCR frame, discarded unless OCR-debug on | **MED** | Low | Agent 1 | **Fix now** |
| **F3** | Streaming translation re-derives source-invariants + rebuilds preamble/label tuples on **every SSE chunk** (O(chunks×len)) | **MED** | Low | Agent 3 | **Fix now** |
| **F4** | Every keystroke in a PaddleOCR settings field **destroys warmed OCR engines + flushes frame cache** → multi-second stall | **MED** | Low-Med | Agent 4 | **Fix now** |
| **F5** | Cache `store`/`get`/`evict` run `log_debug` (regex+write+flush, ~45µs) **inside the global cache lock** | **MED** | Low | Agent 2 | **Fix now** |
| **F6** | ~13 unthrottled `log_debug` lines per translation, ~6 on the Tk UI thread (build+sanitize+flush per line) | MED | Low-Med | **Agents 2 + 4** | **Fix now (safe subset)** |
| **F7** | `str.maketrans` table rebuilt per call + same OCR string normalized 4–10× per candidate | Low | Low | Agent 1 | **Fix now** |
| **F8** | PaddleOCR backend re-`convert("RGB")`s already-RGB images (+ re-prepares whole frame on fallback) | Low-Med | Low | Agent 1 | **Fix now** |
| **F9** | `RuntimeMetrics.snapshot()` sorts each timing list **twice** (p50 then p90) for all metrics even when caller needs one | Low | Low | Agent 4 | **Fix now** |
| **F10** | Cache-key `json.dumps`+MD5 + profile/route SHA-256 recomputed ≥4–10× per request with identical inputs | Low | Low-Med | **Agents 2 + 3** | **Fix now (memoize)** |
| **F11** | Diagnostics metrics panel refreshes at 1 Hz for the app's whole life, even when Debug tab hidden | Low-Med | Low | Agent 4 | **Fix now** |
| **F12** | Fresh `mss` instance constructed + torn down for **every** screen grab (mss docs say reuse per thread) | MED | Low-Med | Agent 1 | **Document — defer** |

Plus one dead-code note (F13) and two verified-clean confirmations (§5).

---

## 3. Detailed findings

### F1 — Prewarm metrics fully republished on every local OCR frame  *(MED / Low — corroborated by Agents 1 & 4)*

- **Location**: trigger `app_logic.py:1245-1256` (`wait_for_paddleocr_prewarm` ready branch); `app_logic.py:746-772` (`_update_paddleocr_prewarm_metrics`); `app_logic.py:788-963` (`_publish_paddleocr_prewarm_metrics`); `app_logic.py:965-985` (`note_paddleocr_first_ocr_wait`); per-frame caller `worker_capture.py:583-592`; sanitize-under-lock `runtime_metrics.py:63-68`.
- **Frequency**: once per locally-OCR'd frame (every cache-miss frame; up to ~3–10/s continuously for the whole session).
- **Problem**: Prewarm is a one-time startup event, but in steady state (`_paddleocr_prewarmed_settings == settings`) the `ready_now` branch republishes byte-identical values every frame: 5 dict copies for the snapshot, ~10 `set_gauge` + ~9 `set_label`, each label running `sanitize_metric_label → sanitize_log_message` (5 compiled-regex subs + a `\s+` collapse) **inside the shared `RuntimeMetrics` RLock**. That is ~50–60 regex executions and ~20 lock acquisitions per frame for zero information change. `note_paddleocr_first_ocr_wait` additionally builds another 5-dict snapshot on its dedup path (`app_logic.py:973-974`) purely to return it to a caller (`worker_capture.py:588-591`) that discards it.
- **Evidence**: `process_local_ocr_frame` unconditionally calls `wait_for_prewarm` then `note_first_wait` per frame; the `ready_now` values (`wait_s=0.0, ready=True`, same generation) are constant after warmup; no "already published" guard exists (contrast the guarded `_apply_font_if_changed` / style caches elsewhere).
- **Impact**: MED — ~100–300 µs pure bookkeeping/frame **plus ~20 acquisitions of the metrics lock/frame**, the same lock the capture thread and the 1 Hz UI snapshot contend for.
- **Fix**: In `wait_for_paddleocr_prewarm`, track `_prewarm_ready_published_generation`; skip `_update_paddleocr_prewarm_metrics` on the `ready_now` path when that generation was already published. In `note_paddleocr_first_ocr_wait`, `return None` on the already-recorded early-return path instead of building a snapshot.
- **Test guard**: `tests/test_paddle_ocr_backend.py:1505-1518` asserts the *first* wait observation per generation — a publish-once-per-generation guard preserves it.

### F2 — Debug preview image built on every OCR frame, then discarded  *(MED / Low — Agent 1, verified)*

- **Location**: `worker_capture.py:598-599` (build), helper `_pil_to_debug_bgr` `worker_capture.py:554-556`; consumers `worker_threads.py:322-323`, `:379-380`, `handlers/ui_interaction_handler.py:404-406`.
- **Frequency**: per local OCR call (~2–10/s), including frames whose OCR text is empty.
- **Problem**: `process_local_ocr_frame` always runs `preview_pil = _prepare_paddleocr_image(...)` (full `convert("RGB")` copy + optional **bicubic upscale** up to 4×) then `_pil_to_debug_bgr` (another `convert("RGB")` + `np.array` copy + `cv2.cvtColor`). That's 3–4 full-frame buffer ops per frame producing an image consumed **only** when OCR-debugging is on.
- **Evidence (verified)**: all three consumers are gated on `frame_snapshot.ocr_debugging` (lines 322, 379) or `ocr_debugging_var` (`save_debug_images`), and null-check the image. The call site (`worker_threads.py:354-358`) passes `capture_snapshot=frame_snapshot`, whose `.ocr_debugging` is the same flag gating every consumer. `CaptureUISnapshot.ocr_debugging` exists (`worker_capture.py:44`, populated `:456-471`).
- **Impact**: MED (high with `upscale > 1`) — ~2–4 ms/frame at upscale 1.0, ~5–15 ms/frame at upscale 2.0, continuously, plus allocator churn.
- **Fix**: build `debug_img` only when `capture_snapshot.ocr_debugging` (fall back to `_read_app_var(app, "ocr_debugging_var")` when no snapshot); otherwise return `None`. Toggling debug on takes effect within one snapshot refresh.

### F3 — Streaming translation re-normalizes the whole buffer + rebuilds invariants per SSE chunk  *(MED / Low — Agent 3, verified)*

- **Location**: `custom_ai_requests.py:100-201` (`_normalize_translation_stream_partial`), closure `:203-220` (`_build_translation_stream_callback`); driven from `custom_ai_transport.py:1441-1444` (passes full `accumulated` per chunk).
- **Frequency**: once per SSE content delta (≈ per generated token) for every streamed translation; stream is the default adaptive promotion. A subtitle is 30–150 chunks (cap 2048).
- **Problem (verified)**: lines 109–126 re-derive `source`, `source_lines`, `source_is_fenced`, `source_casefold`, `source_first_line` — all functions of `source_text`, fixed for the whole stream — on every chunk, plus `candidate.casefold()` over the entire accumulated buffer, plus rebuilding `active_preambles` (6-item, `:162-166`) and `active_labels` (12-item, `:177-181`) from constants. Work per stream is O(chunks × accumulated_length).
- **Impact**: MED — the largest genuinely-wasted CPU in the transport layer; ~0.5–3 ms/response at subtitle sizes, growing quadratically toward tens of ms at the token cap, on the network thread per streamed request.
- **Fix**: compute the seven source-derived invariants **once** in the `_build_translation_stream_callback` closure and pass them into `_normalize_translation_stream_partial` (new optional params). Pure invariant hoist; emitted partials unchanged. `tests/test_custom_ai.py` covers partial sequences.

### F4 — Every keystroke in a PaddleOCR settings field destroys warmed engines + flushes the frame cache  *(MED / Low-Med — Agent 4)*

- **Location**: `app_logic.py:1480-1492` (`on_ocr_parameter_change`), traces `app_logic.py:519-532`, cache clear `:655-662`; unconditional `.set()` focus-out `gui_settings_builder.py:905-914`; downstream `paddle_ocr_backend.py:518-522`, `worker_capture.py:755-768`.
- **Frequency**: per **keystroke** while typing in any of the ~10 traced PaddleOCR fields (Tk `write` traces fire per character), and on every focus-out of the min-score spinbox even with no edit (`.set()` is unconditional).
- **Problem**: `on_ocr_parameter_change` — unlike `settings_changed_callback` directly above it (`app_logic.py:454-461`, which guards on `_fully_initialized/_suppress_traces/_ui_update_in_progress`) — checks **no guard and no changed-value comparison**. Each invocation (1) `clear_paddleocr_runtime_cache` drops all cached engines + invalidates prewarm, so the next OCR frame pays a full multi-second cold start, and (2) `publish_capture_ui_snapshot(bump_generation=True)` makes the capture worker run `_clear_capture_generation_state`, flushing the entire OCR frame cache + queue. Typing `0.50` = up to 4 engine destructions + 4 frame-cache flushes.
- **Evidence**: the asymmetry with `settings_changed_callback` is in the same file; the user's own WIP comment at `on_ocr_model_change` (`app_logic.py:1513-1518`) already documents that "Engine caches are keyed by their complete normalized settings, so retaining them is safe" — F4 is the natural continuation of that WIP into the sibling `on_ocr_parameter_change`.
- **Impact**: MED — user-interaction frequency, but each hit is a multi-second OCR stall (mid-session) or a background engine rebuild + loss of all cached frame results; worst while translation is running.
- **Fix**: in `on_ocr_parameter_change`, snapshot `self.get_current_paddleocr_settings()` and no-op (skip cache clear + generation bump) when equal to the last-applied settings; store the normalized tuple. In the focus-out handlers, call `.set()` only when the clamped value differs from the current string. **Risk note**: must compare *normalized* settings (already available) so genuine changes still invalidate.

### F5 — Log I/O executed while holding the unified cache's global lock  *(MED / Low — Agent 2, benchmarked)*

- **Location**: `unified_translation_cache.py:805-834` (`store`), `:766-789` (`get` HIT), `:841-861` (`_evict_lru_entries`), timer start `:655-675`.
- **Frequency**: `store` runs 2× per successful translation (dual-write peer contract — 3,904 STORE lines vs 1,952 translations in one 5 MB log); `get` runs 2–4× per request, including **on the Tk UI thread** (all submit paths call `start_async_translation` via `root.after`).
- **Problem**: The HIT/STORE/evict `log_debug` calls run **inside `with self.lock:`**. `log_debug` is not cheap: 5–6 regex passes + synchronous file `write()`+`flush()` under the writer lock, and can trigger a synchronous 5 MB×3 rotation. Any thread blocked on the cache lock during a flush stall — including the UI thread's per-subtitle instant-cache lookup — stalls with it. `_schedule_persistence_locked` also starts an OS thread (`Timer.start()`) under the lock.
- **Evidence (benchmarked in-repo)**: `store`=73.0 µs, `get` HIT=67.9 µs, of which `log_debug` ≈ 45 µs — **~2/3 of the lock-hold time is log I/O**. The log rotated through three 5 MB backups in a week, so rotation-under-lock does occur. The MISS path already uses `log_debug_coalesced` — the authors know these are hot; HIT/STORE just weren't given the same treatment.
- **Impact**: MED — classic lock-convoy tail risk; the blocked party can be the Tk event loop that displays subtitles.
- **Fix**: record the outcome in a local (`hit`, `evicted_n`, provider/langs) and emit `log_debug` **after** the `with self.lock:` block; switch STORE/HIT to `log_debug_coalesced` keyed like MISS. Optionally create the persistence `Timer` under the lock but `.start()` it after.

### F6 — ~13 unthrottled `log_debug` lines per translation, ~6 on the UI thread  *(MED / Low-Med — Agents 2 & 4)*

- **Location**: `worker_threads.py:1428, 1488-1492, 1508-1511, 1555, 1564-1567, 1589-1594, 1626-1631, 1637-1641`; `worker_translation.py:1212-1216`; `handlers/translation_requests.py:490-494, 1256-1261`.
- **Frequency**: ~13 lines per successful translation (log-category counts total ~25,000 of 50,649 lines in one 5 MB file; peak 67 lines/s); `process_translation_response` / `start_async_translation` emit 5–6 of them on the Tk UI thread.
- **Problem**: each `log_debug` pays 5–6 redaction regex passes + `strftime` + a double UTF-8 encode (`_encoded_write_size` re-encodes to size the write) + write+flush under the writer lock (measured **44.6 µs avg**, ms-level tails on disk/rotation stalls). ~13× ≈ 0.6 ms synchronous I/O per translation, partly on the very UI thread whose display latency the surrounding code measures.
- **Impact**: MED — pure overhead on the hottest pipeline; the flush-per-line design converts any disk hiccup into pipeline latency; also inflates F5's lock-hold windows.
- **Fix (safe subset)**: demote the purely-informational per-sequence lines ("Processing async translation", "completed in", "finished", "Processing newer sequence", "display scheduling took") to `log_debug_coalesced` (5 s). **Keep all error/warn paths verbatim** for diagnosability. (A larger logger-side option — buffer the debug writer and flush on a 0.5–1 s timer — is noted for later; more invasive, deferred.)

### F7 — `str.maketrans` rebuilt per call + OCR string re-normalized 4–10× per candidate  *(Low / Low — Agent 1, verified)*

- **Location**: `worker_ocr.py:115-132` (`_normalize_local_ocr_submit_text`), callers `:157-158, 234-244, 248-268, 344-370, 508-527`.
- **Problem (verified)**: line 120 builds a fresh 7-entry `str.maketrans` dict on every call; the same short strings are re-normalized many times per evaluation, and `_should_skip_local_ocr_resubmit` (`:518-527`) re-normalizes `last_local_ocr_submitted_text` from scratch even though `_remember_local_ocr_submit` (`:535-538`) already stored `last_local_ocr_submitted_norm`.
- **Impact**: Low — ~50–200 µs/candidate, zero-cost fix.
- **Fix**: hoist the table to a module constant `_SUBMIT_TEXT_TRANS`; in `_should_skip_local_ocr_resubmit` prefer the stored `last_local_ocr_submitted_norm`; optionally `functools.lru_cache(maxsize=256)` on `_normalize_local_ocr_submit_text`.

### F8 — PaddleOCR backend re-`convert("RGB")`s already-RGB images  *(Low-Med / Low — Agent 1, verified)*

- **Location**: `paddle_ocr_backend.py:703-708, 1030-1034, 1054, 1108-1113`; source `prepare_paddleocr_image` `:525-535`.
- **Problem (verified)**: `prepare_paddleocr_image` **always** returns a `convert("RGB")` result (`:529`), yet `:708` calls `.convert("RGB")` on it again (PIL same-mode convert = full copy) before `np.array` copies again; same at `:1054` for line crops. On the (routine) subtitle fast-path→fallback transition, the fully-prepared image is discarded and `recognize_with_paddleocr` re-runs `convert("RGB")` + bicubic upscale on the original — expensive interpolation done **twice** with `upscale > 1`.
- **Impact**: Low-Med — one redundant full-frame copy/frame + a duplicated prepare on every fallback frame.
- **Fix**: compute `prepared = prepare_paddleocr_image(...)` once in `recognize_subtitle_with_paddleocr`; thread it via optional `prepared_image=None` params to `prepare_paddleocr_subtitle_line_images` / `recognize_with_paddleocr`; replace `np.array(x.convert("RGB"))` with `np.asarray(x)` where the input is guaranteed RGB.

### F9 — `RuntimeMetrics.snapshot()` sorts each timing list twice  *(Low / Low — Agent 4)*

- **Location**: `runtime_metrics.py:77-99` (`snapshot`), `:149-156` (`_percentile`); single-metric callers `app_capture_ocr.py:142-151`, `worker_threads.py:557-579`.
- **Problem**: `snapshot()` calls `_percentile(values, 50)` then `_percentile(values, 90)`, and `_percentile` sorts on each call — so every timing list (≤240 samples) is sorted **twice** per snapshot, and reading one metric's p50 pays to percentile **all** metrics, under the shared RLock. `snapshot()` runs ~1.5–2×/s.
- **Impact**: Low — tens to a few hundred µs/snapshot of lock-held work.
- **Fix**: sort once per metric in `snapshot()` and index both percentiles; add `get_timing(name)` that sorts only the requested metric, and switch the two single-metric callers to it.

### F10 — Cache-key + profile/route identity recomputed with identical inputs  *(Low / Low-Med — Agents 2 & 3)*

- **Location**: `unified_translation_cache.py:64-113` (`_generate_cache_key`); `custom_ai_capabilities.py:132-140, 230-240, 770-794, 816-832`; `handlers/translation_requests.py:679-698`.
- **Problem**: for the live `custom_ai` provider, `_generate_cache_key` builds a dict → `json.dumps(sort_keys=True)` → MD5 on **every** `get`/`store` (≥4×/translation, byte-identical params); `_credential_scope_key` SHA-256s the same API key 6–10×/request; `_canonical_wire_endpoint_cache_key`/`_ordered_candidates` re-`urlsplit` the same URL 4–5×; `_translation_prompt_cache_key` re-`json.dumps`+SHA-256s an identical identity dict.
- **Impact**: Low — benchmarked `_generate_cache_key` = 23.2 µs/call (10-entry context); aggregate ~0.2–0.5 ms/translation, mostly UI-thread µs (URL cost largely absorbed by `urllib.parse`'s internal cache).
- **Fix**: memoize the `custom_ai` `params_hash` with a small `functools.lru_cache`d helper keyed on the normalized value tuple (context already arrives as a tuple → hashable). Optionally a bounded `_profile_derived(profile)` memo for credential-scope/endpoint/prompt keys. **Risk note**: cache identity must not drift — key the memo on the exact content tuple so a changed field naturally misses.

### F11 — Diagnostics metrics panel refreshes at 1 Hz for the app's whole life  *(Low-Med / Low — Agent 4)*

- **Location**: `gui_builder.py:366-392` (`_refresh_runtime_metrics_panel` self-reschedule), started `gui_diagnostics_builder.py:80-81`, cancelled only at shutdown `app_lifecycle.py:883-888`.
- **Problem**: the only gate is "widget exists" (true from init on), so every 1000 ms — even when the Debug tab was never opened and the app is idle — it does ~5 gauge sets (lock), a full `snapshot()` (F9 double-sorts), full text formatting, and a Tk `Text` delete+insert, producing identical content.
- **Impact**: Low-Med — permanent background churn (timer wakeups, metrics-lock contention, Tk re-layout) serving no one when the tab is hidden (the common case).
- **Fix**: refresh only while the Debug tab is selected (compare `tab_control.select()` to `tab_debug` at the top; keep rescheduling at a slower cadence when hidden), and/or skip the `Text` rewrite when `content` equals the previously rendered string.

### F12 — Fresh `mss` instance per screen grab  *(MED / Low-Med — Agent 1) — DOCUMENT / DEFER*

- **Location**: `ocr_utils.py:92-102` (`_capture_with_mss`), called per frame from `worker_capture.py:771` and every 500 ms from the Tk preview thread `app_capture_ocr.py:489-492`.
- **Problem**: `with factory() as sct:` constructs and closes an `mss.mss()` per grab. On Windows each instantiation builds new `ctypes.WinDLL` wrappers for user32/gdi32, sets argtypes/restypes for ~15 C funcs, makes a DPI call, and — because the cached bbox starts empty — reallocates the GDI DIB section + a `width*height*4` buffer (multi-MB) each grab, which `close()` frees. mss documents reusing one instance per thread.
- **Impact**: MED — ~0.5–3 ms setup/teardown + a multi-MB buffer/GDI alloc-free per frame, 10–20×/s forever, on the single hottest path.
- **Why deferred**: mss instances must not be shared across threads (capture thread + preview thread both call this), so the fix needs a `threading.local()` cache **plus** stale-handle recovery (drop+retry on grab failure after display-topology changes). That is the one item here with genuine thread-safety/correctness surface, so it is documented for an explicit decision rather than applied blind. **Recommended fix** (when accepted): module-level `_MSS_LOCAL = threading.local()`; reuse `_MSS_LOCAL.sct` when `mss_factory is None`, else create+store (no `with`); wrap `grab` in try/except that closes+discards the cached instance and retries once; keep the `mss_factory` test-injection bypassing the cache.

### F13 — Dead code (note only)

`worker_threads.py:400-454` (similarity/stability branch incl. `re.findall`) is unreachable: at `:388` `ocr_model` is always `PADDLEOCR_MODEL_CODE` and `_route_local_ocr_candidate_for_translation` returns only the four values that all `continue`. Zero runtime cost; can be deleted for clarity in a separate cleanup.

---

## 4. Fix plan for this pass

**Applied in this pass (10 findings), each verified against the test suite:**

| # | What changed | Files |
|---|---|---|
| F1 | Publish prewarm "ready" metrics once per generation; `note_paddleocr_first_ocr_wait` returns `None` on its dedup path | `app_logic.py` |
| F2 | Build the debug preview image only when OCR-debugging is on | `worker_capture.py` (+2 tests) |
| F3 | Hoist per-stream source invariants out of the per-SSE-chunk normalizer | `custom_ai_requests.py` |
| F4 | No-op `on_ocr_parameter_change` when normalized PaddleOCR settings are unchanged (no engine destroy / no frame-cache flush) | `app_logic.py` |
| F5 | Emit cache HIT/MISS/STORE logs **after** releasing the cache lock | `unified_translation_cache.py` |
| F7 | Hoist `str.maketrans` to a module constant; memoize the normalizer; prefer the stored norm | `worker_ocr.py` |
| F8 | Drop redundant `convert("RGB")`; reuse one prepared frame across fast-path + fallback | `paddle_ocr_backend.py` |
| F9 | Sort each timing list once in `snapshot()`; add `get_timing(name)` for single-metric callers | `runtime_metrics.py`, `app_capture_ocr.py`, `worker_threads.py` |
| F10 | Memoize the `custom_ai` params hash (byte-identical output — persisted cache keys verified stable) | `unified_translation_cache.py` |
| F11 | Skip the diagnostics `Text` re-layout when the rendered content is unchanged | `gui_builder.py` |

**Deferred (documented for an explicit decision, NOT applied):**

- **F6** (coalesce the ~13 per-translation info logs): changes developer-facing diagnostic verbosity, and the maintainer runs with `debug_logging_enabled=True` where per-sequence tracing is genuinely used. F5 already removed the dangerous part (log I/O under the cache lock), so the residual is pure micro-CPU. Left to the maintainer's judgment.
- **F12** (`mss` instance reuse): the only item with real thread-safety / stale-handle surface (display-topology changes). Needs `threading.local()` + grab-failure recovery that cannot be validated without hardware. Recommended fix is in §3/F12.
- **F13** (dead-code removal): separate cleanup.

**Constraints honored:** the maintainer's `on_ocr_model_change` WIP in `app_logic.py` is untouched (F1/F4 edit sibling methods only). F10 was proven byte-identical to the previous cache-key computation (a from-scratch reproduction of the old JSON+MD5 matched for every representative profile/context case), so no persisted SQLite entry is orphaned. No change alters emitted translations or error-log content.

**Verification:** full suite green — **827** tests in `tests/` + **35** root-level tests, plus 2 new tests (`test_process_local_ocr_frame_skips_debug_image_when_debugging_off` and the enabled-path assertion) covering F2.

---

## 5. Verified already-optimal (no action)

- `pyside_overlay.py` guards redundant redraws: `_apply_font_if_changed` (236-284), style caches (875-893), `_last_text_bg_color` (940-944).
- Capture loop uses immutable snapshots, ¼-res MD5 dedup (reused by the OCR thread), and coalesced logging; PaddleOCR engine caches avoid re-init on the hit path.
- SQLite persistence runs deltas under a separate `_persistence_lock` off the hot lock, debounced, with 60 s access-time throttling; `_log_custom_short_call` offloads its write to a background executor.
- `config_manager.py` snapshots settings into Tk vars/dataclasses at load — no repeated config reads on hot paths.
- Happy-path HTTP response bodies are parsed once; streaming line iteration is linear; the HTTP session is pooled; capability memory prevents re-sending known-rejected params.

---

## 6. Runtime log analysis — latest run (`translator_debug.log`, 2026-07-26 21:01)

Method: aggregate counts / timing distributions over the newest debug log (2,693 lines, 88 translations). No raw secret/body content read.

### L1 — Redundant capture-snapshot republishes  *(FIXED this pass)*

| Field | Detail |
|---|---|
| **Evidence** | Of 164 `published UI snapshot` events, **133 were the same `generation=10`** (21 more were `generation=11`). The periodic adaptive-interval refresh (`app_capture_ocr.py:234,306`, 2 s throttle) republishes every cycle even when no snapshot input changed. |
| **Cost** | Each republish rebuilt the snapshot, re-stored it, and wrote a per-publish log line (regex sanitize + file write/flush). |
| **Fix** | `worker_capture.py:publish_capture_ui_snapshot` now returns the existing object and skips the store + log when `bump_generation` is False and the freshly built `CaptureUISnapshot` equals the stored one. First publish and every real change still store + log; the bump path is unchanged. |
| **Tests** | +2 (`test_unchanged_republish_without_bump_is_skipped`, `test_changed_republish_updates_snapshot`). |

### Observations — NOT code-fixed (config / product tradeoffs, reported for the maintainer)

| # | Signal (this run) | Interpretation | Recommended action |
|---|---|---|---|
| O1 | **89% of translations complete via failover** (78/88); grok failover **errors 9×** (`ValueError - URL HTTP…`), **profile unavailable 7×**, **cooling-skip 12×**, **deadline exhausted 7×** | The first-choice profile is failing/slow almost every request, so nearly every translation pays a wasted primary attempt before falling over. This is largely **profile configuration**, not a code defect. | Check the primary Custom AI profile (endpoint/key/model). Consider making it the failover, or fixing the recurring grok `URL HTTP` error. |
| O2 | **Translation latency**: total median 2.45 s / **p90 10.2 s** / max 12.6 s; worker median 1.61 s / **p90 10.0 s** (API-bound); queue wait median 0.50 s / p90 1.17 s | The tail is dominated by the slow/failing route (O1), not local CPU. Overflow slow-route protection fires heavily (49 blocks, 11 entries) — the adaptive system working as designed under slow upstreams. | Resolving O1 should collapse the p90 tail. No code change advised. |
| O3 | **~91% of translations superseded** (80/88 `Processing newer sequence`; 55 stale-timer skips) | Because each translation can take up to 10 s (O1/O2), newer subtitles supersede in-flight ones; the staleness guards correctly discard the stale work. Symptom of O1, not a bug. | Follows from O1. |
| O4 | **Cache hit rate ≈ 0** (2 real HITs; MISS log is 5 s-coalesced so undercounted) vs 157 STOREs | The cache key includes rolling translation **context**, so a repeated subtitle with different surrounding context is a miss. This is the intended context-aware-quality tradeoff (prior audits' "H3 context-blind cache" was deferred as quality-risky). | Leave as-is unless a context-free fallback lookup is explicitly desired (quality tradeoff). |

**PaddleOCR note:** subtitle fast-path→full-fallback fired 73× (37 no-crop + 36 no-usable-text); 39 fallbacks produced text, 32 were empty. F8 already removed the fallback's redundant re-prepare; the remaining full-OCR cost on empty frames is accuracy-driven, not a clean CPU win.
