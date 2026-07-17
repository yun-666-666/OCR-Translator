# Local Capture Baseline Floor Design

## Goal

Ensure the local OCR capture thread never runs faster than the adaptive/user scan interval while preserving the existing queue-pressure slowdown and gradual recovery behavior.

## Scope

This increment changes only the local-OCR interval calculation in `worker_capture.py` and focused tests in `tests/test_latency_optimization.py`. API OCR pacing, capture backpressure, duplicate-frame handling, OCR processing, UI settings, and adaptive interval generation remain unchanged.

## Design

Extract the three existing local queue-pressure branches into `_next_local_capture_interval(base_scan_interval, current_scan_interval, queue_fullness)`.

- Above 70% queue fullness: preserve `base * (1 + fullness)`.
- Above 40% queue fullness: preserve `base * 1.25`.
- At or below 40%: preserve the 5% gradual decay but clamp to `base`, not the hard 50 ms safety floor.

The capture loop continues to compute `base_scan_interval = max(0.05, current_scan_interval / 1000)` and calls the helper only for local OCR. No thread, queue, or Tk ownership changes.

## Testing

Pure unit tests prove:

- An empty queue with a 200 ms base and 50 ms current state returns 200 ms.
- A previously elevated interval decays by 5% but never below the base.
- The 40%/70% pressure thresholds keep their previous multipliers.

Then run the capture/latency-focused tests, the full `tests/` suite, compilation, and diff hygiene.

## Risk assessment

Risk is low: the helper is a direct extraction of existing arithmetic with one corrected lower bound. A larger `worker_capture.py` split is explicitly out of scope because it would combine scheduling, capture, queue, and UI/Tk concerns without a measured need.
