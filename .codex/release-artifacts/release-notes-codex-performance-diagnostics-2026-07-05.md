Runtime performance diagnostics release for branch codex/optimized-sync-2026-07-02.

Target commit: 83dba5e3edf53724a8494f608c23ab4ebc3eee8c
Target branch: codex/optimized-sync-2026-07-02

Highlights:
- Added an in-memory runtime metrics collector with bounded timing windows.
- Added a Debug tab Performance Diagnostics panel with periodic refresh,
  Reset metrics, and Copy summary.
- Added low-overhead instrumentation for capture, OCR, translation queue/API
  latency, cache hits, in-flight skips, stale discards, stream partials, race
  winner labels, provider cooldown, active translation calls, and OCR queue size.
- Race winner/provider labels and copied summaries are sanitized before display.

Uploaded asset:
- OCR-Translator-clean-source-2026-07-05_2026-07-05_14-07-41.zip
- SHA256: CD63D4587FBCC689219B48F76458A059B847DF966D63957CABD353F04488488D

Verification:
- python -m unittest discover -s tests: 278 passed.
- ZIP entry audit found no .git/.codex/runtime config/cache/log/generated paths.
- Secret-pattern scan found no OpenAI/Google/GitHub/Slack/AWS-style tokens.

This release is intentionally created from the optimization branch, not main.
