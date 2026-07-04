# Latency Mode Request Snapshot Design

Worker callback construction and provider execution currently read latency mode
at different times. A setting change between them can mismatch stream callback,
race/safe transport, and duplicate in-flight identity.

Include normalized latency mode in the in-flight key. Capture it once in the
worker, pass it through `translate_text_with_timeout()` and `translate_text()`,
and let `_custom_ai_translate()` use the supplied snapshot instead of rereading
UI state. Direct callers without an override retain current behavior.
