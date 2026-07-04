# Translation Error Redaction Design

OCR errors sanitize the active API key, but translation and race aggregation
currently format arbitrary provider exceptions directly into logs and UI text.

Add one profile-aware sanitizer helper in the translation handler. Use it for
the main translation catch and sanitize each race candidate exception before
aggregation. Preserve exception type and diagnostic text while replacing the
specific profile credential.
