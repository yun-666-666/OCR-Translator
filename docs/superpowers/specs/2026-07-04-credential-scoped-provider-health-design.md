# Credential-Scoped Provider Health Design

Equivalent endpoints may be backed by distinct API accounts with independent
quota and cooldown. Endpoint-only identity incorrectly deduplicates those race
candidates and shares rate-limit backoff.

Add a short SHA-256 fingerprint of the API key to race execution identity and
rate-limit keys. The raw secret is never stored or logged. Empty credentials
share one stable anonymous scope. Successful URL and parameter-capability
learning remain endpoint-scoped.
