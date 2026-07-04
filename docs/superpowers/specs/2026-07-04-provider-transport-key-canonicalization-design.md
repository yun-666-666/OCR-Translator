# Provider Transport Key Canonicalization Design

Provider cooldown, output-limit capability, and successful URL caches currently
key on a trimmed but otherwise literal base URL. Network-equivalent aliases
split transport learning and cooldown state.

Canonicalize scheme, hostname, and default ports while preserving path
semantics. When reusing a successful concrete URL through an alias, match it to
the current candidate spelling by canonical URL so the cached candidate is
still promoted.
