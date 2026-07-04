# Canonical Translation Cache Endpoint Design

Custom AI cache parameters currently retain configured base URL spelling.
Network-equivalent edits therefore invalidate translation cache and in-flight
identity despite sending requests to the same final endpoint.

Resolve the wire-specific first request URL and canonicalize it with the
provider's base URL key function before adding it to cache parameters. Chat and
Responses remain distinct because their final endpoint paths differ.
