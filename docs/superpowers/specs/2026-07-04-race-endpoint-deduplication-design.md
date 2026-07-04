# Race Endpoint Deduplication Design

Race identity currently prefers profile ID. Separate profiles with different
IDs but the same normalized base URL, model, wire API, and reasoning effort are
therefore called in parallel despite representing the same endpoint.

Define race execution identity from normalized endpoint semantics only. Use it
for candidate deduplication and the cross-race in-flight set. Profile IDs remain
available for UI and cache ownership, but no longer justify duplicate network
calls to an identical endpoint.
