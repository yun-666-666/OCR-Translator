# Race Cache Request Snapshot Design

Race translations currently rebuild cache parameters after the winning request
finishes. If prompt, line-break, or context settings change while HTTP calls
are active, a result generated with old inputs is stored under the new cache
identity.

Capture custom prompt, line-break mode, and approved context before dispatch.
Use that immutable snapshot for every candidate request, the winner cache key,
and the active-profile alias cache key. Normal cache-key construction remains
unchanged for non-race callers.

Verification changes settings inside a provider call and proves returned cache
parameters still describe the actual request inputs.
