# Single Translation Request Snapshot Design

The normal translation path independently resolved profile/cache parameters for
its initial lookup, resolved them again inside the cache helper, and read
context a third time for the provider call. Concurrent context changes could
therefore make lookup, provider input, and stored cache identity disagree.

Resolve profile, languages, prompt, line-break mode, and context once. Pass the
resolved cache parameters into the cache helper, derive provider inputs from
the same immutable values, and store under that identity. This also removes two
context-window scans from every cache miss.
