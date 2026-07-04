# Credential-Scoped Translation Cache Design

Changing the API key inside an existing profile leaves profile ID, URL, and
model unchanged. The current translation cache and in-flight identity therefore
reuse results created under the previous tenant.

Add the provider's non-reversible credential fingerprint to request cache
parameters, the unified Custom AI parameter hash, and the in-flight key.
Persistent entries from the previous credential naturally become misses.
