# Race Wire Endpoint Normalization Design

Simple trailing-slash normalization misses equivalent configured forms such as
`/v1` and `/v1/chat/completions`, or `/v1` and `/v1/responses`.

Build race execution identity from the first canonical request URL produced by
the provider's wire-specific URL candidate builder. Keep wire API in the
signature so Chat and Responses remain distinct.
