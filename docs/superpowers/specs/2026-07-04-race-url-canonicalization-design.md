# Race URL Canonicalization Design

## Problem

Wire-aware endpoint normalization still compares URL strings literally.
Scheme and hostname are case-insensitive, and explicit default ports are
network-equivalent, so profiles such as `HTTPS://HOST:443/v1` and
`https://host/v1/chat/completions` can still launch duplicate race requests.

## Design

After wire-specific endpoint expansion, canonicalize only RFC-safe URL parts:

- lowercase scheme and hostname;
- remove port 443 for HTTPS and port 80 for HTTP;
- preserve user info, non-default ports, path, query, and fragment exactly;
- preserve path case.

Malformed URLs fall back to the existing string identity.

## Verification

- HTTPS/443 and HTTP/80 aliases compare equal.
- Non-default ports remain distinct.
- Path case remains distinct.
- Existing Chat/Responses endpoint-path normalization remains green.
