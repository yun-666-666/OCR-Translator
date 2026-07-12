import re
from urllib.parse import urlsplit


_URL_RE = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
_SENSITIVE_PAYLOAD_RE = re.compile(
    r"(?P<prefix>(?<![\w-])['\"]?"
    r"(?:authorization|cookie|response_body|content)"
    r"['\"]?\s*[:=]\s*)[^\r\n]*",
    re.IGNORECASE,
)
_NAMED_SECRET_RE = re.compile(
    r"(?P<prefix>(?<![\w-])['\"]?"
    r"(?:api[_-]?key|token|secret|password)"
    r"['\"]?\s*[:=]\s*)"
    r"(?P<value>\"(?:\\.|[^\"\\\r\n])*\"|"
    r"'(?:\\.|[^'\\\r\n])*'|[^\r\n,;]+)",
    re.IGNORECASE,
)


def sanitize_url(value):
    """Return an HTTP(S) URL containing only its public location."""
    try:
        parsed = urlsplit(str(value).strip())
        scheme = parsed.scheme.lower()
        host = parsed.hostname
        port = parsed.port
    except Exception:
        return "[redacted-url]"

    if scheme not in ("http", "https") or not host:
        return "[redacted-url]"

    host = host.lower()
    if re.search(r"[\s\x00-\x1f\x7f/\\?#@]", host):
        return "[redacted-url]"

    if ":" in host:
        host = f"[{host}]"

    default_port = 80 if scheme == "http" else 443
    if port is not None and port != default_port:
        host = f"{host}:{port}"

    return f"{scheme}://{host}{parsed.path}"


def sanitize_error_text(message, known_secrets=(), max_length=500):
    """Return bounded diagnostic text without credentials or response content."""
    try:
        text = str(message)
    except Exception:
        text = "[redacted]"

    text = _URL_RE.sub(lambda match: sanitize_url(match.group(0)), text)

    for secret in known_secrets or ():
        try:
            secret_text = str(secret)
        except Exception:
            continue
        if secret_text:
            text = text.replace(secret_text, "[redacted]")

    text = _SENSITIVE_PAYLOAD_RE.sub(
        lambda match: f"{match.group('prefix')}[redacted]", text
    )
    text = _NAMED_SECRET_RE.sub(
        lambda match: f"{match.group('prefix')}[redacted]", text
    )
    text = re.sub(r"\s+", " ", text).strip()

    if max_length > 0 and len(text) > max_length:
        if max_length <= 3:
            return "..."[:max_length]
        return f"{text[: max_length - 3].rstrip()}..."
    return text
