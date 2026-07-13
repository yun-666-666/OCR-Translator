"""Immutable, secret-safe identity for one API OCR submission."""

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
import hashlib
import ipaddress
import json
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple
from urllib.parse import urlsplit, urlunsplit


_DEFAULT_IMAGE_DETAIL = "auto"
_DEFAULT_IMAGE_FORMAT = "webp"
_DEFAULT_IMAGE_MODE = "balanced_webp"
_DEFAULT_IMAGE_QUALITY = 85
_IMAGE_DETAILS = {"auto", "low", "high"}
_IMAGE_FORMATS = {"webp", "png", "jpeg"}
_MIME_BY_IMAGE_FORMAT = {
    "webp": "image/webp",
    "png": "image/png",
    "jpeg": "image/jpeg",
}
_IMAGE_MODES = {
    "lossless_webp",
    "balanced_webp",
    "small_grayscale_webp",
}


class _FrozenList(tuple):
    """Private immutable representation of a profile list."""

    __slots__ = ()


class _FrozenTuple(tuple):
    """Private immutable representation of a profile tuple."""

    __slots__ = ()


class _FrozenSet(frozenset):
    """Private immutable representation of a profile set."""

    __slots__ = ()


class _FrozenFrozenSet(frozenset):
    """Private immutable representation of a profile frozenset."""

    __slots__ = ()


_SAFE_PROFILE_PRIMITIVE_TYPES = {
    type(None),
    bool,
    int,
    float,
    str,
    bytes,
}
_SUPPORTED_PROFILE_CONTAINER_TYPES = {dict, list, tuple, set, frozenset}


def _freeze_profile_value(value: Any, active_container_ids=None) -> Any:
    """Freeze safe profile data without preserving aliases or arbitrary objects."""
    value_type = type(value)
    if value_type in _SAFE_PROFILE_PRIMITIVE_TYPES:
        return value
    if value_type not in _SUPPORTED_PROFILE_CONTAINER_TYPES:
        raise TypeError(
            "Unsupported API OCR profile value type: {}".format(
                value_type.__name__
            )
        )

    if active_container_ids is None:
        active_container_ids = set()
    value_id = id(value)
    if value_id in active_container_ids:
        raise ValueError("API OCR profile values must not contain cycles")
    active_container_ids.add(value_id)
    try:
        if value_type is dict:
            return MappingProxyType(
                {
                    _freeze_profile_value(key, active_container_ids): (
                        _freeze_profile_value(item, active_container_ids)
                    )
                    for key, item in value.items()
                }
            )
        if value_type is list:
            return _FrozenList(
                _freeze_profile_value(item, active_container_ids)
                for item in value
            )
        if value_type is tuple:
            return _FrozenTuple(
                _freeze_profile_value(item, active_container_ids)
                for item in value
            )
        if value_type is set:
            return _FrozenSet(
                _freeze_profile_value(item, active_container_ids)
                for item in value
            )
        return _FrozenFrozenSet(
            _freeze_profile_value(item, active_container_ids)
            for item in value
        )
    finally:
        active_container_ids.remove(value_id)


def _thaw_profile_value(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {
            _thaw_profile_value(key): _thaw_profile_value(item)
            for key, item in value.items()
        }
    if isinstance(value, _FrozenList):
        return [_thaw_profile_value(item) for item in value]
    if isinstance(value, _FrozenTuple):
        return tuple(_thaw_profile_value(item) for item in value)
    if isinstance(value, _FrozenSet):
        return {_thaw_profile_value(item) for item in value}
    if isinstance(value, _FrozenFrozenSet):
        return frozenset(_thaw_profile_value(item) for item in value)
    if type(value) in _SAFE_PROFILE_PRIMITIVE_TYPES:
        return value
    raise TypeError(
        "Unexpected frozen API OCR profile value type: {}".format(
            type(value).__name__
        )
    )


def _normalized_token(value: Any, default: str = "") -> str:
    normalized = str(value or default).strip().lower().replace("-", "_")
    return normalized or default


def _normalize_provider(value: Any) -> str:
    return "_".join(_normalized_token(value).split())


def _normalize_wire_api(value: Any) -> str:
    normalized = _normalized_token(value)
    if normalized in {"response", "responses", "openai_responses"}:
        return "responses"
    if normalized in {
        "chat",
        "chat_completion",
        "chat_completions",
        "openai_chat_completions",
    }:
        return "chat_completions"
    return "chat_completions"


def _normalize_endpoint_hostname(hostname: str) -> str:
    try:
        parsed_address = ipaddress.ip_address(hostname)
    except ValueError:
        if all(character in "0123456789." for character in hostname):
            raise ValueError
        try:
            normalized_hostname = hostname.encode("idna").decode("ascii").lower()
        except UnicodeError:
            raise ValueError from None
        domain_without_root = normalized_hostname.rstrip(".")
        if not domain_without_root or len(domain_without_root) > 253:
            raise ValueError
        for label in domain_without_root.split("."):
            if (
                not label
                or len(label) > 63
                or label.startswith("-")
                or label.endswith("-")
                or not all(
                    character.isascii()
                    and (character.isalnum() or character == "-")
                    for character in label
                )
            ):
                raise ValueError
        return normalized_hostname
    if parsed_address.version == 6:
        return "[{}]".format(parsed_address.compressed.lower())
    return str(parsed_address)


def _validate_raw_endpoint_authority(raw_url: str) -> None:
    if any(
        character.isspace()
        or ord(character) < 32
        or ord(character) == 127
        for character in raw_url
    ):
        raise ValueError
    scheme_separator = raw_url.find("://")
    if scheme_separator < 1:
        raise ValueError
    authority_start = scheme_separator + 3
    authority_end = len(raw_url)
    for separator in "/?#":
        position = raw_url.find(separator, authority_start)
        if position >= 0:
            authority_end = min(authority_end, position)
    authority = raw_url[authority_start:authority_end]
    if not authority or "\\" in authority or "%" in authority:
        raise ValueError


def _authority_has_explicit_port(netloc: str) -> bool:
    host_and_port = netloc.rsplit("@", 1)[-1]
    if host_and_port.startswith("["):
        closing_bracket = host_and_port.find("]")
        return closing_bracket >= 0 and len(host_and_port) > closing_bracket + 1
    return ":" in host_and_port


def _normalize_endpoint(value: Any) -> Tuple[str, str]:
    raw_url = str(value or "").strip()
    if not raw_url:
        raise ValueError("Invalid API OCR base URL")
    try:
        _validate_raw_endpoint_authority(raw_url)
        parts = urlsplit(raw_url)
        scheme = parts.scheme.lower()
        hostname = parts.hostname or ""
        port = parts.port
        if scheme not in {"http", "https"} or not hostname:
            raise ValueError
        hostname = _normalize_endpoint_hostname(hostname)
        if _authority_has_explicit_port(parts.netloc) and port is None:
            raise ValueError
        if port is not None and not 1 <= port <= 65535:
            raise ValueError
        if port is not None and not (
            (scheme == "http" and port == 80)
            or (scheme == "https" and port == 443)
        ):
            hostname = "{}:{}".format(hostname, port)
        path = (parts.path or "").rstrip("/")
        normalized_url = urlunsplit((scheme, hostname, path, "", ""))
        endpoint_scope_values = (
            parts.username or "",
            parts.password or "",
            parts.query or "",
        )
        if any(endpoint_scope_values):
            serialized_scope = json.dumps(
                ["api-ocr-endpoint-scope-v1", *endpoint_scope_values],
                ensure_ascii=True,
                separators=(",", ":"),
            )
            endpoint_scope_digest = hashlib.sha256(
                serialized_scope.encode("utf-8")
            ).hexdigest()
        else:
            endpoint_scope_digest = ""
        return normalized_url, endpoint_scope_digest
    except (TypeError, ValueError):
        raise ValueError("Invalid API OCR base URL") from None


def _credential_scope_digest(profile: Mapping[str, Any]) -> str:
    api_key = str(profile.get("api_key") or "")
    if api_key:
        payload = api_key
    else:
        credential_ref = str(profile.get("api_key_ref") or "").strip()
        if not credential_ref:
            return ""
        payload = "credential-ref:{}".format(credential_ref)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalize_image_detail(value: Any) -> str:
    normalized = _normalized_token(value, _DEFAULT_IMAGE_DETAIL)
    return normalized if normalized in _IMAGE_DETAILS else _DEFAULT_IMAGE_DETAIL


def _normalize_image_format(value: Any) -> str:
    normalized = _normalized_token(value, _DEFAULT_IMAGE_FORMAT)
    if normalized == "jpg":
        normalized = "jpeg"
    return normalized if normalized in _IMAGE_FORMATS else _DEFAULT_IMAGE_FORMAT


def _normalize_image_mode(value: Any) -> str:
    normalized = _normalized_token(value, _DEFAULT_IMAGE_MODE)
    return normalized if normalized in _IMAGE_MODES else _DEFAULT_IMAGE_MODE


def _normalize_image_quality(value: Any) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = _DEFAULT_IMAGE_QUALITY
    return max(1, min(100, normalized))


def _normalize_mime_type(value: Any, image_format: str) -> str:
    expected_mime = _MIME_BY_IMAGE_FORMAT[image_format]
    normalized = str(value or "").strip().lower()
    if not normalized:
        return expected_mime
    if normalized == "image/jpg":
        normalized = "image/jpeg"
    if normalized not in _MIME_BY_IMAGE_FORMAT.values():
        raise ValueError("Unsupported API OCR image MIME type")
    if normalized != expected_mime:
        raise ValueError("API OCR image format and MIME type do not match")
    return normalized


def _stable_identity(namespace: str, values: Tuple[Any, ...]) -> str:
    serialized = json.dumps(
        values,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return "api_ocr:{}:{}".format(namespace, digest)


@dataclass(frozen=True, repr=False, init=False)
class ApiOcrRequestSnapshot:
    """Frozen request fields shared by API OCR execution and cache bookkeeping."""

    generation: int
    sequence: int
    provider: str
    profile_id: str
    model: str
    base_url: str
    wire_api: str
    credential_scope_digest: str
    endpoint_scope_digest: str
    source_language: str
    keep_linebreaks: bool
    latency_mode: str
    reasoning_effort: str
    image_detail: str
    image_format: str
    image_mode: str
    image_quality: int
    mime_type: str
    _profile: Mapping[str, Any] = field(repr=False, compare=False, hash=False)

    @classmethod
    def _from_frozen(cls, **values: Any) -> "ApiOcrRequestSnapshot":
        instance = object.__new__(cls)
        for field_name, field_value in values.items():
            object.__setattr__(instance, field_name, field_value)
        return instance

    @classmethod
    def create(
        cls,
        *,
        generation: Any,
        sequence: Any,
        provider: Any,
        profile: Mapping[str, Any],
        source_language: Any,
        keep_linebreaks: Any,
        latency_mode: Any,
        reasoning_effort: Any,
        image_detail: Any,
        image_format: Any,
        image_mode: Any,
        image_quality: Any,
        mime_type: Any,
    ) -> "ApiOcrRequestSnapshot":
        profile_data = dict(profile or {})
        normalized_reasoning = _normalized_token(reasoning_effort)
        profile_data["reasoning_effort"] = normalized_reasoning
        normalized_base_url, endpoint_scope_digest = _normalize_endpoint(
            profile_data.get("base_url")
        )
        frozen_profile = _freeze_profile_value(profile_data)
        normalized_format = _normalize_image_format(image_format)
        normalized_mime = _normalize_mime_type(mime_type, normalized_format)
        return cls._from_frozen(
            generation=int(generation),
            sequence=int(sequence),
            provider=_normalize_provider(provider),
            profile_id=str(profile_data.get("id") or "").strip(),
            model=str(profile_data.get("model") or "").strip(),
            base_url=normalized_base_url,
            wire_api=_normalize_wire_api(profile_data.get("wire_api")),
            credential_scope_digest=_credential_scope_digest(profile_data),
            endpoint_scope_digest=endpoint_scope_digest,
            source_language=str(source_language or "").strip(),
            keep_linebreaks=bool(keep_linebreaks),
            latency_mode=_normalized_token(latency_mode, "safe"),
            reasoning_effort=normalized_reasoning,
            image_detail=_normalize_image_detail(image_detail),
            image_format=normalized_format,
            image_mode=_normalize_image_mode(image_mode),
            image_quality=_normalize_image_quality(image_quality),
            mime_type=normalized_mime,
            _profile=frozen_profile,
        )

    @property
    def request_token(self) -> Tuple[int, int]:
        return (self.generation, self.sequence)

    @property
    def profile_view(self) -> Mapping[str, Any]:
        """Return read-only, content-safe metadata for diagnostics and inspection."""
        return MappingProxyType(
            {
                "id": self.profile_id,
                "base_url": self.base_url,
                "model": self.model,
                "wire_api": self.wire_api,
                "credential_scope_digest": self.credential_scope_digest,
                "endpoint_scope_digest": self.endpoint_scope_digest,
            }
        )

    def profile_copy(self) -> Dict[str, Any]:
        """Return fresh raw provider input, including credentials. Do not log it."""
        return _thaw_profile_value(self._profile)

    @property
    def cache_model_identity(self) -> str:
        return _stable_identity(
            "model",
            (
                self.provider,
                self.profile_id,
                self.base_url,
                self.model,
                self.wire_api,
                self.credential_scope_digest,
                self.endpoint_scope_digest,
            ),
        )

    @property
    def cache_mode_identity(self) -> str:
        return _stable_identity(
            "mode",
            (
                self.source_language,
                self.keep_linebreaks,
                self.latency_mode,
                self.reasoning_effort,
                self.image_detail,
                self.image_format,
                self.image_mode,
                self.image_quality,
                self.mime_type,
            ),
        )

    def __repr__(self) -> str:
        return (
            "ApiOcrRequestSnapshot(request_token={!r}, "
            "cache_model_identity={!r}, cache_mode_identity={!r})"
        ).format(
            self.request_token,
            self.cache_model_identity,
            self.cache_mode_identity,
        )
