import base64
import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections import deque
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse, urlsplit, urlunsplit

from credential_store import create_default_credential_store
from logger import log_debug
from ocr_utils import normalize_api_ocr_image_detail


class _HTMLTitleParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._in_title = False
        self.title_parts = []

    def handle_starttag(self, tag, attrs):
        if str(tag).lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if str(tag).lower() == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(str(data))


ACTIVE_PROFILE_KINDS = {"translation", "ocr"}
CUSTOM_AI_CREDENTIAL_SERVICE = "OCR-Translator-CustomAI"
CUSTOM_AI_PROFILE_TEMP_STALE_SECONDS = 300.0
CUSTOM_AI_LATENCY_MODE_NONE = "none"
CUSTOM_AI_LATENCY_MODE_SAFE = "safe"
CUSTOM_AI_LATENCY_MODE_STREAM = "stream"
CUSTOM_AI_LATENCY_MODE_RACE = "race"
CUSTOM_AI_LATENCY_MODE_ADAPTIVE = "adaptive"
CUSTOM_AI_LATENCY_MODES = {
    CUSTOM_AI_LATENCY_MODE_NONE,
    CUSTOM_AI_LATENCY_MODE_SAFE,
    CUSTOM_AI_LATENCY_MODE_STREAM,
    CUSTOM_AI_LATENCY_MODE_RACE,
    CUSTOM_AI_LATENCY_MODE_ADAPTIVE,
}
CUSTOM_AI_SAFE_TRANSPORT_MODES = {
    CUSTOM_AI_LATENCY_MODE_SAFE,
    CUSTOM_AI_LATENCY_MODE_STREAM,
    CUSTOM_AI_LATENCY_MODE_RACE,
}
CUSTOM_AI_WIRE_API_CHAT_COMPLETIONS = "chat_completions"
CUSTOM_AI_WIRE_API_RESPONSES = "responses"
CUSTOM_AI_WIRE_APIS = {
    CUSTOM_AI_WIRE_API_CHAT_COMPLETIONS,
    CUSTOM_AI_WIRE_API_RESPONSES,
}
CUSTOM_AI_STRUCTURED_OUTPUT_OFF = "off"
CUSTOM_AI_STRUCTURED_OUTPUT_AUTO = "auto"
CUSTOM_AI_STRUCTURED_OUTPUT_STRICT = "strict"
CUSTOM_AI_STRUCTURED_OUTPUT_MODES = {
    CUSTOM_AI_STRUCTURED_OUTPUT_OFF,
    CUSTOM_AI_STRUCTURED_OUTPUT_AUTO,
    CUSTOM_AI_STRUCTURED_OUTPUT_STRICT,
}
CUSTOM_AI_STRUCTURED_OUTPUT_CONTRACT_TEXT = "text"
CUSTOM_AI_STRUCTURED_OUTPUT_CONTRACT_JSON_SCHEMA = "json_schema"
CUSTOM_AI_REASONING_EFFORT_LOW = "low"
CUSTOM_AI_REASONING_EFFORT_MEDIUM = "medium"
CUSTOM_AI_REASONING_EFFORT_HIGH = "high"
CUSTOM_AI_REASONING_EFFORT_ULTRA = "ultra"
CUSTOM_AI_REASONING_EFFORT_NONE = "none"
CUSTOM_AI_REASONING_EFFORTS = {
    CUSTOM_AI_REASONING_EFFORT_NONE,
    CUSTOM_AI_REASONING_EFFORT_LOW,
    CUSTOM_AI_REASONING_EFFORT_MEDIUM,
    CUSTOM_AI_REASONING_EFFORT_HIGH,
    CUSTOM_AI_REASONING_EFFORT_ULTRA,
}
CUSTOM_AI_REASONING_EFFORT_CONTRACTS = {
    *CUSTOM_AI_REASONING_EFFORTS,
    CUSTOM_AI_REASONING_EFFORT_NONE,
}
CUSTOM_AI_REASONING_REQUEST_KINDS = {
    "translation",
    "ocr",
}
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 15.0
TRANSLATION_MIN_OUTPUT_TOKENS = 64
TRANSLATION_MAX_OUTPUT_TOKENS = 2048
TRANSLATION_OUTPUT_TOKENS_PER_CHAR = 4
TRANSLATION_OUTPUT_WRAPPER_LABELS = frozenset({
    "translation:",
    "translation：",
    "translated text:",
    "translated text：",
    "translation result:",
    "translation result：",
    "译文:",
    "译文：",
    "翻译:",
    "翻译：",
    "翻译结果:",
    "翻译结果：",
})
TRANSLATION_OUTPUT_PREAMBLES = (
    "sure, here is the translation:",
    "sure, here's the translation:",
    "certainly, here is the translation:",
    "certainly, here's the translation:",
    "here is the translation:",
    "here's the translation:",
)


from custom_ai_policy import (
    CustomAILatencyModeAdvisor,
    CustomAILatencyModeDecision,
    CustomAIRequestTimeoutDecision,
    build_translation_json_schema,
    build_translation_response_format,
    normalize_custom_ai_latency_mode,
    normalize_custom_ai_reasoning_effort,
    normalize_custom_ai_reasoning_request_kind,
    normalize_custom_ai_structured_output_mode,
    normalize_custom_ai_wire_api,
)
from custom_ai_profiles import CustomAIProfileManager
from custom_ai_capabilities import CustomAICapabilitiesMixin
from custom_ai_requests import CustomAIRequestsMixin
from custom_ai_transport import CustomAITransportMixin


class CustomAIProvider(CustomAICapabilitiesMixin, CustomAIRequestsMixin, CustomAITransportMixin):
    """OpenAI Chat Completions compatible translation and OCR provider."""

    def __init__(self, http_client=None, timeout=30):
        self.http_client = http_client
        self._owns_http_client = http_client is None
        self.timeout = timeout
        self.context_window = []
        self._client_lock = threading.Lock()
        self._url_cache_lock = threading.Lock()
        self._rate_limit_lock = threading.Lock()
        self._capability_lock = threading.Lock()
        self._successful_chat_urls = {}
        self._successful_responses_urls = {}
        self._successful_models_urls = {}
        self._rate_limit_cooldowns = {}
        self._rate_limit_backoff_counts = {}
        self._profile_unavailable_cooldowns = {}
        self._profile_health_request_sequences = {}
        self._profile_health_event_sequences = {}
        self._unsupported_output_limit_keys = set()
        self._unsupported_structured_output_keys = set()
        self._unsupported_reasoning_effort_keys = set()
        self._unsupported_prompt_cache_key_keys = set()
