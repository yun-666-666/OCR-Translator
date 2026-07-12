"""Persistent Custom AI profile management."""

import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

from credential_store import create_default_credential_store as _base_credential_store_factory
from custom_ai_policy import (
    ACTIVE_PROFILE_KINDS,
    CUSTOM_AI_CREDENTIAL_SERVICE,
    CUSTOM_AI_PROFILE_TEMP_STALE_SECONDS,
    CUSTOM_AI_REASONING_EFFORT_LOW,
    CUSTOM_AI_STRUCTURED_OUTPUT_AUTO,
    CUSTOM_AI_WIRE_API_CHAT_COMPLETIONS,
    normalize_custom_ai_reasoning_effort,
    normalize_custom_ai_structured_output_mode,
    normalize_custom_ai_wire_api,
)


def _facade():
    return sys.modules.get("custom_ai")


def _log_debug(message):
    facade = _facade()
    if facade is not None:
        return facade.log_debug(message)


def _create_default_credential_store(*args, **kwargs):
    facade = _facade()
    factory = getattr(facade, "create_default_credential_store", None)
    if factory is None:
        factory = _base_credential_store_factory
    return factory(*args, **kwargs)


class CredentialPersistenceError(RuntimeError):
    """Raised when a secret cannot be stored without plaintext fallback."""


class CustomAIProfileManager:
    """Persist and manage user-defined OpenAI-compatible AI endpoint profiles."""

    def __init__(self, path="custom_ai_profiles.json", credential_store=None):
        self.path = Path(path)
        self.credential_store = credential_store or _create_default_credential_store(CUSTOM_AI_CREDENTIAL_SERVICE)
        self._data_lock = threading.RLock()
        self._transaction_lock = threading.RLock()
        self.data = {
            "profiles": [],
            "active_translation_profile_id": None,
            "active_ocr_profile_id": None,
        }
        self._cleanup_stale_atomic_temp_files()
        self.load()

    def _cleanup_stale_atomic_temp_files(self):
        cutoff = time.time() - CUSTOM_AI_PROFILE_TEMP_STALE_SECONDS
        pattern = f".{self.path.name}.*.tmp"
        try:
            candidates = list(self.path.parent.glob(pattern))
        except Exception as error:
            _log_debug(
                "Custom AI profiles temporary-file scan failed: "
                f"{type(error).__name__}"
            )
            return
        for candidate in candidates:
            try:
                if candidate.stat().st_mtime > cutoff:
                    continue
                candidate.unlink()
            except Exception as error:
                _log_debug(
                    "Custom AI profiles stale temporary-file cleanup failed: "
                    f"{type(error).__name__}"
                )

    def load(self):
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8-sig") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.data.update({
                    "profiles": loaded.get("profiles", []),
                    "active_translation_profile_id": loaded.get("active_translation_profile_id"),
                    "active_ocr_profile_id": loaded.get("active_ocr_profile_id"),
                })
                if self._sanitize():
                    self.save()
        except Exception as e:
            _log_debug(f"Custom AI profiles load failed: {e}")

    def _snapshot_data(self):
        with self._data_lock:
            return {
                "profiles": [
                    dict(profile)
                    for profile in self.data.get("profiles", [])
                    if isinstance(profile, dict)
                ],
                "active_translation_profile_id": self.data.get(
                    "active_translation_profile_id"
                ),
                "active_ocr_profile_id": self.data.get(
                    "active_ocr_profile_id"
                ),
            }

    def _publish_data(self, staged_data):
        published_data = {
            "profiles": [
                dict(profile)
                for profile in staged_data.get("profiles", [])
                if isinstance(profile, dict)
            ],
            "active_translation_profile_id": staged_data.get(
                "active_translation_profile_id"
            ),
            "active_ocr_profile_id": staged_data.get(
                "active_ocr_profile_id"
            ),
        }
        with self._data_lock:
            self.data = published_data

    def _save_staged_data(self, staged_data):
        try:
            return bool(self.save(staged_data))
        except CredentialPersistenceError:
            raise
        except Exception as error:
            _log_debug(
                "Custom AI profiles staged save failed: "
                f"{type(error).__name__}"
            )
            return False

    def save(self, data=None):
        temporary_path = None
        try:
            serialized_data = self.serialize_for_disk(data)
            if self.path.parent and str(self.path.parent) != ".":
                self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.path.with_name(
                f".{self.path.name}.{uuid.uuid4().hex}.tmp"
            )
            with temporary_path.open("x", encoding="utf-8", newline="\n") as f:
                json.dump(
                    serialized_data,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            return True
        except CredentialPersistenceError:
            raise
        except Exception as e:
            _log_debug(f"Custom AI profiles save failed: {e}")
            return False
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except Exception as cleanup_error:
                    _log_debug(
                        "Custom AI profiles temporary-file cleanup failed: "
                        f"{type(cleanup_error).__name__}"
                    )

    def serialize_for_disk(self, data=None):
        data = self._snapshot_data() if data is None else data
        for profile in data.get("profiles", []):
            if (
                isinstance(profile, dict)
                and (
                    profile.get("_credential_persistence_error")
                    or (
                        str(profile.get("api_key") or "")
                        and not str(
                            profile.get("api_key_ref")
                            or profile.get("credential_ref")
                            or ""
                        ).strip()
                    )
                )
            ):
                raise CredentialPersistenceError(
                    "API key could not be stored securely"
                )
        serialized_profiles = []
        for profile in data.get("profiles", []):
            if not isinstance(profile, dict):
                continue
            serialized = {
                "id": profile.get("id"),
                "name": profile.get("name"),
                "base_url": profile.get("base_url"),
                "model": profile.get("model"),
                "enabled": bool(profile.get("enabled", True)),
                "wire_api": normalize_custom_ai_wire_api(profile.get("wire_api")),
                "structured_output_mode": normalize_custom_ai_structured_output_mode(
                    profile.get("structured_output_mode")
                ),
            }
            credential_ref = str(profile.get("api_key_ref") or profile.get("credential_ref") or "").strip()
            if credential_ref:
                serialized["api_key_ref"] = credential_ref
            serialized["reasoning_effort"] = normalize_custom_ai_reasoning_effort(
                profile.get("reasoning_effort")
                or profile.get("model_reasoning_effort")
            )
            serialized_profiles.append(serialized)
        return {
            "profiles": serialized_profiles,
            "active_translation_profile_id": data.get("active_translation_profile_id"),
            "active_ocr_profile_id": data.get("active_ocr_profile_id"),
        }

    def _credential_ref(self, profile_id):
        return f"custom-ai:{profile_id}:api_key"

    def _versioned_credential_ref(self, profile_id):
        return f"{self._credential_ref(profile_id)}:{uuid.uuid4().hex}"

    def _log_credential_issue(self, action, credential_ref, error):
        _log_debug(
            "Custom AI credential "
            f"{action} failed for {credential_ref}: {type(error).__name__}"
        )

    def _store_profile_api_key(self, profile, api_key, action):
        credential_ref = str(profile.get("api_key_ref") or profile.get("credential_ref") or "").strip()
        if not credential_ref:
            credential_ref = self._credential_ref(profile["id"])
        try:
            self.credential_store.set_secret(credential_ref, str(api_key))
            profile["api_key_ref"] = credential_ref
            profile["api_key"] = str(api_key)
            profile.pop("_api_key_plaintext_fallback", None)
            profile.pop("_credential_persistence_error", None)
            return True
        except Exception as e:
            profile.pop("_api_key_plaintext_fallback", None)
            self._log_credential_issue(action, credential_ref, e)
            raise CredentialPersistenceError(
                "API key could not be stored securely"
            ) from None

    def _resolve_profile_api_key(self, profile, credential_ref):
        try:
            api_key = self.credential_store.get_secret(credential_ref)
        except Exception as e:
            self._log_credential_issue("read", credential_ref, e)
            return ""
        if api_key is None:
            _log_debug(f"Custom AI credential read returned no key for {credential_ref}")
            return ""
        return str(api_key)

    def _delete_profile_api_key(self, profile):
        credential_ref = str(profile.get("api_key_ref") or profile.get("credential_ref") or "").strip()
        if not credential_ref and profile.get("id"):
            credential_ref = self._credential_ref(profile["id"])
        if not credential_ref:
            return
        try:
            self.credential_store.delete_secret(credential_ref)
        except Exception as e:
            self._log_credential_issue("delete", credential_ref, e)

    def _sanitize(self):
        profiles = []
        seen_ids = set()
        should_save = False
        credential_migration_failed = False
        for profile in self.data.get("profiles", []):
            if not isinstance(profile, dict):
                continue
            profile_id = str(profile.get("id") or uuid.uuid4())
            if profile_id in seen_ids:
                profile_id = str(uuid.uuid4())
            seen_ids.add(profile_id)
            sanitized = {
                "id": profile_id,
                "name": str(profile.get("name") or "Custom AI").strip() or "Custom AI",
                "base_url": str(profile.get("base_url") or "").strip(),
                "model": str(profile.get("model") or "").strip(),
                "enabled": bool(profile.get("enabled", True)),
                "wire_api": normalize_custom_ai_wire_api(profile.get("wire_api")),
                "structured_output_mode": normalize_custom_ai_structured_output_mode(
                    profile.get("structured_output_mode")
                ),
            }
            plaintext_key = str(profile.get("api_key") or "")
            credential_ref = str(profile.get("api_key_ref") or profile.get("credential_ref") or "").strip()
            profile_should_save = False
            profile_credential_migration_failed = False
            if plaintext_key:
                if not credential_ref:
                    credential_ref = self._credential_ref(profile_id)
                sanitized["api_key_ref"] = credential_ref
                try:
                    if self._store_profile_api_key(
                        sanitized,
                        plaintext_key,
                        "migration",
                    ):
                        profile_should_save = True
                except CredentialPersistenceError:
                    sanitized["api_key"] = plaintext_key
                    sanitized["api_key_ref"] = credential_ref
                    sanitized["_credential_persistence_error"] = True
                    profile_credential_migration_failed = True
                    credential_migration_failed = True
            elif credential_ref:
                sanitized["api_key_ref"] = credential_ref
                sanitized["api_key"] = self._resolve_profile_api_key(sanitized, credential_ref)
            else:
                sanitized["api_key"] = ""
            reasoning_source = profile.get("reasoning_effort") or profile.get(
                "model_reasoning_effort"
            )
            reasoning_effort = normalize_custom_ai_reasoning_effort(
                reasoning_source
            )
            sanitized["reasoning_effort"] = reasoning_effort
            if reasoning_source != reasoning_effort:
                profile_should_save = True
            if profile_should_save and not profile_credential_migration_failed:
                should_save = True
            profiles.append(sanitized)
        self.data["profiles"] = profiles
        self._repair_active_ids()
        return should_save and not credential_migration_failed

    def _first_available_profile_id(self, data=None):
        data = self.data if data is None else data
        enabled = next((p for p in data.get("profiles", []) if p.get("enabled", True)), None)
        if enabled:
            return enabled["id"]
        first = next(iter(data.get("profiles", [])), None)
        return first["id"] if first else None

    def _repair_active_ids(self):
        fallback_id = self._first_available_profile_id()
        for kind in ACTIVE_PROFILE_KINDS:
            active_key = self._active_key(kind)
            active_id = self.data.get(active_key)
            if active_id and not self.get_profile(active_id):
                self.data[active_key] = fallback_id

    def _active_key(self, kind):
        self._validate_kind(kind)
        return f"active_{kind}_profile_id"

    def _validate_kind(self, kind):
        if kind not in ACTIVE_PROFILE_KINDS:
            raise ValueError(f"Invalid active profile kind: {kind}")

    def list_profiles(self, kind=None, enabled_only=False):
        with self._data_lock:
            profiles = [
                dict(profile)
                for profile in self.data.get("profiles", [])
            ]
        if kind is not None:
            self._validate_kind(kind)
        if enabled_only:
            profiles = [p for p in profiles if p.get("enabled", True)]
        return profiles

    def get_profile(self, profile_id):
        with self._data_lock:
            for profile in self.data.get("profiles", []):
                if profile.get("id") == profile_id:
                    return dict(profile)
        return None

    def get_active_profile(self, kind):
        with self._data_lock:
            active_id = self.data.get(self._active_key(kind))
            if not active_id:
                return None
            for profile in self.data.get("profiles", []):
                if profile.get("id") == active_id:
                    return dict(profile)
            return None

    def set_active_profile(self, kind, profile_id):
        self._validate_kind(kind)
        with self._transaction_lock:
            staged_data = self._snapshot_data()
            profile = next(
                (
                    item
                    for item in staged_data.get("profiles", [])
                    if item.get("id") == profile_id
                ),
                None,
            )
            if not profile:
                raise ValueError(f"No profile with id {profile_id}")
            staged_data[self._active_key(kind)] = profile_id
            if not self._save_staged_data(staged_data):
                raise RuntimeError("Failed to persist active Custom AI profile")
            self._publish_data(staged_data)
            return self.get_profile(profile_id)

    def add_profile(
        self,
        name,
        base_url,
        api_key,
        model,
        enabled=True,
        kind=None,
        wire_api=CUSTOM_AI_WIRE_API_CHAT_COMPLETIONS,
        reasoning_effort=CUSTOM_AI_REASONING_EFFORT_LOW,
        structured_output_mode=CUSTOM_AI_STRUCTURED_OUTPUT_AUTO,
    ):
        if kind is not None:
            self._validate_kind(kind)
        if not str(api_key or ""):
            raise ValueError("API key is required")
        profile = {
            "id": str(uuid.uuid4()),
            "name": str(name).strip(),
            "base_url": str(base_url).strip(),
            "api_key": str(api_key),
            "model": str(model).strip(),
            "enabled": bool(enabled),
            "wire_api": normalize_custom_ai_wire_api(wire_api),
            "structured_output_mode": normalize_custom_ai_structured_output_mode(
                structured_output_mode
            ),
        }
        profile["reasoning_effort"] = normalize_custom_ai_reasoning_effort(
            reasoning_effort
        )
        self._validate_profile(profile)
        with self._transaction_lock:
            staged_data = self._snapshot_data()
            profile["api_key_ref"] = self._credential_ref(profile["id"])
            try:
                self._store_profile_api_key(profile, str(api_key), "write")
            except CredentialPersistenceError:
                self._delete_profile_api_key(profile)
                raise
            try:
                staged_data["profiles"].append(profile)
                for active_kind in ACTIVE_PROFILE_KINDS:
                    active_key = self._active_key(active_kind)
                    if not staged_data.get(active_key):
                        staged_data[active_key] = profile["id"]
                if not self._save_staged_data(staged_data):
                    raise RuntimeError(
                        "Failed to persist new Custom AI profile"
                    )
            except Exception:
                self._delete_profile_api_key(profile)
                raise
            self._publish_data(staged_data)
            return self.get_profile(profile["id"])

    def update_profile(self, profile_id, **updates):
        if "api_key" in updates and not str(updates.get("api_key") or ""):
            raise ValueError("API key is required")
        with self._transaction_lock:
            staged_data = self._snapshot_data()
            staged_profile = next(
                (
                    item
                    for item in staged_data.get("profiles", [])
                    if item.get("id") == profile_id
                ),
                None,
            )
            if not staged_profile:
                raise ValueError(f"No profile with id {profile_id}")
            previous_ref = str(
                staged_profile.get("api_key_ref")
                or staged_profile.get("credential_ref")
                or ""
            ).strip()
            for key in [
                "name",
                "base_url",
                "api_key",
                "model",
                "enabled",
                "wire_api",
                "reasoning_effort",
                "model_reasoning_effort",
                "structured_output_mode",
            ]:
                if key in updates:
                    if key == "model_reasoning_effort":
                        staged_profile["reasoning_effort"] = updates[key]
                    else:
                        staged_profile[key] = updates[key]
            staged_profile["name"] = str(
                staged_profile.get("name") or ""
            ).strip()
            staged_profile["base_url"] = str(
                staged_profile.get("base_url") or ""
            ).strip()
            staged_profile["api_key"] = str(
                staged_profile.get("api_key") or ""
            )
            staged_profile["model"] = str(
                staged_profile.get("model") or ""
            ).strip()
            staged_profile["enabled"] = bool(
                staged_profile.get("enabled", True)
            )
            staged_profile["wire_api"] = normalize_custom_ai_wire_api(
                staged_profile.get("wire_api")
            )
            staged_profile["structured_output_mode"] = normalize_custom_ai_structured_output_mode(
                staged_profile.get("structured_output_mode")
            )
            staged_profile["reasoning_effort"] = normalize_custom_ai_reasoning_effort(
                staged_profile.get("reasoning_effort")
            )
            self._validate_profile(staged_profile)

            staged_new_credential = "api_key" in updates
            if staged_new_credential:
                staged_profile["api_key_ref"] = self._versioned_credential_ref(
                    profile_id
                )
                staged_profile.pop("credential_ref", None)
                try:
                    self._store_profile_api_key(
                        staged_profile,
                        str(updates.get("api_key") or ""),
                        "write",
                    )
                except CredentialPersistenceError:
                    self._delete_profile_api_key(staged_profile)
                    raise

            try:
                if not self._save_staged_data(staged_data):
                    raise RuntimeError(
                        "Failed to persist Custom AI profile update"
                    )
            except Exception:
                if staged_new_credential:
                    self._delete_profile_api_key(staged_profile)
                raise

            self._publish_data(staged_data)
            if staged_new_credential and previous_ref:
                self._delete_profile_api_key({"api_key_ref": previous_ref})
            return self.get_profile(profile_id)

    def delete_profile(self, profile_id):
        with self._transaction_lock:
            staged_data = self._snapshot_data()
            removed = None
            remaining = []
            for profile in staged_data.get("profiles", []):
                if profile.get("id") == profile_id:
                    removed = profile
                else:
                    remaining.append(profile)
            if not removed:
                return False
            staged_data["profiles"] = remaining
            replacement_id = self._first_available_profile_id(staged_data)
            for kind in ACTIVE_PROFILE_KINDS:
                active_key = self._active_key(kind)
                if staged_data.get(active_key) == profile_id:
                    staged_data[active_key] = replacement_id
            if not self._save_staged_data(staged_data):
                raise RuntimeError("Failed to persist Custom AI profile deletion")
            self._publish_data(staged_data)
            self._delete_profile_api_key(removed)
            return True

    def _validate_profile(self, profile):
        if not profile.get("name"):
            raise ValueError("Profile name is required")
        if not profile.get("base_url"):
            raise ValueError("API URL is required")
        if not profile.get("api_key") and not profile.get("api_key_ref"):
            raise ValueError("API key is required")
        if not profile.get("model"):
            raise ValueError("Model name is required")
