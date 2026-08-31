"""Persistent Custom AI profile management."""

import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

from credential_store import (
    CredentialStoreError,
    create_default_credential_store as _base_credential_store_factory,
)
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


class CustomAIProfileManager:
    """Persist and manage user-defined OpenAI-compatible AI endpoint profiles."""

    def __init__(
        self,
        path="custom_ai_profiles.json",
        credential_store=None,
        *,
        base_dir=None,
    ):
        profile_path = Path(path)
        if base_dir is not None and not profile_path.is_absolute():
            profile_path = Path(base_dir) / profile_path
        self.path = profile_path
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
        except Exception as error:
            _log_debug(
                "Custom AI profiles staged save failed: "
                f"{type(error).__name__}"
            )
            return False

    def save(self, data=None):
        temporary_path = None
        try:
            data_to_save = self._snapshot_data() if data is None else data
            if self._has_unmigrated_plaintext_api_key(data_to_save):
                _log_debug(
                    "Custom AI profiles save skipped because a legacy plaintext "
                    "credential has not migrated to the credential store"
                )
                return False
            if self.path.parent and str(self.path.parent) != ".":
                self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.path.with_name(
                f".{self.path.name}.{uuid.uuid4().hex}.tmp"
            )
            with temporary_path.open("x", encoding="utf-8", newline="\n") as f:
                json.dump(
                    self.serialize_for_disk(data_to_save),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            return True
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

    @staticmethod
    def _has_unmigrated_plaintext_api_key(data):
        """Prevent writes that would preserve a credential-store migration failure."""
        if not isinstance(data, dict):
            return False
        for profile in data.get("profiles", []):
            if not isinstance(profile, dict):
                continue
            credential_ref = str(
                profile.get("api_key_ref") or profile.get("credential_ref") or ""
            ).strip()
            if str(profile.get("api_key") or "") and not credential_ref:
                return True
        return False

    def serialize_for_disk(self, data=None):
        data = self._snapshot_data() if data is None else data
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
                "translation_failover_enabled": bool(
                    profile.get("translation_failover_enabled", False)
                ),
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

    def _credential_failure_message(self, action, profile_id):
        safe_id = str(profile_id or "unknown").strip() or "unknown"
        return (
            f"Failed to {action} Custom AI credential for profile {safe_id}; "
            "profile changes were not saved"
        )

    def _store_profile_api_key(self, profile, api_key, action):
        """Store a secret in the credential backend. Raises on failure (no plaintext fallback)."""
        credential_ref = str(profile.get("api_key_ref") or profile.get("credential_ref") or "").strip()
        if not credential_ref:
            credential_ref = self._credential_ref(profile["id"])
        try:
            self.credential_store.set_secret(credential_ref, str(api_key))
        except Exception as e:
            self._log_credential_issue(action, credential_ref, e)
            raise CredentialStoreError(
                self._credential_failure_message(action, profile.get("id"))
            ) from e
        profile["api_key_ref"] = credential_ref
        profile["api_key"] = str(api_key)
        profile.pop("_api_key_plaintext_fallback", None)
        profile.pop("credential_ref", None)
        return True

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
        for profile in self.data.get("profiles", []):
            if not isinstance(profile, dict):
                continue
            profile_id = str(profile.get("id") or uuid.uuid4())
            if profile_id in seen_ids:
                profile_id = str(uuid.uuid4())
            seen_ids.add(profile_id)
            translation_failover_enabled = bool(
                profile.get("translation_failover_enabled", False)
            )
            sanitized = {
                "id": profile_id,
                "name": str(profile.get("name") or "Custom AI").strip() or "Custom AI",
                "base_url": str(profile.get("base_url") or "").strip(),
                "model": str(profile.get("model") or "").strip(),
                "enabled": bool(profile.get("enabled", True)),
                "translation_failover_enabled": translation_failover_enabled,
                "wire_api": normalize_custom_ai_wire_api(profile.get("wire_api")),
                "structured_output_mode": normalize_custom_ai_structured_output_mode(
                    profile.get("structured_output_mode")
                ),
            }
            if (
                "translation_failover_enabled" not in profile
                or profile.get("translation_failover_enabled")
                is not translation_failover_enabled
            ):
                should_save = True
            plaintext_key = str(profile.get("api_key") or "")
            credential_ref = str(profile.get("api_key_ref") or profile.get("credential_ref") or "").strip()
            if plaintext_key:
                # Migrate plaintext off disk only when the credential store accepts it.
                # On failure keep runtime key in memory and leave disk untouched (no
                # plaintext fallback write introduced by this process).
                if credential_ref:
                    sanitized["api_key_ref"] = credential_ref
                try:
                    self._store_profile_api_key(sanitized, plaintext_key, "migration")
                    should_save = True
                except CredentialStoreError:
                    sanitized.pop("_api_key_plaintext_fallback", None)
                    sanitized["api_key"] = plaintext_key
                    if credential_ref:
                        sanitized["api_key_ref"] = credential_ref
                    else:
                        sanitized.pop("api_key_ref", None)
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
                should_save = True
            profiles.append(sanitized)
        self.data["profiles"] = profiles
        self._repair_active_ids()
        return should_save

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
        translation_failover_enabled=False,
    ):
        if kind is not None:
            self._validate_kind(kind)
        api_key = str(api_key or "")
        profile = {
            "id": str(uuid.uuid4()),
            "name": str(name).strip(),
            "base_url": str(base_url).strip(),
            "api_key": api_key,
            "model": str(model).strip(),
            "enabled": bool(enabled),
            "translation_failover_enabled": bool(translation_failover_enabled),
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
            if api_key:
                try:
                    self._store_profile_api_key(profile, api_key, "write")
                except CredentialStoreError as error:
                    raise CredentialStoreError(
                        self._credential_failure_message("write", profile.get("id"))
                    ) from error
            staged_data["profiles"].append(profile)
            for active_kind in ACTIVE_PROFILE_KINDS:
                active_key = self._active_key(active_kind)
                if not staged_data.get(active_key):
                    staged_data[active_key] = profile["id"]
            if not self._save_staged_data(staged_data):
                self._delete_profile_api_key(profile)
                raise RuntimeError("Failed to persist new Custom AI profile")
            self._publish_data(staged_data)
            return self.get_profile(profile["id"])

    def update_profile(self, profile_id, **updates):
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
                "translation_failover_enabled",
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
            staged_profile["translation_failover_enabled"] = bool(
                staged_profile.get("translation_failover_enabled", False)
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

            api_key_updated = "api_key" in updates
            staged_new_credential = False
            if api_key_updated:
                new_api_key = str(updates.get("api_key") or "")
                staged_profile.pop("credential_ref", None)
                if new_api_key:
                    staged_profile["api_key_ref"] = self._versioned_credential_ref(
                        profile_id
                    )
                    try:
                        self._store_profile_api_key(
                            staged_profile,
                            new_api_key,
                            "write",
                        )
                    except CredentialStoreError as error:
                        # Staged snapshot is discarded; published memory/disk stay intact.
                        raise CredentialStoreError(
                            self._credential_failure_message("write", profile_id)
                        ) from error
                    staged_new_credential = True
                else:
                    staged_profile.pop("api_key_ref", None)
                    staged_profile["api_key"] = ""

            if not self._save_staged_data(staged_data):
                if staged_new_credential:
                    self._delete_profile_api_key(staged_profile)
                raise RuntimeError("Failed to persist Custom AI profile update")

            self._publish_data(staged_data)
            if api_key_updated and previous_ref:
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
        if not profile.get("model"):
            raise ValueError("Model name is required")
