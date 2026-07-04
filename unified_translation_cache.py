# unified_translation_cache.py
import hashlib
import heapq
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

from logger import log_debug


CACHE_SCHEMA_VERSION = 2
SQLITE_HEADER = b"SQLite format 3\x00"
SQLITE_SUFFIXES = {".sqlite", ".sqlite3", ".db"}
ACCESS_TIME_PERSIST_INTERVAL_SECONDS = 60.0


class UnifiedTranslationCache:
    """
    Unified translation cache for all translation providers.
    Replaces multiple overlapping LRU caches with a single, efficient cache system.
    Thread-safe, configurable, and integrates with existing file caches.
    """

    def __init__(
        self,
        max_size=1000,
        persistence_path=None,
        persistence_delay_seconds=0.25,
    ):
        self.max_size = max_size
        self.lock = threading.RLock()
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.persistence_delay_seconds = max(0.0, float(persistence_delay_seconds))
        self._persistence_lock = threading.Lock()
        self._persistence_timer = None
        self._persistence_generation = 0
        self._persisted_generation = 0
        self._closed = False

        # Unified cache storage
        # Key format: (text_hash, source_lang, target_lang, provider, params_hash)
        self._cache = {}
        self._access_times = {}
        self._provider_keys = {}
        self._last_access_persist_request_times = {}

        # Dirty persistence state
        self._dirty_upserts = {}
        self._dirty_deletes = set()
        self._dirty_provider_clears = set()
        self._full_resync_required = False

        self._load_persisted_entries()

        log_debug(f"Initialized unified translation cache with max_size={max_size}")

    def _generate_cache_key(self, text, source_lang, target_lang, provider, **kwargs):
        """Generate a unique cache key for the translation request."""
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

        params_str = ""
        if provider == "marianmt" and "beam_size" in kwargs:
            params_str = f"_beam{kwargs['beam_size']}"
        elif provider == "deepl_api" and "model_type" in kwargs:
            params_str = f"_model{kwargs['model_type']}"
        elif provider == "google_api" and "format" in kwargs:
            params_str = f"_fmt{kwargs['format']}"
        elif provider == "custom_ai":
            params_str = json.dumps(
                {
                    "profile_id": kwargs.get("profile_id", ""),
                    "base_url": str(kwargs.get("base_url", "")).strip().rstrip("/"),
                    "model": kwargs.get("model", ""),
                    "wire_api": str(
                        kwargs.get("wire_api") or "chat_completions"
                    ).strip().lower(),
                    "reasoning_effort": str(
                        kwargs.get("reasoning_effort") or ""
                    ).strip().lower(),
                    "custom_prompt": kwargs.get("custom_prompt", ""),
                    "keep_linebreaks": bool(kwargs.get("keep_linebreaks", False)),
                    "context": list(kwargs.get("context", ()) or ()),
                },
                ensure_ascii=False,
                sort_keys=True,
            )

        params_hash = (
            hashlib.md5(params_str.encode("utf-8")).hexdigest()[:8]
            if params_str
            else ""
        )
        return (
            text_hash,
            source_lang.lower(),
            target_lang.lower(),
            provider.lower(),
            params_hash,
        )

    def _add_provider_key_locked(self, cache_key):
        self._provider_keys.setdefault(cache_key[3], set()).add(cache_key)

    def _discard_provider_key_locked(self, cache_key):
        provider = cache_key[3]
        provider_keys = self._provider_keys.get(provider)
        if provider_keys is None:
            return
        provider_keys.discard(cache_key)
        if not provider_keys:
            self._provider_keys.pop(provider, None)

    def _remove_cache_entry_locked(self, cache_key):
        self._cache.pop(cache_key, None)
        self._access_times.pop(cache_key, None)
        self._last_access_persist_request_times.pop(cache_key, None)
        self._discard_provider_key_locked(cache_key)

    def _mark_entry_upsert_dirty_locked(self, cache_key):
        if self._full_resync_required:
            return
        self._dirty_deletes.discard(cache_key)
        self._dirty_upserts[cache_key] = (
            self._cache[cache_key],
            self._access_times.get(cache_key, time.time()),
        )

    def _mark_provider_clear_dirty_locked(self, provider, removed_keys):
        if self._full_resync_required:
            return
        provider_lower = provider.lower()
        self._dirty_provider_clears.add(provider_lower)
        for cache_key in removed_keys:
            self._dirty_upserts.pop(cache_key, None)
            self._dirty_deletes.discard(cache_key)

    def _mark_full_resync_dirty_locked(self):
        self._full_resync_required = True
        self._dirty_upserts.clear()
        self._dirty_deletes.clear()
        self._dirty_provider_clears.clear()

    def _clear_dirty_state_locked(self):
        self._full_resync_required = False
        self._dirty_upserts.clear()
        self._dirty_deletes.clear()
        self._dirty_provider_clears.clear()

    def _delete_file_if_exists_locked(self, path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _remove_persistence_storage_locked(self):
        if not self.persistence_path:
            return
        self._delete_file_if_exists_locked(self.persistence_path)
        self._delete_file_if_exists_locked(
            self.persistence_path.with_suffix(self.persistence_path.suffix + ".tmp")
        )
        self._delete_file_if_exists_locked(
            Path(f"{self.persistence_path}-journal")
        )

    def _sqlite_temp_path(self):
        return self.persistence_path.with_suffix(self.persistence_path.suffix + ".tmp")

    def _is_sqlite_file(self, path):
        try:
            with path.open("rb") as f:
                return f.read(len(SQLITE_HEADER)) == SQLITE_HEADER
        except OSError:
            return False

    def _legacy_json_candidate_paths(self):
        if not self.persistence_path:
            return []
        candidates = []
        if self.persistence_path.suffix.lower() in SQLITE_SUFFIXES:
            candidates.append(self.persistence_path.with_suffix(".json"))
        deduped = []
        for candidate in candidates:
            if candidate != self.persistence_path and candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _load_legacy_json_payload(self, path):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            log_debug(f"Unified cache legacy JSON load failed from {path}: {e}")
            return None

        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != CACHE_SCHEMA_VERSION
            or not isinstance(payload.get("entries"), list)
        ):
            log_debug(f"Ignored invalid legacy unified cache payload at {path}")
            return None

        return payload

    def _restore_entries_locked(self, entries):
        now = time.time()
        self._cache.clear()
        self._access_times.clear()
        self._provider_keys.clear()
        self._last_access_persist_request_times.clear()

        for entry in entries:
            key_parts = entry.get("key")
            translation = entry.get("translation")
            if not isinstance(key_parts, list) or len(key_parts) != 5 or translation is None:
                continue
            cache_key = tuple(str(part) for part in key_parts)
            self._cache[cache_key] = translation
            self._add_provider_key_locked(cache_key)
            try:
                access_time = float(entry.get("access_time", now))
            except Exception:
                access_time = now
            self._access_times[cache_key] = access_time
            self._last_access_persist_request_times[cache_key] = access_time

        excess_entries = len(self._cache) - self.max_size
        if excess_entries > 0:
            self._evict_lru_entries(excess_entries)

    def _open_sqlite_connection(self, path, create=False):
        if create and path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(str(path), timeout=30.0)
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA temp_store = MEMORY")
        return connection

    def _initialize_sqlite_schema(self, connection):
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    text_hash TEXT NOT NULL,
                    source_lang TEXT NOT NULL,
                    target_lang TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    translation TEXT NOT NULL,
                    access_time REAL NOT NULL,
                    PRIMARY KEY (text_hash, source_lang, target_lang, provider, params_hash)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cache_entries_provider
                ON cache_entries(provider)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO cache_meta (key, value)
                VALUES ('schema_version', ?)
                """,
                (str(CACHE_SCHEMA_VERSION),),
            )

    def _snapshot_locked(self):
        now = time.time()
        return [
            {
                "key": list(cache_key),
                "translation": translation,
                "access_time": self._access_times.get(cache_key, now),
            }
            for cache_key, translation in self._cache.items()
        ]

    def _sqlite_row_from_snapshot_entry(self, entry):
        cache_key = entry["key"]
        return (
            cache_key[0],
            cache_key[1],
            cache_key[2],
            cache_key[3],
            cache_key[4],
            entry["translation"],
            float(entry["access_time"]),
        )

    def _sqlite_delete_row_from_key(self, cache_key):
        return tuple(cache_key)

    def _write_full_snapshot_sqlite_locked(self, snapshot):
        if not self.persistence_path:
            return True

        if not snapshot:
            self._remove_persistence_storage_locked()
            return True

        temp_path = self._sqlite_temp_path()
        self._delete_file_if_exists_locked(temp_path)

        connection = None
        try:
            connection = self._open_sqlite_connection(temp_path, create=True)
            self._initialize_sqlite_schema(connection)
            with connection:
                connection.executemany(
                    """
                    INSERT INTO cache_entries (
                        text_hash,
                        source_lang,
                        target_lang,
                        provider,
                        params_hash,
                        translation,
                        access_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        self._sqlite_row_from_snapshot_entry(entry)
                        for entry in snapshot
                    ],
                )
            connection.close()
            connection = None
            os.replace(temp_path, self.persistence_path)
            return True
        except Exception as e:
            log_debug(f"Unified cache SQLite full snapshot save failed: {e}")
            return False
        finally:
            if connection is not None:
                connection.close()
            self._delete_file_if_exists_locked(temp_path)

    def _apply_sqlite_delta_locked(self, operation):
        if not self.persistence_path:
            return True

        if operation["cache_size"] == 0:
            self._remove_persistence_storage_locked()
            return True

        connection = None
        try:
            connection = self._open_sqlite_connection(self.persistence_path, create=True)
            self._initialize_sqlite_schema(connection)
            with connection:
                for provider in operation["provider_clears"]:
                    connection.execute(
                        "DELETE FROM cache_entries WHERE provider = ?",
                        (provider,),
                    )
                if operation["deletes"]:
                    connection.executemany(
                        """
                        DELETE FROM cache_entries
                        WHERE text_hash = ?
                          AND source_lang = ?
                          AND target_lang = ?
                          AND provider = ?
                          AND params_hash = ?
                        """,
                        operation["deletes"],
                    )
                if operation["upserts"]:
                    connection.executemany(
                        """
                        INSERT INTO cache_entries (
                            text_hash,
                            source_lang,
                            target_lang,
                            provider,
                            params_hash,
                            translation,
                            access_time
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(text_hash, source_lang, target_lang, provider, params_hash)
                        DO UPDATE SET
                            translation = excluded.translation,
                            access_time = excluded.access_time
                        """,
                        operation["upserts"],
                    )
                connection.execute(
                    """
                    INSERT OR REPLACE INTO cache_meta (key, value)
                    VALUES ('schema_version', ?)
                    """,
                    (str(CACHE_SCHEMA_VERSION),),
                )
            return True
        except Exception as e:
            log_debug(f"Unified cache SQLite delta save failed: {e}")
            return False
        finally:
            if connection is not None:
                connection.close()

    def _capture_persistence_operation_locked(self):
        if not self.persistence_path:
            return None
        generation = self._persistence_generation
        if generation <= self._persisted_generation:
            return None

        if self._full_resync_required:
            return {
                "mode": "full",
                "generation": generation,
                "snapshot": self._snapshot_locked(),
            }

        return {
            "mode": "delta",
            "generation": generation,
            "cache_size": len(self._cache),
            "provider_clears": tuple(sorted(self._dirty_provider_clears)),
            "deletes": [
                self._sqlite_delete_row_from_key(cache_key)
                for cache_key in self._dirty_deletes
            ],
            "upserts": [
                (
                    cache_key[0],
                    cache_key[1],
                    cache_key[2],
                    cache_key[3],
                    cache_key[4],
                    translation,
                    access_time,
                )
                for cache_key, (translation, access_time)
                in self._dirty_upserts.items()
            ],
        }

    def _apply_persistence_operation(self, operation):
        if not operation:
            return True

        with self._persistence_lock:
            if operation["mode"] == "full":
                return self._write_full_snapshot_sqlite_locked(operation["snapshot"])
            return self._apply_sqlite_delta_locked(operation)

    def _finalize_persistence_success_locked(self, generation):
        self._persisted_generation = max(self._persisted_generation, generation)
        if self._persistence_generation == generation:
            self._clear_dirty_state_locked()
        else:
            self._mark_full_resync_dirty_locked()

    def _load_sqlite_entries(self, path):
        connection = None
        try:
            connection = self._open_sqlite_connection(path, create=False)
            self._initialize_sqlite_schema(connection)
            rows = connection.execute(
                """
                SELECT
                    text_hash,
                    source_lang,
                    target_lang,
                    provider,
                    params_hash,
                    translation,
                    access_time
                FROM cache_entries
                """
            ).fetchall()
        except Exception as e:
            log_debug(f"Unified cache SQLite load failed from {path}: {e}")
            return
        finally:
            if connection is not None:
                connection.close()

        trimmed_snapshot = None
        with self.lock:
            self._cache.clear()
            self._access_times.clear()
            self._provider_keys.clear()
            self._last_access_persist_request_times.clear()
            for row in rows:
                cache_key = (
                    str(row[0]),
                    str(row[1]),
                    str(row[2]),
                    str(row[3]),
                    str(row[4]),
                )
                self._cache[cache_key] = row[5]
                self._add_provider_key_locked(cache_key)
                try:
                    access_time = float(row[6])
                except Exception:
                    access_time = time.time()
                self._access_times[cache_key] = access_time
                self._last_access_persist_request_times[cache_key] = access_time

            excess_entries = len(self._cache) - self.max_size
            if excess_entries > 0:
                self._evict_lru_entries(excess_entries)
                trimmed_snapshot = self._snapshot_locked()

        if trimmed_snapshot is not None:
            self._apply_persistence_operation(
                {
                    "mode": "full",
                    "generation": 0,
                    "snapshot": trimmed_snapshot,
                }
            )

        log_debug(f"Loaded {len(self._cache)} persisted unified cache entries")

    def _migrate_legacy_json_payload(self, payload, source_path):
        with self.lock:
            self._restore_entries_locked(payload.get("entries", []))
            snapshot = self._snapshot_locked()

        succeeded = self._apply_persistence_operation(
            {
                "mode": "full",
                "generation": 0,
                "snapshot": snapshot,
            }
        )
        if not succeeded:
            return False

        if source_path != self.persistence_path:
            with self._persistence_lock:
                self._delete_file_if_exists_locked(source_path)

        log_debug(
            f"Migrated {len(self._cache)} legacy unified cache entries from {source_path}"
        )
        return True

    def _load_persisted_entries(self):
        if not self.persistence_path:
            return

        if self.persistence_path.exists():
            if self._is_sqlite_file(self.persistence_path):
                self._load_sqlite_entries(self.persistence_path)
                return

            legacy_payload = self._load_legacy_json_payload(self.persistence_path)
            if legacy_payload is not None:
                self._migrate_legacy_json_payload(
                    legacy_payload,
                    self.persistence_path,
                )
            return

        for legacy_path in self._legacy_json_candidate_paths():
            if not legacy_path.exists():
                continue
            legacy_payload = self._load_legacy_json_payload(legacy_path)
            if legacy_payload is None:
                continue
            self._migrate_legacy_json_payload(legacy_payload, legacy_path)
            return

    def _cancel_persistence_timer_locked(self):
        timer = self._persistence_timer
        self._persistence_timer = None
        if timer is not None:
            timer.cancel()

    def _schedule_persistence_locked(self):
        if (
            not self.persistence_path
            or self._closed
            or self._persistence_timer is not None
        ):
            return

        timer = threading.Timer(
            self.persistence_delay_seconds,
            self._run_scheduled_persistence,
        )
        timer.daemon = True
        self._persistence_timer = timer
        timer.start()

    def _run_scheduled_persistence(self):
        with self.lock:
            self._persistence_timer = None
            if self._closed:
                return
            operation = self._capture_persistence_operation_locked()

        succeeded = self._apply_persistence_operation(operation)

        with self.lock:
            if succeeded and operation is not None:
                self._finalize_persistence_success_locked(operation["generation"])
            if (
                succeeded
                and not self._closed
                and self._persisted_generation < self._persistence_generation
            ):
                self._schedule_persistence_locked()

    def flush(self):
        """Synchronously persist the latest cache state."""
        with self.lock:
            self._cancel_persistence_timer_locked()
            operation = self._capture_persistence_operation_locked()

        succeeded = self._apply_persistence_operation(operation)

        with self.lock:
            if succeeded and operation is not None:
                self._finalize_persistence_success_locked(operation["generation"])

        return succeeded

    def close(self):
        """Flush pending persistence and stop future timer scheduling."""
        with self.lock:
            access_updates_pending = False
            if self.persistence_path:
                for cache_key, access_time in self._access_times.items():
                    last_requested = self._last_access_persist_request_times.get(
                        cache_key,
                        0.0,
                    )
                    if access_time <= last_requested:
                        continue
                    self._mark_entry_upsert_dirty_locked(cache_key)
                    self._last_access_persist_request_times[cache_key] = access_time
                    access_updates_pending = True
            if access_updates_pending:
                self._persistence_generation += 1
            self._closed = True
            self._cancel_persistence_timer_locked()
        return self.flush()

    def get(self, text, source_lang, target_lang, provider, **kwargs):
        """
        Get translation from cache. Returns None if not found.
        """
        cache_key = self._generate_cache_key(
            text,
            source_lang,
            target_lang,
            provider,
            **kwargs,
        )

        with self.lock:
            try:
                translation = self._cache[cache_key]
            except KeyError:
                if provider.lower() == "deepl_api" and "model_type" in kwargs:
                    log_debug(
                        f"Unified cache MISS: {provider} {source_lang}->{target_lang} "
                        f"(model_type={kwargs['model_type']})"
                    )
                else:
                    log_debug(f"Unified cache MISS: {provider} {source_lang}->{target_lang}")
                return None
            else:
                access_time = time.time()
                self._access_times[cache_key] = access_time
                last_requested = self._last_access_persist_request_times.get(
                    cache_key,
                    0.0,
                )
                if (
                    self.persistence_path
                    and not self._closed
                    and access_time - last_requested
                    >= ACCESS_TIME_PERSIST_INTERVAL_SECONDS
                ):
                    self._persistence_generation += 1
                    self._mark_entry_upsert_dirty_locked(cache_key)
                    self._last_access_persist_request_times[cache_key] = access_time
                    self._schedule_persistence_locked()
                if provider.lower() == "deepl_api" and "model_type" in kwargs:
                    log_debug(
                        f"Unified cache HIT: {provider} {source_lang}->{target_lang} "
                        f"(model_type={kwargs['model_type']})"
                    )
                else:
                    log_debug(f"Unified cache HIT: {provider} {source_lang}->{target_lang}")
                return translation

    def store(self, text, source_lang, target_lang, provider, translation, **kwargs):
        """
        Store translation in cache.
        """
        cache_key = self._generate_cache_key(
            text,
            source_lang,
            target_lang,
            provider,
            **kwargs,
        )
        operation = None
        missing = object()

        with self.lock:
            existing_translation = self._cache.get(cache_key, missing)
            is_new_entry = existing_translation is missing
            translation_changed = is_new_entry or existing_translation != translation

            if is_new_entry and len(self._cache) >= self.max_size:
                self._evict_lru_entries()

            self._cache[cache_key] = translation
            if is_new_entry:
                self._add_provider_key_locked(cache_key)
            access_time = time.time()
            self._access_times[cache_key] = access_time

            if translation_changed:
                self._persistence_generation += 1
                self._mark_entry_upsert_dirty_locked(cache_key)
                self._last_access_persist_request_times[cache_key] = access_time
                if self._closed:
                    operation = self._capture_persistence_operation_locked()
                else:
                    self._schedule_persistence_locked()

            if provider.lower() == "deepl_api" and "model_type" in kwargs:
                log_debug(
                    f"Unified cache STORE: {provider} {source_lang}->{target_lang} "
                    f"(model_type={kwargs['model_type']})"
                )
            else:
                log_debug(f"Unified cache STORE: {provider} {source_lang}->{target_lang}")

        if operation is not None:
            succeeded = self._apply_persistence_operation(operation)
            with self.lock:
                if succeeded:
                    self._finalize_persistence_success_locked(operation["generation"])

    def _evict_lru_entries(self, entries_to_evict=None):
        """Evict least recently used entries (10% of cache size)."""
        evict_count = (
            max(1, self.max_size // 10)
            if entries_to_evict is None
            else max(0, int(entries_to_evict))
        )
        evict_count = min(evict_count, len(self._cache))

        oldest_items = heapq.nsmallest(
            evict_count,
            enumerate(self._access_times.items()),
            key=lambda indexed_item: (indexed_item[1][1], indexed_item[0]),
        )
        for _, (cache_key, _) in oldest_items:
            self._remove_cache_entry_locked(cache_key)
            if not self._full_resync_required and cache_key[3] not in self._dirty_provider_clears:
                self._dirty_upserts.pop(cache_key, None)
                self._dirty_deletes.add(cache_key)

        log_debug(f"Evicted {evict_count} LRU cache entries")

    def clear_all(self):
        """Clear all cached translations."""
        with self.lock:
            entries_cleared = len(self._cache)
            self._cache.clear()
            self._access_times.clear()
            self._provider_keys.clear()
            self._last_access_persist_request_times.clear()
            self._persistence_generation += 1
            self._mark_full_resync_dirty_locked()
            self._cancel_persistence_timer_locked()
            operation = self._capture_persistence_operation_locked()

        succeeded = self._apply_persistence_operation(operation)
        with self.lock:
            if succeeded and operation is not None:
                self._finalize_persistence_success_locked(operation["generation"])

        log_debug(f"Cleared unified translation cache ({entries_cleared} entries)")

    def clear_provider(self, provider):
        """Clear cache entries for a specific provider."""
        with self.lock:
            provider_lower = provider.lower()
            keys_to_remove = list(self._provider_keys.get(provider_lower, ()))

            if not keys_to_remove:
                log_debug(f"Cleared 0 cache entries for provider: {provider}")
                return

            for key in keys_to_remove:
                self._remove_cache_entry_locked(key)

            self._persistence_generation += 1
            self._mark_provider_clear_dirty_locked(provider, keys_to_remove)
            self._cancel_persistence_timer_locked()
            operation = self._capture_persistence_operation_locked()

        succeeded = self._apply_persistence_operation(operation)
        with self.lock:
            if succeeded and operation is not None:
                self._finalize_persistence_success_locked(operation["generation"])

        log_debug(f"Cleared {len(keys_to_remove)} cache entries for provider: {provider}")

    def get_stats(self):
        """Get cache statistics."""
        with self.lock:
            provider_counts = {
                provider: len(keys)
                for provider, keys in self._provider_keys.items()
            }

            return {
                "total_entries": len(self._cache),
                "max_size": self.max_size,
                "utilization": f"{len(self._cache) / self.max_size * 100:.1f}%",
                "provider_breakdown": provider_counts,
            }
