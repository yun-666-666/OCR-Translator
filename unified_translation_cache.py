# unified_translation_cache.py
import hashlib
import json
import os
import threading
import time
from pathlib import Path

from logger import log_debug


CACHE_SCHEMA_VERSION = 2


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
        """
        Initialize the unified translation cache.
        
        Args:
            max_size: Maximum number of entries in the LRU cache
        """
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
        self._access_times = {}  # For LRU eviction

        self._load_persisted_entries()
        
        log_debug(f"Initialized unified translation cache with max_size={max_size}")
    
    def _generate_cache_key(self, text, source_lang, target_lang, provider, **kwargs):
        """Generate a unique cache key for the translation request."""
        # Create deterministic hash of text to handle large inputs efficiently
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Include provider-specific parameters in the key
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
        # Add more provider-specific parameters as needed
        
        params_hash = hashlib.md5(params_str.encode('utf-8')).hexdigest()[:8] if params_str else ""
        
        return (text_hash, source_lang.lower(), target_lang.lower(), provider.lower(), params_hash)

    def _load_persisted_entries(self):
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            with self.persistence_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != CACHE_SCHEMA_VERSION
            ):
                with self.lock:
                    self._persistence_generation += 1
                log_debug("Ignored legacy unified cache persistence schema")
                return
            entries = payload.get("entries", []) if isinstance(payload, dict) else []
            now = time.time()

            with self.lock:
                for entry in entries:
                    key_parts = entry.get("key")
                    translation = entry.get("translation")
                    if not isinstance(key_parts, list) or translation is None:
                        continue
                    cache_key = tuple(str(part) for part in key_parts)
                    self._cache[cache_key] = translation
                    try:
                        self._access_times[cache_key] = float(entry.get("access_time", now))
                    except Exception:
                        self._access_times[cache_key] = now

                excess_entries = len(self._cache) - self.max_size
                if excess_entries > 0:
                    self._evict_lru_entries(excess_entries)

            log_debug(f"Loaded {len(self._cache)} persisted unified cache entries")
        except Exception as e:
            log_debug(f"Unified cache persistence load failed: {e}")

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
            generation = self._persistence_generation
            snapshot = self._snapshot_locked()

        succeeded = self._write_snapshot(snapshot, generation)

        with self.lock:
            if (
                succeeded
                and not self._closed
                and self._persisted_generation < self._persistence_generation
            ):
                self._schedule_persistence_locked()

    def _write_snapshot(self, snapshot, generation):
        if not self.persistence_path:
            return True

        with self._persistence_lock:
            with self.lock:
                if generation <= self._persisted_generation:
                    return True

            try:
                if self.persistence_path.parent:
                    self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

                temp_path = self.persistence_path.with_suffix(
                    self.persistence_path.suffix + ".tmp"
                )
                if not snapshot:
                    if temp_path.exists():
                        temp_path.unlink()
                    if self.persistence_path.exists():
                        self.persistence_path.unlink()
                else:
                    payload = {
                        "schema_version": CACHE_SCHEMA_VERSION,
                        "entries": snapshot,
                    }
                    with temp_path.open("w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    os.replace(temp_path, self.persistence_path)
            except Exception as e:
                log_debug(f"Unified cache persistence save failed: {e}")
                return False

            with self.lock:
                self._persisted_generation = max(
                    self._persisted_generation,
                    generation,
                )
            return True

    def flush(self):
        """Synchronously persist the latest cache snapshot."""
        with self.lock:
            self._cancel_persistence_timer_locked()
            generation = self._persistence_generation
            if generation <= self._persisted_generation:
                return True
            snapshot = self._snapshot_locked()

        return self._write_snapshot(snapshot, generation)

    def close(self):
        """Flush pending persistence and stop future timer scheduling."""
        with self.lock:
            self._closed = True
            self._cancel_persistence_timer_locked()
        return self.flush()
    
    def get(self, text, source_lang, target_lang, provider, **kwargs):
        """
        Get translation from cache. Returns None if not found.
        
        Args:
            text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code  
            provider: Translation provider ('google_api', 'deepl_api', 'marianmt')
            **kwargs: Provider-specific parameters (e.g., beam_size for MarianMT, model_type for DeepL)
            
        Returns:
            Cached translation if found, None otherwise
        """
        cache_key = self._generate_cache_key(text, source_lang, target_lang, provider, **kwargs)
        
        with self.lock:
            if cache_key in self._cache:
                # Update access time for LRU
                self._access_times[cache_key] = time.time()
                translation = self._cache[cache_key]
                
                # Enhanced debug logging for DeepL model types
                if provider.lower() == "deepl_api" and "model_type" in kwargs:
                    log_debug(f"Unified cache HIT: {provider} {source_lang}->{target_lang} (model_type={kwargs['model_type']})")
                else:
                    log_debug(f"Unified cache HIT: {provider} {source_lang}->{target_lang}")
                return translation
            
            # Enhanced debug logging for DeepL model types
            if provider.lower() == "deepl_api" and "model_type" in kwargs:
                log_debug(f"Unified cache MISS: {provider} {source_lang}->{target_lang} (model_type={kwargs['model_type']})")
            else:
                log_debug(f"Unified cache MISS: {provider} {source_lang}->{target_lang}")
            return None
    
    def store(self, text, source_lang, target_lang, provider, translation, **kwargs):
        """
        Store translation in cache.
        
        Args:
            text: Source text that was translated
            source_lang: Source language code
            target_lang: Target language code
            provider: Translation provider ('google_api', 'deepl_api', 'marianmt')
            translation: The translated text
            **kwargs: Provider-specific parameters (e.g., beam_size for MarianMT, model_type for DeepL)
        """
        cache_key = self._generate_cache_key(text, source_lang, target_lang, provider, **kwargs)
        late_snapshot = None
        late_generation = None
        
        with self.lock:
            # Evict old entries if cache is full
            if cache_key not in self._cache and len(self._cache) >= self.max_size:
                self._evict_lru_entries()
            
            # Store the translation
            self._cache[cache_key] = translation
            self._access_times[cache_key] = time.time()
            self._persistence_generation += 1
            if self._closed:
                late_generation = self._persistence_generation
                late_snapshot = self._snapshot_locked()
            else:
                self._schedule_persistence_locked()
            
            # Enhanced debug logging for DeepL model types
            if provider.lower() == "deepl_api" and "model_type" in kwargs:
                log_debug(f"Unified cache STORE: {provider} {source_lang}->{target_lang} (model_type={kwargs['model_type']})")
            else:
                log_debug(f"Unified cache STORE: {provider} {source_lang}->{target_lang}")

        if late_snapshot is not None:
            self._write_snapshot(late_snapshot, late_generation)
    
    def _evict_lru_entries(self, entries_to_evict=None):
        """Evict least recently used entries (10% of cache size)."""
        evict_count = (
            max(1, self.max_size // 10)
            if entries_to_evict is None
            else max(0, int(entries_to_evict))
        )
        evict_count = min(evict_count, len(self._cache))
        
        # Sort by access time and remove oldest
        sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
        for cache_key, _ in sorted_items[:evict_count]:
            self._cache.pop(cache_key, None)
            self._access_times.pop(cache_key, None)
        
        log_debug(f"Evicted {evict_count} LRU cache entries")
    
    def clear_all(self):
        """Clear all cached translations."""
        with self.lock:
            entries_cleared = len(self._cache)
            self._cache.clear()
            self._access_times.clear()
            self._persistence_generation += 1
            self._cancel_persistence_timer_locked()
            generation = self._persistence_generation
            snapshot = self._snapshot_locked()
        self._write_snapshot(snapshot, generation)
        log_debug(f"Cleared unified translation cache ({entries_cleared} entries)")
    
    def clear_provider(self, provider):
        """Clear cache entries for a specific provider."""
        with self.lock:
            provider_lower = provider.lower()
            keys_to_remove = [k for k in self._cache.keys() if k[3] == provider_lower]
            
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._access_times.pop(key, None)

            self._persistence_generation += 1
            self._cancel_persistence_timer_locked()
            generation = self._persistence_generation
            snapshot = self._snapshot_locked()
        self._write_snapshot(snapshot, generation)
        log_debug(f"Cleared {len(keys_to_remove)} cache entries for provider: {provider}")
    
    def get_stats(self):
        """Get cache statistics."""
        with self.lock:
            provider_counts = {}
            for key in self._cache.keys():
                provider = key[3]  # provider is at index 3
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
            
            return {
                "total_entries": len(self._cache),
                "max_size": self.max_size,
                "utilization": f"{len(self._cache) / self.max_size * 100:.1f}%",
                "provider_breakdown": provider_counts
            }
