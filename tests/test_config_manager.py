import configparser
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config_manager


class ConfigManagerPathAndAtomicSaveTests(unittest.TestCase):
    def _set_config_dir(self, path):
        previous = os.environ.get(config_manager.CONFIG_DIR_ENV)
        os.environ[config_manager.CONFIG_DIR_ENV] = str(path)
        return previous

    def _restore_config_dir(self, previous):
        if previous is None:
            os.environ.pop(config_manager.CONFIG_DIR_ENV, None)
        else:
            os.environ[config_manager.CONFIG_DIR_ENV] = previous

    def test_resolve_app_config_path_uses_env_override(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            previous = self._set_config_dir(tmp_dir)
            try:
                resolved = config_manager.get_app_config_path()
            finally:
                self._restore_config_dir(previous)

        self.assertEqual(resolved, Path(tmp_dir) / config_manager.CONFIG_FILENAME)

    def test_default_app_config_dir_is_stable_app_data_location(self):
        default_dir = config_manager.get_default_app_config_dir()
        self.assertEqual(default_dir.name, config_manager.CONFIG_APP_DIR_NAME)
        self.assertTrue(default_dir.is_absolute())

    def test_legacy_config_migrates_once_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            legacy = root / "legacy" / config_manager.CONFIG_FILENAME
            target = root / "appdata" / config_manager.CONFIG_FILENAME
            legacy.parent.mkdir(parents=True)
            payload = b"[Settings]\nscan_interval = 321\n"
            legacy.write_bytes(payload)

            first_migrated, first_error = config_manager.migrate_legacy_config_if_needed(
                legacy_path=legacy,
                target_path=target,
            )
            second_migrated, second_error = config_manager.migrate_legacy_config_if_needed(
                legacy_path=legacy,
                target_path=target,
            )

            self.assertTrue(first_migrated)
            self.assertIsNone(first_error)
            self.assertFalse(second_migrated)
            self.assertIsNone(second_error)
            self.assertEqual(target.read_bytes(), payload)
            # Legacy file is retained for recoverability.
            self.assertTrue(legacy.exists())
            self.assertEqual(legacy.read_bytes(), payload)

    def test_existing_target_config_is_not_overwritten_by_migration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            legacy = root / "legacy" / config_manager.CONFIG_FILENAME
            target = root / "appdata" / config_manager.CONFIG_FILENAME
            legacy.parent.mkdir(parents=True)
            target.parent.mkdir(parents=True)
            legacy.write_bytes(b"[Settings]\nscan_interval = 111\n")
            existing = b"[Settings]\nscan_interval = 999\n"
            target.write_bytes(existing)

            migrated, error = config_manager.migrate_legacy_config_if_needed(
                legacy_path=legacy,
                target_path=target,
            )

            self.assertFalse(migrated)
            self.assertIsNone(error)
            self.assertEqual(target.read_bytes(), existing)
            self.assertEqual(legacy.read_bytes(), b"[Settings]\nscan_interval = 111\n")

    def test_atomic_save_preserves_old_config_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            previous = self._set_config_dir(tmp_dir)
            try:
                config_path = Path(tmp_dir) / config_manager.CONFIG_FILENAME
                original = b"[Settings]\nscan_interval = 111\n"
                config_path.write_bytes(original)

                config = configparser.ConfigParser()
                config["Settings"] = {"scan_interval": "222"}

                with patch("config_manager.os.replace", side_effect=OSError("replace failed")):
                    saved = config_manager.save_app_config(config)

                self.assertFalse(saved)
                self.assertEqual(config_path.read_bytes(), original)
                # Only this module's temp family may exist; none should remain after failure.
                leftover_temps = list(Path(tmp_dir).glob(f".{config_manager.CONFIG_FILENAME}.*.tmp"))
                self.assertEqual(leftover_temps, [])
            finally:
                self._restore_config_dir(previous)

    def test_atomic_save_preserves_old_config_when_write_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            previous = self._set_config_dir(tmp_dir)
            try:
                config_path = Path(tmp_dir) / config_manager.CONFIG_FILENAME
                original = b"[Settings]\nscan_interval = 111\n"
                config_path.write_bytes(original)

                config = configparser.ConfigParser()
                config["Settings"] = {"scan_interval": "222"}

                real_open = Path.open

                def failing_open(self, *args, **kwargs):
                    mode = args[0] if args else kwargs.get("mode", "r")
                    if "x" in str(mode) or "w" in str(mode):
                        raise OSError("write failed")
                    return real_open(self, *args, **kwargs)

                with patch.object(Path, "open", failing_open):
                    saved = config_manager.save_app_config(config)

                self.assertFalse(saved)
                self.assertEqual(config_path.read_bytes(), original)
            finally:
                self._restore_config_dir(previous)

    def test_load_app_config_uses_override_dir_not_cwd(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            previous = self._set_config_dir(tmp_dir)
            try:
                config_path = Path(tmp_dir) / config_manager.CONFIG_FILENAME
                config_path.write_text(
                    "[Settings]\nscan_interval = 555\n",
                    encoding="utf-8",
                )
                loaded = config_manager.load_app_config()
            finally:
                self._restore_config_dir(previous)

        self.assertEqual(loaded["Settings"]["scan_interval"], "555")


if __name__ == "__main__":
    unittest.main()
