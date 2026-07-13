"""Regression coverage for user-owned prompt and configuration file persistence."""

import configparser
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import app_configuration
import config_manager


class AtomicCustomPromptPersistenceTests(unittest.TestCase):
    def _app_for(self, prompt_path, current_text="previous prompt"):
        return types.SimpleNamespace(
            custom_prompt_file=str(prompt_path),
            custom_prompt_text=current_text,
        )

    def _temporary_prompt_files(self, prompt_path):
        return list(prompt_path.parent.glob(f".{prompt_path.name}.*.tmp"))

    def test_save_keeps_existing_prompt_and_memory_when_temp_write_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            prompt_path = Path(temporary_directory) / "custom_prompt.txt"
            original_bytes = "previous prompt".encode("utf-8-sig")
            prompt_path.write_bytes(original_bytes)
            app = self._app_for(prompt_path)

            with patch("builtins.open", side_effect=OSError("write unavailable")):
                saved = app_configuration.AppConfigurationMixin.save_custom_prompt(
                    app,
                    "replacement prompt",
                )

            self.assertFalse(saved)
            self.assertEqual(prompt_path.read_bytes(), original_bytes)
            self.assertEqual(app.custom_prompt_text, "previous prompt")
            self.assertEqual(self._temporary_prompt_files(prompt_path), [])

    def test_save_keeps_existing_prompt_and_memory_when_publish_replace_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            prompt_path = Path(temporary_directory) / "custom_prompt.txt"
            original_bytes = "previous prompt".encode("utf-8-sig")
            prompt_path.write_bytes(original_bytes)
            app = self._app_for(prompt_path)

            with patch(
                "app_configuration.os.replace",
                side_effect=OSError("publish unavailable"),
            ):
                saved = app_configuration.AppConfigurationMixin.save_custom_prompt(
                    app,
                    "replacement prompt",
                )

            self.assertFalse(saved)
            self.assertEqual(prompt_path.read_bytes(), original_bytes)
            self.assertEqual(app.custom_prompt_text, "previous prompt")
            self.assertEqual(self._temporary_prompt_files(prompt_path), [])

    def test_save_publishes_utf8_sig_prompt_without_temporary_residue(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            prompt_path = Path(temporary_directory) / "custom_prompt.txt"
            app = self._app_for(prompt_path)
            replacement_prompt = "新的提示词"

            saved = app_configuration.AppConfigurationMixin.save_custom_prompt(
                app,
                replacement_prompt,
            )

            self.assertTrue(saved)
            self.assertEqual(app.custom_prompt_text, replacement_prompt)
            self.assertEqual(prompt_path.read_text(encoding="utf-8-sig"), replacement_prompt)
            self.assertTrue(prompt_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertEqual(self._temporary_prompt_files(prompt_path), [])

    def test_load_empty_prompt_publishes_default_with_utf8_sig(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            prompt_path = Path(temporary_directory) / "custom_prompt.txt"
            prompt_path.write_text(" \n", encoding="utf-8-sig")
            app = self._app_for(prompt_path)

            app_configuration.AppConfigurationMixin.load_custom_prompt(app)

            self.assertEqual(app.custom_prompt_text, app_configuration.DEFAULT_CUSTOM_PROMPT)
            self.assertEqual(
                prompt_path.read_text(encoding="utf-8-sig"),
                app_configuration.DEFAULT_CUSTOM_PROMPT,
            )
            self.assertTrue(prompt_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertEqual(self._temporary_prompt_files(prompt_path), [])


class CorruptConfigurationRecoveryTests(unittest.TestCase):
    def _load_in_directory(self, directory):
        previous_directory = os.getcwd()
        os.chdir(directory)
        try:
            return config_manager.load_app_config()
        finally:
            os.chdir(previous_directory)

    def test_malformed_config_is_copied_before_atomic_default_recovery(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            config_path = directory / "ocr_translator_config.ini"
            original_bytes = b"[Settings\nprivate_setting = do-not-log-this\n"
            config_path.write_bytes(original_bytes)

            with patch.object(config_manager, "log_debug") as log_debug:
                recovered = self._load_in_directory(directory)

            backups = list(directory.glob(".corrupt-ocr_translator_config.ini-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original_bytes)
            self.assertEqual(recovered["Settings"]["translation_model"], "custom_ai")
            persisted = configparser.ConfigParser()
            persisted.read(config_path, encoding="utf-8")
            self.assertEqual(persisted["Settings"]["translation_model"], "custom_ai")
            self.assertNotEqual(config_path.read_bytes(), original_bytes)
            log_messages = [str(call.args[0]) for call in log_debug.call_args_list if call.args]
            self.assertFalse(any("do-not-log-this" in message for message in log_messages))

    def test_non_utf8_config_is_copied_before_atomic_default_recovery(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            config_path = directory / "ocr_translator_config.ini"
            original_bytes = b"[Settings]\nprivate_setting = do-not-log-this\n\xff"
            config_path.write_bytes(original_bytes)

            with patch.object(config_manager, "log_debug") as log_debug:
                recovered = self._load_in_directory(directory)

            backups = list(directory.glob(".corrupt-ocr_translator_config.ini-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original_bytes)
            self.assertEqual(recovered["Settings"]["translation_model"], "custom_ai")
            persisted = configparser.ConfigParser()
            persisted.read(config_path, encoding="utf-8")
            self.assertEqual(persisted["Settings"]["translation_model"], "custom_ai")
            self.assertNotEqual(config_path.read_bytes(), original_bytes)
            log_messages = [str(call.args[0]) for call in log_debug.call_args_list if call.args]
            self.assertFalse(any("do-not-log-this" in message for message in log_messages))

    def test_missing_config_creates_valid_default_without_corrupt_backup(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)

            loaded = self._load_in_directory(directory)

            config_path = directory / "ocr_translator_config.ini"
            persisted = configparser.ConfigParser()
            persisted.read(config_path, encoding="utf-8")
            self.assertEqual(loaded["Settings"]["translation_model"], "custom_ai")
            self.assertEqual(persisted["Settings"]["translation_model"], "custom_ai")
            self.assertEqual(list(directory.glob(".corrupt-ocr_translator_config.ini-*")), [])


if __name__ == "__main__":
    unittest.main()
