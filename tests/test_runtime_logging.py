import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import logger
from handlers.translation_handler import TranslationHandler
from handlers.ui_interaction_handler import UIInteractionHandler


class RotatingTextWriterTests(unittest.TestCase):
    def tearDown(self):
        close_writers = getattr(logger, "close_log_writers", None)
        if callable(close_writers):
            close_writers()

    def test_writer_reuses_stream_and_remains_reusable_after_clear(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "runtime.log"
            writer = logger._RotatingTextWriter(path, max_bytes=1024, backup_count=2)

            writer.write("first\n")
            first_stream = writer._stream
            writer.write("second\n")

            self.assertIs(writer._stream, first_stream)
            writer.clear("cleared\n")
            writer.write("after\n")
            writer.close()
            content = path.read_text(encoding="utf-8-sig")
            self.assertEqual(content, "cleared\nafter\n")

    def test_writer_rotation_keeps_only_configured_backups(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "runtime.log"
            writer = logger._RotatingTextWriter(path, max_bytes=90, backup_count=2)

            for index in range(8):
                writer.write(f"line-{index}-" + ("x" * 32) + "\n")
            writer.close()

            self.assertTrue(path.exists())
            self.assertTrue(Path(f"{path}.1").exists())
            self.assertTrue(Path(f"{path}.2").exists())
            self.assertFalse(Path(f"{path}.3").exists())
            for candidate in (path, Path(f"{path}.1"), Path(f"{path}.2")):
                self.assertLessEqual(candidate.stat().st_size, 93)

    def test_writer_preserves_platform_newline_format(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "runtime.log"
            writer = logger._RotatingTextWriter(path, max_bytes=1024, backup_count=1)

            writer.write("line\n")
            writer.close()

            self.assertTrue(path.read_bytes().endswith(os.linesep.encode("ascii")))

    def test_writer_preserves_every_line_from_concurrent_threads(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "runtime.log"
            writer = logger._RotatingTextWriter(
                path,
                max_bytes=1024 * 1024,
                backup_count=1,
            )
            expected = {
                f"worker-{worker_id}-line-{line_id}"
                for worker_id in range(8)
                for line_id in range(50)
            }

            def write_lines(worker_id):
                for line_id in range(50):
                    writer.write(f"worker-{worker_id}-line-{line_id}\n")

            threads = [
                threading.Thread(target=write_lines, args=(worker_id,))
                for worker_id in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            writer.close()

            actual = set(path.read_text(encoding="utf-8-sig").splitlines())
            self.assertEqual(actual, expected)

    def test_unittest_process_uses_pid_specific_temporary_log_directory(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OCR_TRANSLATOR_LOG_DIR", None)
            path = logger.resolve_runtime_log_path("translator_debug.log")

        self.assertEqual(
            path.parent,
            Path(tempfile.gettempdir()) / f"ocr-translator-tests-{os.getpid()}",
        )
        self.assertEqual(path.name, "translator_debug.log")

    def test_unittest_process_exports_log_directory_for_child_processes(self):
        expected = (
            Path(tempfile.gettempdir())
            / f"ocr-translator-tests-{os.getpid()}"
        )

        logger.ensure_test_log_environment()

        self.assertEqual(
            os.environ.get("OCR_TRANSLATOR_LOG_DIR"),
            str(expected),
        )

    def test_log_debug_keeps_timestamp_format_in_resolved_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(
                os.environ,
                {"OCR_TRANSLATOR_LOG_DIR": tmp_dir},
            ):
                logger.log_debug("hello")
                logger.close_log_writers()

            content = (Path(tmp_dir) / "translator_debug.log").read_text(
                encoding="utf-8-sig"
            )
            self.assertRegex(
                content,
                r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}: hello\n$",
            )

    def test_custom_ai_short_log_uses_shared_rotating_writer(self):
        handler = object.__new__(TranslationHandler)
        handler._custom_session_started = set()
        profile = {"name": "Test Provider", "model": "test-model"}

        with patch(
            "handlers.translation_handler.append_rotating_text",
            create=True,
        ) as append_text:
            handler._log_custom_short_call(
                "translation",
                profile,
                "translated",
                {"prompt_tokens": 3, "completion_tokens": 2},
                0.25,
            )

        append_text.assert_called_once()
        args, kwargs = append_text.call_args
        self.assertEqual(args[0], "CustomAI_Translation_Short_Log.txt")
        self.assertIn("SESSION 1 STARTED", args[1])
        self.assertIn("translated", args[1])
        self.assertEqual(kwargs["max_bytes"], 2 * 1024 * 1024)
        self.assertEqual(kwargs["backup_count"], 2)

    def test_ui_clear_debug_log_uses_shared_writer(self):
        handler = object.__new__(UIInteractionHandler)
        handler.app = object()
        handler.refresh_debug_log = Mock()

        with patch(
            "handlers.ui_interaction_handler.clear_runtime_debug_log",
            create=True,
        ) as clear_log:
            handler.clear_debug_log()

        clear_log.assert_called_once_with()
        handler.refresh_debug_log.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
