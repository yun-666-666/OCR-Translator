import ast
import os
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import logger
from handlers.translation_handler import TranslationHandler
from handlers.ui_interaction_handler import UIInteractionHandler


class LogCoalescingGateTests(unittest.TestCase):
    def test_first_event_is_visible_and_window_emits_suppressed_summary(self):
        now = [100.0]
        gate = logger._LogCoalescingGate(clock=lambda: now[0])

        self.assertEqual(
            gate.prepare("cache-miss", "first", interval_seconds=5.0),
            "first",
        )
        now[0] = 101.0
        self.assertIsNone(
            gate.prepare("cache-miss", "second", interval_seconds=5.0)
        )
        now[0] = 105.0
        self.assertEqual(
            gate.prepare("cache-miss", "current", interval_seconds=5.0),
            "current (suppressed 1 similar event since previous log)",
        )

    def test_event_keys_are_independent_and_clear_restores_first_visibility(self):
        now = [100.0]
        gate = logger._LogCoalescingGate(clock=lambda: now[0])

        self.assertEqual(gate.prepare("cache", "cache first"), "cache first")
        self.assertEqual(gate.prepare("queue", "queue first"), "queue first")
        self.assertIsNone(gate.prepare("cache", "cache hidden"))

        gate.clear()

        self.assertEqual(
            gate.prepare("cache", "cache after clear"),
            "cache after clear",
        )

    def test_concurrent_events_keep_exact_suppressed_count(self):
        now = [100.0]
        gate = logger._LogCoalescingGate(clock=lambda: now[0])
        prepared = []
        prepared_lock = threading.Lock()
        start = threading.Barrier(20)

        def submit(index):
            start.wait()
            result = gate.prepare("queue", f"event-{index}")
            with prepared_lock:
                prepared.append(result)

        threads = [
            threading.Thread(target=submit, args=(index,))
            for index in range(20)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(sum(message is not None for message in prepared), 1)
        now[0] = 105.0
        self.assertEqual(
            gate.prepare("queue", "periodic"),
            "periodic (suppressed 19 similar events since previous log)",
        )

    def test_wrapper_falls_back_to_direct_log_when_coalescing_fails(self):
        with patch.object(
            logger._debug_log_coalescer,
            "prepare",
            side_effect=RuntimeError("coalescer unavailable"),
        ):
            with patch.object(logger, "log_debug") as direct_log:
                result = logger.log_debug_coalesced(
                    ["unhashable", "event", "key"],
                    "current diagnostic remains visible",
                    interval_seconds=5.0,
                )

        self.assertTrue(result)
        direct_log.assert_called_once_with("current diagnostic remains visible")


class RuntimeTextSummaryTests(unittest.TestCase):
    def test_summary_reports_shape_without_retaining_content(self):
        summary = logger.summarize_text_for_log(
            "source-secret\nsecond line"
        )

        self.assertEqual(summary, "chars=25 lines=2")
        self.assertNotIn("source-secret", summary)
        self.assertEqual(
            logger.summarize_text_for_log(None),
            "chars=0 lines=0",
        )

    def test_summary_never_raises_for_hostile_string_conversion(self):
        class HostileValue:
            def __str__(self):
                raise RuntimeError("cannot stringify")

        self.assertEqual(
            logger.summarize_text_for_log(HostileValue()),
            "chars=0 lines=0",
        )


class RuntimeContentFreeLogSourceTests(unittest.TestCase):
    def test_main_debug_log_calls_do_not_interpolate_runtime_text_directly(self):
        sensitive_names = {
            "cached_result",
            "cleaned_text_main",
            "context_string",
            "ocr_result",
            "original_text",
            "source_text",
            "target_text",
            "text_to_translate",
            "text_to_translate_dl",
            "text_to_translate_gt",
            "text_to_translate_mm",
            "translated_api_text",
            "translated_text",
            "translation_result",
        }
        marian_sensitive_names = {
            "full_text",
            "result",
            "sentence",
            "text",
            "translated",
        }
        production_files = {
            Path("handlers/cache_manager.py"): sensitive_names,
            Path("handlers/llm_provider_base.py"): sensitive_names,
            Path("handlers/translation_handler.py"): sensitive_names,
            Path("worker_threads.py"): sensitive_names,
            Path("marian_mt_translator.py"): marian_sensitive_names,
        }
        marian_translation_methods = {
            "_sequential_fallback_translate",
            "_split_into_sentences",
            "_translate_batch",
            "_translate_batch_input",
            "_translate_batch_sentences",
            "_translate_single_input",
            "_translate_text_cached",
            "translate",
        }
        violations = []

        for path, file_sensitive_names in production_files.items():
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            scan_roots = (tree,)
            if path.name == "marian_mt_translator.py":
                scan_roots = tuple(
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name in marian_translation_methods
                )
            for call in (
                node
                for scan_root in scan_roots
                for node in ast.walk(scan_root)
                if isinstance(node, ast.Call)
            ):
                function_name = getattr(call.func, "id", None)
                if function_name not in {"log_debug", "log_debug_coalesced"}:
                    continue
                for formatted in (
                    node
                    for argument in call.args
                    for node in ast.walk(argument)
                    if isinstance(node, ast.FormattedValue)
                ):
                    if (
                        isinstance(formatted.value, ast.Call)
                        and getattr(formatted.value.func, "id", None)
                        == "summarize_text_for_log"
                    ):
                        continue
                    interpolated_names = {
                        node.id
                        for node in ast.walk(formatted.value)
                        if isinstance(node, ast.Name)
                    }
                    leaked_names = sorted(
                        interpolated_names & file_sensitive_names
                    )
                    if leaked_names:
                        violations.append(
                            f"{path}:{call.lineno}: {', '.join(leaked_names)}"
                        )

        self.assertEqual(violations, [])

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

    def test_shared_tail_reader_does_not_race_with_windows_rotation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "runtime.log"
            errors = []
            producer_done = threading.Event()

            def produce():
                try:
                    for index in range(1000):
                        logger.append_rotating_text(
                            path,
                            f"line-{index}-" + ("x" * 40) + "\n",
                            max_bytes=512,
                            backup_count=2,
                        )
                except Exception as error:
                    errors.append(error)
                finally:
                    producer_done.set()

            def consume():
                while not producer_done.is_set():
                    try:
                        logger.read_shared_log_tail(
                            path,
                            max_lines=20,
                            max_bytes=512,
                            backup_count=2,
                            block_size=256,
                        )
                    except FileNotFoundError:
                        pass
                    except Exception as error:
                        errors.append(error)
                        return

            producer = threading.Thread(target=produce)
            consumer = threading.Thread(target=consume)
            producer.start()
            consumer.start()
            producer.join()
            consumer.join()
            logger.close_log_writers()

            self.assertEqual(errors, [])

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

    def test_read_log_tail_returns_only_requested_utf8_lines(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "runtime.log"
            lines = [f"行-{index}\r\n" for index in range(500)]
            path.write_text("".join(lines), encoding="utf-8-sig", newline="")

            tail = logger.read_log_tail(path, max_lines=3, block_size=64)

            self.assertEqual(tail, [line.replace("\r\n", "\n") for line in lines[-3:]])

    def test_custom_ai_short_log_uses_shared_rotating_writer(self):
        handler = TranslationHandler(object())
        profile = {"name": "Test Provider", "model": "test-model"}

        with patch(
            "handlers.translation_handler.append_rotating_text",
            create=True,
        ) as append_text:
            with patch(
                "handlers.translation_handler.is_debug_logging_enabled",
                return_value=True,
                create=True,
            ):
                handler._log_custom_short_call(
                    "translation",
                    profile,
                    "translated",
                    {"prompt_tokens": 3, "completion_tokens": 2},
                    0.25,
                )
                handler.close()

        append_text.assert_called_once()
        args, kwargs = append_text.call_args
        self.assertEqual(args[0], "CustomAI_Translation_Short_Log.txt")
        self.assertIn("SESSION 1 STARTED", args[1])
        self.assertIn("Result: chars=10 lines=1", args[1])
        self.assertNotIn("translated", args[1])
        self.assertEqual(kwargs["max_bytes"], 2 * 1024 * 1024)
        self.assertEqual(kwargs["backup_count"], 2)

    def test_custom_ai_short_log_includes_result_body_only_when_opted_in(self):
        app = types.SimpleNamespace(
            custom_ai_log_content_enabled=True,
            custom_ai_log_content_enabled_var=types.SimpleNamespace(
                get=lambda: True,
            ),
        )
        handler = TranslationHandler(app)
        profile = {"name": "Test Provider", "model": "test-model"}

        with patch(
            "handlers.translation_handler.append_rotating_text",
            create=True,
        ) as append_text:
            with patch(
                "handlers.translation_handler.is_debug_logging_enabled",
                return_value=True,
                create=True,
            ):
                handler._log_custom_short_call(
                    "translation",
                    profile,
                    "translated",
                    {"prompt_tokens": 3, "completion_tokens": 2},
                    0.25,
                )
                handler.close()

        block = append_text.call_args.args[1]
        self.assertIn(
            "Result: chars=10 lines=1\n"
            "--------------------\ntranslated\n--------------------\n",
            block,
        )

    def test_custom_ai_short_log_reads_plain_policy_snapshot_from_worker_thread(self):
        profile = {"name": "Test Provider", "model": "test-model"}

        for content_enabled in (False, True):
            with self.subTest(content_enabled=content_enabled):
                tk_getter = Mock(side_effect=AssertionError("Tk getter touched"))
                app = types.SimpleNamespace(
                    custom_ai_log_content_enabled=content_enabled,
                    custom_ai_log_content_enabled_var=types.SimpleNamespace(
                        get=tk_getter,
                    ),
                )
                handler = TranslationHandler(app)
                thread_errors = []

                def log_from_worker():
                    try:
                        handler._log_custom_short_call(
                            "translation",
                            profile,
                            "translated",
                            {"prompt_tokens": 3, "completion_tokens": 2},
                            0.25,
                        )
                    except Exception as error:
                        thread_errors.append(error)

                with patch(
                    "handlers.translation_handler.append_rotating_text",
                    create=True,
                ) as append_text:
                    with patch(
                        "handlers.translation_handler.is_debug_logging_enabled",
                        return_value=True,
                        create=True,
                    ):
                        worker = threading.Thread(target=log_from_worker)
                        worker.start()
                        worker.join()
                        handler.close()

                self.assertEqual(thread_errors, [])
                tk_getter.assert_not_called()
                append_text.assert_called_once()
                block = append_text.call_args.args[1]
                self.assertIn("Result: chars=10 lines=1", block)
                if content_enabled:
                    self.assertIn("\ntranslated\n", block)
                else:
                    self.assertNotIn("translated", block)

    def test_custom_ai_short_log_skips_writer_when_debug_logging_is_disabled(self):
        handler = TranslationHandler(object())
        profile = {"name": "Test Provider", "model": "test-model"}

        with patch(
            "handlers.translation_handler.append_rotating_text",
            create=True,
        ) as append_text:
            with patch(
                "handlers.translation_handler.is_debug_logging_enabled",
                return_value=False,
                create=True,
            ):
                handler._log_custom_short_call(
                    "translation",
                    profile,
                    "translated",
                    {"prompt_tokens": 3, "completion_tokens": 2},
                    0.25,
                )
                handler.close()

        append_text.assert_not_called()

    def test_custom_ai_short_log_updates_metrics_when_disk_logging_is_disabled(self):
        handler = TranslationHandler(object())
        profile = {"name": "Test Provider", "model": "test-model"}
        handler._record_custom_prompt_cache_usage = Mock(return_value=0.5)
        handler._record_custom_ai_latency_observation = Mock()

        with patch(
            "handlers.translation_handler.append_rotating_text",
            create=True,
        ) as append_text:
            with patch(
                "handlers.translation_handler.is_debug_logging_enabled",
                return_value=False,
                create=True,
            ):
                handler._log_custom_short_call(
                    "translation",
                    profile,
                    "translated",
                    {"prompt_tokens": 3, "completion_tokens": 2},
                    0.25,
                )
                handler.close()

        handler._record_custom_prompt_cache_usage.assert_called_once_with(
            "translation",
            {"prompt_tokens": 3, "completion_tokens": 2},
            profile=profile,
        )
        handler._record_custom_ai_latency_observation.assert_called_once_with(
            0.25,
            success=True,
            profile=profile,
        )
        append_text.assert_not_called()

    def test_custom_ai_short_log_content_setting_defaults_false_and_is_persisted(self):
        from config_manager import DEFAULT_CONFIG_SETTINGS

        self.assertEqual(
            DEFAULT_CONFIG_SETTINGS.get("custom_ai_log_content_enabled"),
            "False",
        )
        app_source = Path("app_logic.py").read_text(encoding="utf-8-sig")
        save_source = Path("handlers/ui_interaction_handler.py").read_text(
            encoding="utf-8-sig"
        )
        self.assertIn(
            "content_logging_enabled = self.config.getboolean(",
            app_source,
        )
        self.assertIn("'custom_ai_log_content_enabled'", app_source)
        self.assertIn("fallback=False", app_source)
        self.assertIn(
            "self.custom_ai_log_content_enabled = content_logging_enabled",
            app_source,
        )
        self.assertIn(
            "self.custom_ai_log_content_enabled_var = "
            "tk.BooleanVar(value=content_logging_enabled)",
            app_source,
        )
        self.assertIn(
            "cfg['custom_ai_log_content_enabled'] = "
            "str(self.app.custom_ai_log_content_enabled_var.get())",
            save_source,
        )

    def test_content_logging_checkbox_helper_updates_snapshot_then_saves_once(self):
        import gui_diagnostics_builder

        helper = getattr(
            gui_diagnostics_builder,
            "_apply_custom_ai_log_content_policy",
            None,
        )
        self.assertTrue(callable(helper))
        tk_getter = Mock(return_value=True)
        app = types.SimpleNamespace(
            custom_ai_log_content_enabled=False,
            custom_ai_log_content_enabled_var=types.SimpleNamespace(
                get=tk_getter,
            ),
            save_settings=Mock(),
        )

        helper(app)

        self.assertTrue(app.custom_ai_log_content_enabled)
        tk_getter.assert_called_once_with()
        app.save_settings.assert_called_once_with()

    def test_debug_tab_has_one_content_logging_checkbox_in_existing_layout(self):
        source = Path("gui_diagnostics_builder.py").read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        create_debug_tab = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "create_debug_tab"
        )
        checkbuttons = [
            node
            for node in ast.walk(create_debug_tab)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ttk"
            and node.func.attr == "Checkbutton"
        ]
        scrollable_tabs = [
            node
            for node in ast.walk(create_debug_tab)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "create_scrollable_tab"
        ]

        self.assertEqual(len(checkbuttons), 1)
        self.assertEqual(len(scrollable_tabs), 1)
        checkbox_source = ast.unparse(checkbuttons[0])
        self.assertIn("ttk.Checkbutton(button_frame", checkbox_source)
        self.assertIn(
            "'Write recognized/translated content to diagnostic logs'",
            checkbox_source,
        )
        self.assertIn(
            "variable=getattr(app, 'custom_ai_log_content_enabled_var', None)",
            checkbox_source,
        )
        self.assertIn(
            "command=lambda: _apply_custom_ai_log_content_policy(app)",
            checkbox_source,
        )

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

    def test_ui_refresh_debug_log_uses_tail_reader(self):
        log_text = types.SimpleNamespace(
            winfo_exists=lambda: True,
            config=Mock(),
            delete=Mock(),
            insert=Mock(),
            see=Mock(),
        )
        handler = object.__new__(UIInteractionHandler)
        handler.app = types.SimpleNamespace(log_text=log_text)

        with patch(
            "handlers.ui_interaction_handler.read_debug_log_tail",
            create=True,
            return_value=["first\n", "second\n"],
        ) as read_tail:
            handler.refresh_debug_log()

        read_tail.assert_called_once_with(max_lines=200)
        self.assertEqual(log_text.insert.call_count, 2)
        log_text.see.assert_called_once()


class RuntimeContentFreeHandlerLogTests(unittest.TestCase):
    def test_dialog_noop_does_not_write_debug_events(self):
        handler = TranslationHandler(object())

        with patch(
            "handlers.translation_handler.log_debug"
        ) as direct_log:
            with patch(
                "handlers.translation_handler.log_debug_coalesced",
                create=True,
            ) as coalesced_log:
                result = handler._format_dialog_text(
                    "plain source-secret"
                )

        self.assertEqual(result, "plain source-secret")
        direct_log.assert_not_called()
        coalesced_log.assert_not_called()

    def test_dialog_change_writes_one_content_free_coalesced_event(self):
        handler = TranslationHandler(object())

        with patch(
            "handlers.translation_handler.log_debug"
        ) as direct_log:
            with patch(
                "handlers.translation_handler.log_debug_coalesced",
                create=True,
            ) as coalesced_log:
                result = handler._format_dialog_text(
                    "- Hello. - result-secret"
                )

        self.assertEqual(result, "- Hello.\n- result-secret")
        direct_log.assert_not_called()
        coalesced_log.assert_called_once_with(
            "translation-dialog-format-applied",
            "Dialog formatting applied input chars=24 lines=1 "
            "output chars=24 lines=2",
            interval_seconds=5.0,
        )
        self.assertNotIn(
            "result-secret",
            coalesced_log.call_args.args[1],
        )


if __name__ == "__main__":
    unittest.main()
