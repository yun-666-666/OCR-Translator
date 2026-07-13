"""Direct regression coverage for the shared atomic file publisher."""

import tempfile
import unittest
from pathlib import Path

import atomic_file_io


class AtomicFilePublisherTests(unittest.TestCase):
    def test_writer_callback_publishes_only_after_complete_same_directory_write(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            target_path = Path(temporary_directory) / "settings.ini"
            target_path.write_text("previous", encoding="utf-8")
            writer_calls = []

            def writer(temporary_file):
                writer_calls.append(temporary_file.name)
                temporary_file.write("replacement")

            atomic_file_io.write_atomically(
                target_path,
                writer,
                mode="x",
                encoding="utf-8",
            )

            self.assertEqual(target_path.read_text(encoding="utf-8"), "replacement")
            self.assertEqual(len(writer_calls), 1)
            self.assertEqual(Path(writer_calls[0]).parent, target_path.parent)
            self.assertEqual(list(target_path.parent.glob(".settings.ini.*.tmp")), [])

    def test_replace_failure_keeps_existing_bytes_and_removes_shared_temporary_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            target_path = Path(temporary_directory) / "settings.ini"
            original_bytes = b"previous"
            target_path.write_bytes(original_bytes)

            def reject_replace(source, destination):
                self.assertEqual(Path(source).parent, target_path.parent)
                self.assertEqual(Path(destination), target_path)
                raise OSError("replacement blocked")

            with self.assertRaises(OSError):
                atomic_file_io.write_bytes_atomically(
                    target_path,
                    b"replacement",
                    replace_file=reject_replace,
                )

            self.assertEqual(target_path.read_bytes(), original_bytes)
            self.assertEqual(list(target_path.parent.glob(".settings.ini.*.tmp")), [])

    def test_cleanup_failure_does_not_mask_original_replace_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            target_path = Path(temporary_directory) / "settings.ini"
            original_bytes = b"previous"
            target_path.write_bytes(original_bytes)
            cleanup_attempts = []

            def reject_replace(source, destination):
                raise OSError("original replacement blocked")

            def reject_cleanup(temporary_path):
                cleanup_attempts.append(temporary_path)
                raise PermissionError("cleanup blocked")

            with self.assertRaisesRegex(OSError, "original replacement blocked"):
                atomic_file_io.write_bytes_atomically(
                    target_path,
                    b"replacement",
                    replace_file=reject_replace,
                    remove_file=reject_cleanup,
                )

            self.assertEqual(target_path.read_bytes(), original_bytes)
            self.assertEqual(len(cleanup_attempts), 1)


if __name__ == "__main__":
    unittest.main()
