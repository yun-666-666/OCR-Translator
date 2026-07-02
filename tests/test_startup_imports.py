import subprocess
import sys
import textwrap
import unittest


class StartupImportTests(unittest.TestCase):
    def test_main_import_defers_heavy_ocr_dependencies(self):
        code = textwrap.dedent(
            """
            import json
            import sys

            import main

            heavy_modules = ["cv2", "numpy", "PIL", "tesserocr", "pyautogui", "PySide6", "pyside_overlay"]
            print(json.dumps([name for name in heavy_modules if name in sys.modules]))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
