import subprocess
import sys
import textwrap
import unittest


class StartupImportTests(unittest.TestCase):
    def test_main_import_does_not_load_removed_ocr_packages(self):
        code = textwrap.dedent(
            """
            import json
            import sys

            import main

            removed_modules = ["tess" + "erocr", "py" + "tess" + "eract"]
            print(json.dumps([name for name in removed_modules if name in sys.modules]))
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
