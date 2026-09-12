"""Tests use synthetic placeholders, not generated or funded wallet keys."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "secure_vanity_run.py"
spec = importlib.util.spec_from_file_location("wrapper", SCRIPT)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


class WrapperTests(unittest.TestCase):
    def test_help_without_gpu_or_dependencies(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("0600", result.stdout)

    def test_invalid_parameters_fail_before_gpu_import(self):
        for args in (["0"], ["8", "--timeout", "nan"], ["8", "--timeout", "inf"], ["8", "--max-attempts", "0"]):
            result = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_result_mode_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result.json"
            wrapper.write_private_result(output, {"fixture": "NOT_A_KEY"})
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                wrapper.write_private_result(output, {})
            self.assertEqual(json.loads(output.read_text()), {"fixture": "NOT_A_KEY"})

    def exercise(self, *, found=True, valid=True, save_failure=False):
        result = dict(found=found, privkey_hex="ab", address_hex="synthetic-hex",
                      address_base58="SYNTHETIC8", total_keys=1, elapsed=1.0,
                      mkeys_per_sec=0.001, gpu_id=0, reason="timeout")
        # One byte is deliberately not a valid private key. Crypto/GPU seams are mocked.
        addr = types.ModuleType("tron_vanity.addr")
        addr.is_valid_tron_base58 = lambda value: True
        addr.privkey_to_tron_address = lambda key: ("synthetic-hex", "SYNTHETIC8")
        turbo = types.ModuleType("tron_vanity.turbo_search")
        turbo.search_vanity_turbo = lambda **kwargs: result
        keys = types.ModuleType("tronpy.keys")
        keys.PrivateKey = lambda key: types.SimpleNamespace(public_key=types.SimpleNamespace(
            to_base58check_address=lambda: "SYNTHETIC8" if valid else "MISMATCH"))
        modules = {"tron_vanity": types.ModuleType("tron_vanity"), "tron_vanity.addr": addr,
                   "tron_vanity.turbo_search": turbo, "tronpy": types.ModuleType("tronpy"), "tronpy.keys": keys}
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, modules), \
                patch.object(sys, "argv", [str(SCRIPT), "8"]), patch.object(Path, "home", return_value=Path(folder)):
            output = io.StringIO()
            previous_umask = os.umask(0o077)
            try:
                with contextlib.redirect_stdout(output):
                    if save_failure:
                        with patch.object(wrapper, "write_private_result", side_effect=OSError("synthetic disk error")):
                            with self.assertRaises(OSError):
                                wrapper.main()
                    elif not valid:
                        with self.assertRaises(RuntimeError):
                            wrapper.main()
                    else:
                        code = wrapper.main()
                        self.assertEqual(code, 0 if found else 1)
                files = list((Path(folder) / "tron-results").glob("*.json"))
                if found and valid and not save_failure:
                    self.assertEqual(len(files), 1)
                    self.assertEqual(stat.S_IMODE(files[0].stat().st_mode), 0o600)
                    self.assertEqual(stat.S_IMODE(files[0].parent.stat().st_mode), 0o700)
                    record = json.loads(files[0].read_text())
                    self.assertTrue(record["verified_tronpy"])
                    self.assertIn("私鑰 (HEX)：ab", output.getvalue())
                else:
                    self.assertEqual(files, [])
                    self.assertNotIn("私鑰 (HEX)：", output.getvalue())
            finally:
                os.umask(previous_umask)

    def test_success_saves_and_displays_after_validation(self):
        self.exercise()

    def test_independent_verification_mismatch_fails_closed(self):
        self.exercise(valid=False)

    def test_timeout_does_not_display_or_save_key(self):
        self.exercise(found=False)

    def test_write_failure_does_not_claim_success(self):
        self.exercise(save_failure=True)


if __name__ == "__main__":
    unittest.main()
