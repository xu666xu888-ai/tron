"""No credentials, network, real gcloud calls, GPU, or paid resources required."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


DEPLOY = Path(__file__).resolve().parents[1]


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        shutil.copytree(DEPLOY, self.repo / "deploy", ignore=shutil.ignore_patterns("__pycache__"))
        (self.repo / "src").mkdir()
        (self.repo / "src" / "example.py").write_text("# synthetic fixture\n")
        (self.repo / "requirements.txt").write_text("")
        # This intentionally tracked private-looking fixture must never be archived.
        (self.repo / "tron-results").mkdir()
        (self.repo / "tron-results" / "fixture.txt").write_text("NOT_A_KEY")
        (self.repo / "deploy" / "tests" / "fake_gcloud.py").chmod(0o755)
        for args in (["init", "-q"], ["add", "."], ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"]):
            subprocess.run(["git", *args], cwd=self.repo, check=True, capture_output=True)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.bin / "gcloud").symlink_to(self.repo / "deploy" / "tests" / "fake_gcloud.py")
        (self.bin / "sleep").symlink_to("/usr/bin/true")
        self.env = {**os.environ, "PATH": str(self.bin) + os.pathsep + os.environ["PATH"], "FAKE_CLOUD_DIR": str(self.root)}

    def run_deploy(self, mode="success", arguments=None):
        self.env["FAKE_CLOUD_MODE"] = mode
        return subprocess.run(["bash", "deploy/gcloud_deploy.sh", *(arguments or ["test-project", "asia-southeast1-c", "test-tron"])],
                              cwd=self.repo, env=self.env, capture_output=True, text=True)

    def calls(self):
        path = self.root / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_success_scoped_archive_verify_then_remove_ip(self):
        result = self.run_deploy()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertIn("DEPLOYMENT_READY=", result.stdout)
        create = next(c for c in calls if c[:3] == ["compute", "instances", "create"])
        self.assertIn("--no-service-account", create)
        self.assertIn("--machine-type=g2-standard-4", create)
        self.assertEqual(calls[-1][:3], ["compute", "instances", "delete-access-config"])
        self.assertFalse(any(c[:3] == ["compute", "instances", "stop"] for c in calls))
        self.assertTrue(any("verify_gpu.py" in " ".join(c) for c in calls))
        members = json.loads((self.root / "archive-members.json").read_text())
        self.assertIn("deploy/install_app.sh", members)
        self.assertFalse(any("tron-results" in p or p.startswith(".git/") for p in members))

    def test_existing_vm_untouched(self):
        result = self.run_deploy("exists")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.calls()), 1)

    def test_permission_failures_do_not_create_vm(self):
        for mode in ("list-denied", "firewall-denied"):
            with self.subTest(mode=mode):
                result = self.run_deploy(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(c[:3] == ["compute", "instances", "create"] for c in self.calls()))

    def test_install_or_verify_failure_stops_vm_without_deleting_data(self):
        for mode in ("install-failed", "verify-failed", "ssh-failed", "ip-removal-failed", "create-ambiguous"):
            with self.subTest(mode=mode):
                result = self.run_deploy(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("DEPLOYMENT_READY=", result.stdout)
                calls = self.calls()
                self.assertEqual(calls[-1][:3], ["compute", "instances", "stop"])
                self.assertFalse(any("delete" in c for c in calls))

    def test_foreign_vm_not_stopped_after_create_conflict(self):
        result = self.run_deploy("create-foreign")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(c[:3] == ["compute", "instances", "stop"] for c in self.calls()))

    def test_reboot_branch_rechecks_driver(self):
        result = self.run_deploy("reboot")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [" ".join(c) for c in self.calls()]
        self.assertEqual(sum("--command=nvidia-smi >/dev/null" in c for c in commands), 2)
        self.assertTrue(any("shutdown -r +1" in c for c in commands))

    def test_dirty_source_refused_before_cloud_access(self):
        (self.repo / "src" / "example.py").write_text("# changed\n")
        result = self.run_deploy()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_invalid_arguments_refused(self):
        for args in (["bad;id"], ["test-project", "bad-zone"], ["test-project", "asia-southeast1-c", "bad;name"]):
            self.assertEqual(self.run_deploy(arguments=args).returncode, 2)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
