"""The long-standing check suites, run when the extracted game tables are present.

`tools/validate_sim.py` and `tools/validate_rl.py` read data/ and print a PASS or
FAIL line per check. They run here as subprocesses, so a failing check fails
pytest. Together they take about two minutes; skip them with
`pytest -m "not needs_data"`. Without data/ (as in CI) they skip on their own.
"""
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.needs_data


@pytest.mark.parametrize("script", ["validate_sim.py", "validate_rl.py"])
def test_check_suite(script):
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / script)], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    failed = [line.strip() for line in proc.stdout.splitlines() if line.lstrip().startswith("FAIL")]
    assert proc.returncode == 0 and not failed, "\n".join(failed) or proc.stderr[-2000:]
