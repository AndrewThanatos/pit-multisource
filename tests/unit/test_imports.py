import pkgutil
import subprocess
import sys

import pytest

import pit_multisource


def _modules() -> list[str]:
    found = pkgutil.walk_packages(pit_multisource.__path__, prefix="pit_multisource.")
    return sorted(m.name for m in found)


@pytest.mark.parametrize("module", _modules())
def test_every_module_imports_first_in_a_fresh_process(module: str):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
