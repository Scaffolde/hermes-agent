from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_bridge_token_validation_fails_closed() -> None:
    module_url = (
        Path(__file__).parents[2]
        / "scripts"
        / "whatsapp-bridge"
        / "bridge_http_security.js"
    ).resolve().as_uri()
    script = f"""
import {{ hasValidBridgeToken }} from {json.dumps(module_url)};
console.log(JSON.stringify([
  hasValidBridgeToken('', ''),
  hasValidBridgeToken('secret-value', ''),
  hasValidBridgeToken('secret-value', 'Bearer wrong-value'),
  hasValidBridgeToken('secret-value', 'Bearer secret-value'),
]));
"""

    result = subprocess.run(
        [shutil.which("node") or "node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert json.loads(result.stdout) == [False, False, False, True]
