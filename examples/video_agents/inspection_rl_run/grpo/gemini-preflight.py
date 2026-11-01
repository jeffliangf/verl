import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from examples.video_agents.judge_worker import gemini_address_family
from examples.video_agents.storage import load_external

runner = load_external(Path("/home/jeff/rl-benchmarks/grpo-group8-16h100-20261031/evaluator/scripts/run_gemini_pairwise_eval.py"))
key = os.environ.get("GEMINI_API_KEY") or runner.read_env_file(runner.DEFAULT_ENV_FILE).get("GEMINI_API_KEY", "")
assert key, "Gemini credential is missing"
request = urllib.request.Request(runner.API_BASE + "/models/gemini-3.8-flash", headers={"x-goog-api-key": key})
start = time.monotonic()
try:
    with gemini_address_family("ipv4"):
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
except urllib.error.HTTPError as error:
    status = error.code
result = {
    "host": socket.gethostname(),
    "status": status,
    "address_family": "ipv4",
    "elapsed_seconds": time.monotonic() - start,
}
print(json.dumps(result), flush=True)
path = (
    Path(os.environ["RL_RUN_DIR"])
    / f"gemini-preflight-{socket.gethostname()}.json"
)
path.write_text(json.dumps(result, indent=2) + "\n")
assert status == 200, f"Gemini model metadata request returned HTTP {status}"
