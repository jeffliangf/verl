"""Exercise the staged H3 phase budgets with one real five-second request."""
import json
from pathlib import Path

from examples.video_agents.h3 import generation_slot
from examples.video_agents.storage import read_json, write_json
from examples.video_agents.tournament import generate_video, load_protocol, runtime_protocol

root = Path(__file__).parent / "grpo"
settings = read_json(root / "training-config.json")["video_rl"]
protocol = runtime_protocol(load_protocol(root / "protocol.json"), settings)
protocol["h3_phase_timeouts"] = {"queue": 600, "generation": 150, "download": 120}
directory = root / "queue-timeout-smoke"
with generation_slot(root, 20, directory / "timings.jsonl"):
    video = generate_video(
        directory,
        protocol,
        {"prompt": "A small red sailboat glides across a calm blue lake in daylight, filmed in a steady wide shot.",
         "duration": 5, "seed": 7},
    )
state = read_json(directory / "state.json")
result = {
    "status": state["status"],
    "request": state["request"],
    "provider_timings": state.get("provider_timings"),
    "phase_usage_by_job": state.get("phase_usage_by_job"),
    "bytes": state["bytes"],
    "media": state["media"],
}
write_json(directory / "result.json", result)
print(json.dumps(result), flush=True)
