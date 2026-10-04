"""Run the four original sibling-best evals after a main-controlled group exits."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "studies" / "main_historical"


def now():
    return datetime.now(timezone.utc).isoformat()


parser = argparse.ArgumentParser()
parser.add_argument("--data", choices=("MNIST", "CIFAR10"), required=True)
parser.add_argument("--model", choices=("linear", "mlp", "cnn"), required=True)
args = parser.parse_args()
destination = STUDY / "docs" / f"EARLY_EVAL_{args.data.upper()}_{args.model.upper()}_EXECUTION.json"
if destination.exists():
    raise RuntimeError("Preserve the existing execution record; do not restart this controller.")

native = runpy.run_path(str(STUDY / "run.py"), run_name="completed_group_eval_gate")
native["check_unchanged"]()
from rpipe.structure.algorithm.resume import sibling_train_ids

configuration = dict(native["configs"]())
execution = json.loads((STUDY / "docs" / "EXECUTION.json").read_text(encoding="utf-8"))
rows = []
for rid, cfg in configuration.items():
    if (cfg["algorithm"]["mode"], cfg["data"]["name"], cfg["model"]["name"]) != ("eval", args.data, args.model):
        continue
    parents = sibling_train_ids(STUDY, rid)
    if len(parents) != 1:
        raise RuntimeError(f"Expected one unambiguous sibling train: {rid}, {parents}")
    parent = parents[0]
    exits = [event for event in execution["events"] if event.get("run") == parent and event.get("event") == "exit"]
    if len(exits) != 1 or exits[0].get("exit_code") != 0 or exits[0].get("status") != "succeeded":
        raise RuntimeError(f"Main controller has not reaped a successful train: {parent}")
    if native["state"](parent) != "succeeded" or native["state"](rid) != "pending":
        raise RuntimeError(f"Not fresh pending eval with completed train: {rid}")
    rows.append((rid, parent, int(cfg["seed"])))
rows.sort(key=lambda row: row[2])
if len(rows) != 4 or [row[2] for row in rows] != [0, 1, 2, 3]:
    raise RuntimeError("The original four distinct seeds are required.")

document = {
    "started_at_utc": now(), "pid": os.getpid(), "status": "running",
    "scope": f"Original four planned {args.data}/{args.model} sibling-best eval Runs",
    "resume_policy": "Do not restart while this exact controller or a child remains alive.",
    "runs": [],
}


def save():
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, destination)


save()
print(f"completed-group eval controller pid={os.getpid()}", flush=True)
for rid, parent, seed in rows:
    native["check_unchanged"]()
    row = {"id": rid, "train": parent, "seed": seed, "start_utc": now()}
    document["runs"].append(row)
    logs = STUDY / "runs" / rid / "assets" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", str(STUDY / "run.py"), "one", rid]
    with (logs / "early_eval_launcher.log").open("a", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=stream,
                                 stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        row.update(pid=child.pid, command=command)
        save()
        code = child.wait()
    row.update(exit_code=code, status=native["state"](rid), end_utc=now())
    save()
    print(f"exit eval {rid}: {code}, {row['status']}", flush=True)
    if code != 0 or row["status"] != "succeeded":
        document.update(status="failed", finished_at_utc=now())
        save()
        raise SystemExit(1)
native["check_unchanged"]()
document.update(status="succeeded", finished_at_utc=now())
save()
print("All four original evals succeeded and their child processes exited.", flush=True)
