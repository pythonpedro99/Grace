"""Background entry point: python -m gaf.launch <cfgs.pkl> <out> <workers>. Runs the pickled configs with analysis.execute, so a
notebook can start runs that go on when the kernel is busy, restarted or the browser tab is closed (analysis.launch starts this)."""
import json
import pickle
import sys
import threading
import time
from pathlib import Path

from . import analysis as A


def _keep_claimed(cfgs, out):
    """Every few seconds: a run of this launch that has not started (no progress.json, or still "queued") keeps a fresh progress.json,
    so analysis.launch from another notebook cell does not start it a second time while it waits for a worker."""
    while True:
        for c in cfgs:
            d = Path(out) / c.run_id
            pf = d / "progress.json"
            if (d / "summary.json").exists():
                continue
            try:
                if not pf.exists():
                    d.mkdir(exist_ok=True)
                    pf.write_text(json.dumps({"step": 0, "phase": "queued"}))
                elif json.loads(pf.read_text()).get("phase") == "queued":
                    pf.touch()
            except (OSError, ValueError):
                pass
        time.sleep(5)

if __name__ == "__main__":
    pkl, out, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(pkl, "rb") as f:
        cfgs = pickle.load(f)
    threading.Thread(target=_keep_claimed, args=(cfgs, out), daemon=True).start()
    A.execute(cfgs, out, workers=workers, progress=False)
    print("all done", flush=True)
