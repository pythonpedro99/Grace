"""Background entry point: python -m gaf.launch <cfgs.pkl> <out> <workers>. Runs the pickled configs with analysis.execute, so a
notebook can start runs that go on when the kernel is busy, restarted or the browser tab is closed (analysis.launch starts this)."""
import pickle
import sys

from . import analysis as A

if __name__ == "__main__":
    pkl, out, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(pkl, "rb") as f:
        cfgs = pickle.load(f)
    A.execute(cfgs, out, workers=workers, progress=False)
    print("all done", flush=True)
