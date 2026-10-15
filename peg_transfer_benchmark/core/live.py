"""Live run state for the dashboard (scripts/servers/dashboard_server.py): an append-only events.jsonl, the latest
camera frames as JPEGs, and the STOP file the dashboard (or `touch $RUN_DIR/STOP`) creates to end the run."""
import json
import os
import time

import numpy as np
from PIL import Image


class StopRun(Exception):
    pass


class Live:
    def __init__(self, run_dir):
        self.dir = run_dir
        self.stop_path = os.path.join(run_dir, "STOP")
        self.f = open(os.path.join(run_dir, "events.jsonl"), "a")
        self.t0 = time.time()
        self.stopping = False          # set while shutting down, so the clean-up moves are not interrupted
        self._last_frame = 0.0
        self._last_tip = None
        self._n = 0

    def emit(self, type, **kw):
        kw = dict(type=type, t=round(time.time() - self.t0, 2), **kw)
        self.f.write(json.dumps(kw, default=_jsonable) + "\n")
        self.f.flush()

    def stop_requested(self):
        return os.path.exists(self.stop_path)

    def check_stop(self):
        if not self.stopping and self.stop_requested():
            reason = "STOP pressed"
            try:
                reason = open(self.stop_path).read().strip() or reason
            except OSError:
                pass
            raise StopRun(reason)

    def frame(self, name, img, size=None, quality=82):
        im = Image.fromarray(np.ascontiguousarray(img))
        if size:
            im = im.resize(size, Image.BILINEAR)
        path = os.path.join(self.dir, f"live_{name}.jpg")
        im.save(path + ".tmp", format="JPEG", quality=quality)
        os.replace(path + ".tmp", path)  # atomic: the server never serves half a file

    def tick(self, state, frames, every=0.3):
        """Called after every sim step: checks for STOP each step; every `every` s publishes `state()` (a dict, as a
        "hands" event when it changes) and the camera frames (frames(): {name: (image, size)})."""
        self._n += 1
        self.check_stop()
        now = time.time()
        if now - self._last_frame < every:
            return
        self._last_frame = now
        st = state()
        if st != self._last_tip:
            self._last_tip = st
            self.emit("hands", **st)
        for name, (img, size) in frames().items():
            self.frame(name, img, size)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
