"""Live dashboard for a skin-marking run (stdlib only; runs on the compute node beside the sim).

Serves dashboard/index.html in live mode, streams $RUN_DIR/events.jsonl as server-sent events, serves the latest
camera frames, and on STOP writes $RUN_DIR/STOP (the run lifts the marker, saves and exits). If the run has not
confirmed within --grace seconds, it cancels the Slurm job. Every request needs the access token (printed to
dashboard.txt), so other users on the cluster can't open the page or press STOP.

  python3 dashboard_server.py --run_dir $RUN_DIR --port 8765"""
import argparse
import http.server
import json
import os
import secrets
import socket
import subprocess
import threading
import time
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "..", "..", "dashboard", "index.html")

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--grace", type=float, default=45, help="seconds to wait for a clean stop before scancel")
ap.add_argument("--ledger", default=os.path.join(os.environ.get("P", os.path.expanduser("~/isaac-bench")), "results", "openai_spend.json"),
                help="this provider's spend ledger; passing --cap is flagged on the dashboard")
ap.add_argument("--cap", type=float, default=3.00)
args = ap.parse_args()
os.makedirs(args.run_dir, exist_ok=True)
TOKEN = secrets.token_urlsafe(16)
EVENTS = os.path.join(args.run_dir, "events.jsonl")
STOP = os.path.join(args.run_dir, "STOP")
JOB = os.environ.get("SLURM_JOB_ID")


def run_ended():
    try:
        with open(EVENTS) as f:
            for line in f:
                ev = json.loads(line)
                if ev.get("type") == "status" and ev.get("state") in ("done", "stopped"):
                    return True
    except (OSError, ValueError):
        pass
    return False


def request_stop(reason):
    """Ask the run to stop (it lifts the arms, saves and exits); cancel the job if it doesn't within --grace."""
    if not os.path.exists(STOP):
        with open(STOP, "w") as f:
            f.write(reason)
        threading.Thread(target=watchdog, daemon=True).start()


def budget_guard():
    """Independent of the run: when total spend passes the cap, flag it once on the dashboard. The run is not stopped
    (the user decides, with STOP)."""
    while True:
        try:
            spent = float(json.load(open(args.ledger))["spent"])
        except (OSError, ValueError, KeyError):
            spent = 0.0
        if spent >= args.cap and not run_ended():
            with open(EVENTS, "a") as f:
                f.write(json.dumps(dict(type="budget", text=f"Over budget: ${spent:.4f} spent of ${args.cap:.2f}. "
                                                            "The run continues; press STOP to end it.")) + "\n")
            return
        time.sleep(2)


def watchdog():
    """After STOP: give the run `grace` seconds to stop cleanly, then cancel the job."""
    t0 = time.time()
    while time.time() - t0 < args.grace:
        if run_ended():
            return
        time.sleep(1)
    with open(EVENTS, "a") as f:
        f.write(json.dumps(dict(type="stopped", text=f"No clean stop after {args.grace:.0f} s: cancelling Slurm job {JOB}")) + "\n")
        f.write(json.dumps(dict(type="status", state="stopped")) + "\n")
    if JOB:
        subprocess.run(["scancel", JOB])


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def authed(self):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        cookie = self.headers.get("Cookie", "")
        return q.get("k", [""])[0] == TOKEN or f"k={TOKEN}" in cookie

    def send(self, code, body=b"", ctype="text/plain", extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        for k, v in extra:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.authed():
            return self.send(403, b"Open the link from dashboard.txt (it carries the access token).")
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            html = open(HTML, encoding="utf-8").read().replace("<script>", "<script>window.MARKING_LIVE = true;</script>\n<script>", 1)
            return self.send(200, html.encode(), "text/html; charset=utf-8",
                             [("Set-Cookie", f"k={TOKEN}; Path=/; HttpOnly; SameSite=Strict")])
        if path.startswith("/frame/"):
            name = os.path.basename(path)[:-4] if path.endswith(".jpg") else ""
            f = os.path.join(args.run_dir, f"live_{name}.jpg")
            if (name in ("arm", "wrist") or (name.startswith("photo") and name[5:].isdigit())) and os.path.exists(f):
                return self.send(200, open(f, "rb").read(), "image/jpeg")
            return self.send(404)
        if path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while not os.path.exists(EVENTS):
                    self.wfile.write(b": waiting\n\n")
                    self.wfile.flush()
                    time.sleep(1)
                with open(EVENTS) as f:
                    last = time.time()
                    while True:
                        line = f.readline()
                        if line:
                            self.wfile.write(b"data: " + line.strip().encode() + b"\n\n")
                            self.wfile.flush()
                            continue
                        if time.time() - last > 10:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                            last = time.time()
                        time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                return
        return self.send(404)

    def do_POST(self):
        if not self.authed():
            return self.send(403)
        if urllib.parse.urlparse(self.path).path == "/stop":
            request_stop("STOP pressed on the dashboard")
            return self.send(200, b"stopping")
        return self.send(404)


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


threading.Thread(target=budget_guard, daemon=True).start()
for _try in range(20):          # another job on this node may hold the port: take the next free one
    try:
        srv = Server(("0.0.0.0", args.port), Handler)
        break
    except OSError:
        args.port += 1
else:
    raise SystemExit("no free port for the dashboard")
host = socket.gethostname().split(".")[0]
info = (f"Dashboard for {args.run_dir}\n"
        f"1. Forward the port: in VS Code's Ports panel, 'Forward a Port' -> {host}:{args.port}\n"
        f"   (or from your laptop: ssh -N -L {args.port}:{host}:{args.port} <you>@<cluster login node>)\n"
        f"2. Open: http://localhost:{args.port}/?k={TOKEN}\n")
with open(os.path.join(args.run_dir, "dashboard.txt"), "w") as f:
    f.write(info)
os.chmod(os.path.join(args.run_dir, "dashboard.txt"), 0o600)
print(f"Dashboard on {host}:{args.port}; open link (with access token) is in {args.run_dir}/dashboard.txt", flush=True)
srv.serve_forever()
