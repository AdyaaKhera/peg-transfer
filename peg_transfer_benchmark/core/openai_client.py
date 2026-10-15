"""OpenAI Responses API over plain https (no SDK, so it runs inside Isaac Sim's python), with streaming reasoning
summaries, cancellation, and a persistent budget that refuses any call that could take total spend past the cap.

The key is read from a file (OPENAI_KEY_FILE, default ~/.config/openai/key) or OPENAI_API_KEY; it is never logged.
The stand-in model for $0 runs is in pegboard.py (it plays the task from the ground truth)."""
import base64
import http.client
import io
import json
import math
import os
import queue
import threading
import time

from PIL import Image

# $ per 1M tokens (Standard). The cached price applies to the cached part of the input.
PRICES = {"standard": dict(input=2.00, cached=0.10, output=10.00)}
IMAGE_TOKENS_EST = 2500       # deliberately high guess per image, for the worst-case check only
CHARS_PER_TOKEN_EST = 3.0     # likewise conservative


def cost_of(usage, price):
    cached = usage.get("cached", 0)
    return ((usage.get("input", 0) - cached) * price["input"] + cached * price["cached"] + usage.get("output", 0) * price["output"]) / 1e6


class Budget:
    """Spend across every real call, stored in a JSON ledger so the cap holds across runs and jobs."""

    def __init__(self, path, cap, price):
        self.path, self.cap, self.price = path, float(cap), price
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _load(self):
        try:
            return json.load(open(self.path))
        except (OSError, ValueError):
            return dict(spent=0.0, calls=[])

    def spent(self):
        return float(self._load()["spent"])

    def worst_case(self, est_input_tokens, max_output_tokens):
        return (est_input_tokens * self.price["input"] + max_output_tokens * self.price["output"]) / 1e6

    def allows(self, worst):
        return self.spent() + worst <= self.cap

    def record(self, cost, **meta):
        led = self._load()
        led["spent"] = float(led["spent"]) + float(cost)
        led["calls"].append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), cost=round(float(cost), 6), **meta))
        tmp = self.path + ".tmp"
        json.dump(led, open(tmp, "w"), indent=1)
        os.replace(tmp, self.path)
        return led["spent"]


def read_key():
    path = os.environ.get("OPENAI_KEY_FILE", os.path.expanduser("~/.config/openai/key"))
    if os.path.exists(path):
        return open(path).read().strip()
    return os.environ.get("OPENAI_API_KEY", "").strip()


def png_data_url(img, max_side=1024):
    im = Image.fromarray(img)
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def estimate_input_tokens(texts, n_images, tools):
    chars = sum(len(t) for t in texts) + len(json.dumps(tools))
    return int(chars / CHARS_PER_TOKEN_EST) + IMAGE_TOKENS_EST * n_images


class Cancelled(Exception):
    pass


class OpenAIClient:
    """One call per turn, streamed. on_event(kind, text) gets "reasoning" and "text" deltas as they arrive;
    should_stop() is polled ~4x a second and closes the connection when it returns True."""
    host = "api.openai.com"

    def __init__(self, model, effort="low", max_output_tokens=4000, timeout=240, parallel_tools=True):
        self.model, self.effort, self.max_output, self.timeout = model, effort, int(max_output_tokens), timeout
        self.parallel = parallel_tools
        self.key = read_key()
        self.use_reasoning = effort not in (None, "", "none")
        if not self.key:
            raise RuntimeError("no OpenAI key: put it in ~/.config/openai/key (chmod 600) or set OPENAI_KEY_FILE")

    def body(self, instructions, user_text, images, tools):
        content = [dict(type="input_text", text=user_text)] + [dict(type="input_image", image_url=png_data_url(im), detail="high") for im in images]
        b = dict(model=self.model, instructions=instructions, input=[dict(role="user", content=content)],
                 tools=tools, tool_choice="required", parallel_tool_calls=self.parallel,
                 max_output_tokens=self.max_output, stream=True, store=False)
        if self.use_reasoning:
            b["reasoning"] = dict(effort=self.effort, summary="auto")
        return b

    def call(self, instructions, user_text, images, tools, on_event=lambda k, t: None, should_stop=lambda: False):
        res = self._call(self.body(instructions, user_text, images, tools), on_event, should_stop)
        if res.get("http_status") == 400 and self.use_reasoning and "reasoning" in str(res.get("error", "")):
            # this model takes no reasoning settings: retry once without them (a rejected request is not billed)
            self.use_reasoning = False
            on_event("info", "model does not accept reasoning settings; retrying without them")
            res = self._call(self.body(instructions, user_text, images, tools), on_event, should_stop)
        return res

    def _call(self, body, on_event, should_stop):
        q = queue.Queue()
        conn = http.client.HTTPSConnection(self.host, timeout=self.timeout)
        payload = json.dumps(body).encode()

        def worker():
            try:
                conn.request("POST", "/v1/responses", body=payload, headers={
                    "Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "Accept": "text/event-stream"})
                r = conn.getresponse()
                if r.status != 200:
                    q.put(("http_error", (r.status, r.read().decode(errors="replace")[:2000])))
                    return
                ev_type, data = None, []
                for raw in r:
                    line = raw.decode(errors="replace").rstrip("\r\n")
                    if line.startswith("event:"):
                        ev_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data.append(line[5:].strip())
                    elif line == "" and data:
                        try:
                            q.put(("sse", json.loads("\n".join(data))))
                        except ValueError:
                            pass
                        ev_type, data = None, []
                q.put(("eof", None))
            except Exception as e:  # includes the socket being closed by a cancel
                q.put(("exception", repr(e)))

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        t0, out = time.time(), dict(tool=None, args=None, calls=[], reasoning="", text="", usage=None, error=None, cancelled=False)
        fn_args = {}
        while True:
            if should_stop():
                out["cancelled"] = True
                try:
                    conn.sock and conn.sock.shutdown(2)
                except OSError:
                    pass
                conn.close()
                break
            try:
                kind, ev = q.get(timeout=0.25)
            except queue.Empty:
                continue
            if kind == "http_error":
                out["http_status"], out["error"] = ev[0], f"HTTP {ev[0]}: {ev[1]}"
                break
            if kind == "exception":
                out["error"] = f"connection: {ev}"
                break
            if kind == "eof":
                break
            t = ev.get("type", "")
            if t == "response.reasoning_summary_text.delta":
                out["reasoning"] += ev.get("delta", "")
                on_event("reasoning", ev.get("delta", ""))
            elif t == "response.reasoning_summary_part.added" and out["reasoning"]:
                out["reasoning"] += "\n\n"
                on_event("reasoning", "\n\n")
            elif t == "response.output_text.delta":
                out["text"] += ev.get("delta", "")
                on_event("text", ev.get("delta", ""))
            elif t == "response.output_item.done" and ev.get("item", {}).get("type") == "function_call":
                it = ev["item"]
                try:
                    a = json.loads(it.get("arguments") or "{}")
                except ValueError:
                    a = None
                out["calls"].append((it.get("name"), a))     # every call, in order: pick -> hand_over -> place
                if out["tool"] is None:
                    out["tool"], out["args"] = it.get("name"), a
            elif t in ("response.completed", "response.incomplete", "response.failed"):
                resp = ev.get("response", {})
                u = resp.get("usage") or {}
                out["usage"] = dict(input=u.get("input_tokens", 0), cached=(u.get("input_tokens_details") or {}).get("cached_tokens", 0),
                                    output=u.get("output_tokens", 0), reasoning=(u.get("output_tokens_details") or {}).get("reasoning_tokens", 0))
                if t != "response.completed":
                    out["error"] = f"{t}: {json.dumps(resp.get('incomplete_details') or resp.get('error'))[:300]}"
                break
            elif t == "error":
                out["error"] = f"API error: {json.dumps(ev)[:500]}"
                break
        conn.close()
        out["seconds"] = time.time() - t0
        return out
