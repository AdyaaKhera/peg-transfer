"""Models on OpenRouter (e.g. anthropic/claude-opus-5.5) through its chat-completions API over plain https, so it runs
inside Isaac Sim's python. Same interface as openai_client.OpenAIClient: call(...) streams reasoning and text to
on_event, can be cancelled, and returns every tool call in order with the usage.

Cost comes from OpenRouter's own usage accounting (usage.cost, in dollars) when it reports it, else from PRICES.
The key is read from a file (OPENROUTER_KEY_FILE, default ~/.config/openrouter/key) or OPENROUTER_API_KEY; it is
never logged. Spend goes to its own ledger ($P/results/openrouter_spend.json) with its own cap."""
import http.client
import json
import os
import queue
import threading
import time

from openai_client import png_data_url

# $ per 1M tokens, as OpenRouter lists them
PRICES = {"anthropic/claude-opus-5.5": dict(input=4.00, cached=0.20, output=20.00)}


def read_key():
    path = os.environ.get("OPENROUTER_KEY_FILE", os.path.expanduser("~/.config/openrouter/key"))
    if os.path.exists(path):
        return open(path).read().strip()
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def chat_tools(tools):
    """The task's tools (Responses-API shape) in chat-completions shape."""
    return [dict(type="function", function=dict(name=t["name"], description=t["description"], parameters=t["parameters"]))
            for t in tools]


class OpenRouterClient:
    host = "openrouter.ai"

    def __init__(self, model, effort="low", max_output_tokens=4000, timeout=240):
        self.model, self.effort, self.max_output, self.timeout = model, effort, int(max_output_tokens), timeout
        self.key = read_key()
        if not self.key:
            raise RuntimeError("no OpenRouter key: put it in ~/.config/openrouter/key (chmod 600) or set OPENROUTER_KEY_FILE")

    def body(self, instructions, user_text, images, tools):
        content = [dict(type="text", text=user_text)] + [dict(type="image_url", image_url=dict(url=png_data_url(im))) for im in images]
        # the system prompt is the same every turn: mark it for prompt caching (cheaper re-reads on Anthropic models)
        system = [dict(type="text", text=instructions, cache_control=dict(type="ephemeral"))]
        b = dict(model=self.model, messages=[dict(role="system", content=system), dict(role="user", content=content)],
                 tools=chat_tools(tools), tool_choice="auto",   # Opus 5.5 rejects forced tool use; the prompt asks for a call
                 parallel_tool_calls=True, max_tokens=self.max_output, stream=True, usage=dict(include=True))
        if self.effort not in (None, "", "none"):
            b["reasoning"] = dict(effort=self.effort)
        return b

    def call(self, instructions, user_text, images, tools, on_event=lambda k, t: None, should_stop=lambda: False):
        return self._call(self.body(instructions, user_text, images, tools), on_event, should_stop)

    def _call(self, body, on_event, should_stop):
        q = queue.Queue()
        conn = http.client.HTTPSConnection(self.host, timeout=self.timeout)
        payload = json.dumps(body).encode()

        def worker():
            try:
                conn.request("POST", "/api/v1/chat/completions", body=payload, headers={
                    "Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "Accept": "text/event-stream",
                    "X-Title": "FLS peg transfer benchmark"})
                r = conn.getresponse()
                if r.status != 200:
                    q.put(("http_error", (r.status, r.read().decode(errors="replace")[:2000])))
                    return
                for raw in r:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue                    # blank lines and ": OPENROUTER PROCESSING" keep-alives
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        q.put(("sse", json.loads(data)))
                    except ValueError:
                        pass
                q.put(("eof", None))
            except Exception as e:  # includes the socket being closed by a cancel
                q.put(("exception", repr(e)))

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        t0, out = time.time(), dict(tool=None, args=None, calls=[], reasoning="", text="", usage=None, error=None, cancelled=False)
        partial = {}                                   # tool-call index -> {"name", "args" (accumulated JSON text)}
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
            if ev.get("error"):
                out["error"] = f"API error: {json.dumps(ev['error'])[:500]}"
                break
            for ch in ev.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("reasoning"):
                    out["reasoning"] += delta["reasoning"]
                    on_event("reasoning", delta["reasoning"])
                if delta.get("content"):
                    out["text"] += delta["content"]
                    on_event("text", delta["content"])
                for tc in delta.get("tool_calls") or []:
                    p = partial.setdefault(tc.get("index", len(partial)), dict(name=None, args=""))
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        p["name"] = fn["name"]
                    p["args"] += fn.get("arguments") or ""
                if ch.get("finish_reason") == "length":
                    out["error"] = "stopped at the output token limit"
            u = ev.get("usage")
            if u:
                out["usage"] = dict(input=u.get("prompt_tokens", 0),
                                    cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                                    output=u.get("completion_tokens", 0),
                                    reasoning=(u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))
                if u.get("cost") is not None:
                    out["usage"]["cost_usd"] = float(u["cost"])
        conn.close()
        for k in sorted(partial):                      # every call, in order: pick -> hand_over -> place
            p = partial[k]
            try:
                a = json.loads(p["args"] or "{}")
            except ValueError:
                a = None
            out["calls"].append((p["name"], a))
        if out["calls"]:
            out["tool"], out["args"] = out["calls"][0]
        out["seconds"] = time.time() - t0
        return out
