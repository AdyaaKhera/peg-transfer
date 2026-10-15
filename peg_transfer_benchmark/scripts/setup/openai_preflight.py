"""Free check before the first paid run: is the key readable, does it work, and does the model id exist?
Uses GET /v1/models/<id>, which costs nothing. Run on the login node:  python3 peg_transfer_benchmark/scripts/setup/openai_preflight.py <model id>"""
import json
import os
import stat
import sys
import urllib.error
import urllib.request

model = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OPENAI_MODEL", "")
path = os.environ.get("OPENAI_KEY_FILE", os.path.expanduser("~/.config/openai/key"))
if os.path.exists(path):
    mode = stat.S_IMODE(os.stat(path).st_mode)
    if mode & 0o077:
        sys.exit(f"{path} is readable by others ({oct(mode)}): chmod 600 {path}")
    key = open(path).read().strip()
    print(f"key file ok ({path}, {oct(mode)}, {len(key)} chars, starts {key[:7]}...)")
elif os.environ.get("OPENAI_API_KEY", "").strip():
    key = os.environ["OPENAI_API_KEY"].strip()
    print(f"using OPENAI_API_KEY from the environment ({len(key)} chars, starts {key[:7]}...)")
else:
    sys.exit(f"no key: put it in {path} (chmod 600) or set OPENAI_API_KEY")


def get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        return 200, json.loads(urllib.request.urlopen(req, timeout=20).read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:300]


code, body = get("https://api.openai.com/v1/models")
if code != 200:
    sys.exit(f"key rejected: HTTP {code} {body}")
ids = sorted(m["id"] for m in body.get("data", []))
print(f"key works: {len(ids)} models visible")
if not model:
    sys.exit("no model id given; models with 'gpt' in the name:\n  " + "\n  ".join(i for i in ids if "gpt" in i))
code, body = get(f"https://api.openai.com/v1/models/{model}")
if code == 200:
    print(f"model '{model}' is available to this key. Ready:  MODEL={model} sbatch peg_transfer_benchmark/slurm/run_pegs.sbatch")
else:
    import re
    want = set(re.findall(r"[a-z]+|\d+(?:\.\d+)?", model.lower())) - {"gpt"}
    def score(i):                      # shared words and version numbers, in any order ("gpt-sol-6.1" -> "gpt-6.1-sol")
        have = set(re.findall(r"[a-z]+|\d+(?:\.\d+)?", i.lower()))
        return len(want & have)
    near = sorted((i for i in ids if score(i)), key=lambda i: (-score(i), i))
    sys.exit(f"model '{model}' not found (HTTP {code}). Closest ids: {', '.join(near[:10]) or 'none'}")
