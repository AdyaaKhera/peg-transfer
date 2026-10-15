"""Between rounds: a debrief of the round just played (its reward and every failure, measured by the simulator) and
the model's own reflection on it, a curated list of at most MAX_LESSONS lessons that goes into its next round's
instructions. The model writes the lessons; the harness only reports what happened.

Saturation: a model's series stops when its reward rose by less than SAT_GAIN in each of the last SAT_ROUNDS rounds,
or after MAX_ROUNDS rounds."""
import json

MAX_LESSONS = 10
MAX_ROUNDS = 5
SAT_GAIN, SAT_ROUNDS = 3.0, 2
FAIL_WORDS = ("missed", "closed on nothing", "could not", "cannot", "collisions during", "fell", "lost", "no usable",
              "as low as", "does not count", "bad arguments", "needs exactly", "unknown tool", "already holding")

LESSON_TOOL = dict(
    type="function", name="write_lessons",
    description="Your lessons for the next round, replacing the current list: at most 10, each one short, concrete and "
                "actionable (what to do differently, with numbers where you have them).",
    parameters=dict(type="object", additionalProperties=False, required=["lessons", "note"], properties=dict(
        lessons=dict(type="array", items=dict(type="string"), maxItems=MAX_LESSONS),
        note=dict(type="string", description="one sentence: what you changed in the list and why"))))

INSTRUCTIONS = """You just played one round of the FLS peg transfer with a two-armed robot (pick, hand_over, place).
Below is the debrief, measured by the simulator: the round's reward and its parts, and every action that failed or
caused a contact. Write the lessons you will carry into your next round by calling write_lessons once.

- Keep lessons that helped (the reward went up after you adopted them) and drop or rewrite ones that did not.
- Be concrete: name the situation and the fix ("aim 1-2 mm short of a ring peg's centre towards...", "pick the corner
  pointing away from the nearest peg"), not general advice.
- Lessons must be about your own decisions (where to aim, which corner, what order, how many calls per turn, release
  height): the robot's motion controller is fixed and you cannot change it.
- Write methods, not coordinates: "re-read each peg's centre from the grid every turn", not "aim at (600, 40)". The
  feedback says roughly which way each miss was off: if your misses keep pointing the same way, write down that bias
  and how you will correct it ("my readings land about a centimetre too far left: aim further right").
- You are also shown photos: the board at the end of the round and the overhead photo of up to two turns that went
  wrong. Use them to check your diagnosis (for example, re-measure a peg you missed).
- At most 10 lessons."""


def debrief(result, turns, history):
    """Plain-text debrief of one round. history: earlier rounds [{round, reward}]."""
    rw = result.get("reward") or {}
    m = rw.get("metrics", {})
    lines = [f"Round reward: {rw.get('total', 0):.1f} (higher is better)."]
    if history:
        lines.append("Earlier rounds: " + ", ".join(f"round {h['round']}: {h['reward']:.1f}" for h in history) + ".")
    lines.append("Parts: " + ", ".join(f"{k} {v:g}" for k, v in (rw.get("parts") or {}).items()) + ".")
    lines.append(f"Transfers {m.get('transfers')}/{6 * result.get('phases', 2)}, robot time {m.get('robot_s')} s (the FLS limit is 300 s), "
                 f"{m.get('turns')} turns, {m.get('contact_incidents')} contact incidents, drops {m.get('drops')}, lost {m.get('lost')}; "
                 f"picks {m.get('picks_ok')}/{m.get('picks')} worked, places {m.get('places_ok')}/{m.get('places')} worked; "
                 f"mean place aim error {m.get('mean_aim_mm')} mm, mean grasp offset {m.get('mean_grasp_offset_mm')} mm, "
                 f"mean release height {m.get('mean_release_mm')} mm.")
    calls_per_turn = [len(t.get("calls") or []) for t in turns]
    if calls_per_turn:
        lines.append(f"Tool calls per turn: mean {sum(calls_per_turn) / len(calls_per_turn):.1f} (up to 6 allowed).")
    lines.append("")
    lines.append("Turns with a failure or a contact:")
    n = 0
    for t in turns:
        res = t.get("results") or []
        bad = [r for r in res if any(w in r for w in FAIL_WORDS)]
        if bad:
            calls = "; ".join(f"{c}({', '.join(f'{k}={v}' for k, v in (a or {}).items() if k != 'note')})" for c, a in (t.get("calls") or []))
            lines.append(f"- turn {t['turn']}: {calls} -> " + " | ".join(bad))
            n += 1
    if not n:
        lines.append("- none")
    return "\n".join(lines)


def saturated(history):
    """history: [{round, reward}] in order. True when the series should stop."""
    if len(history) >= MAX_ROUNDS:
        return True
    if len(history) <= SAT_ROUNDS:
        return False
    gains = [history[k]["reward"] - history[k - 1]["reward"] for k in range(len(history) - SAT_ROUNDS, len(history))]
    return all(gn < SAT_GAIN for gn in gains)


def prompt_section(lessons, history):
    """What goes into the next round's instructions."""
    if not lessons and not history:
        return ""
    best = max((h["reward"] for h in history), default=None)
    out = ["", "Lessons from your earlier rounds (you wrote these after each round's debrief"
           + (f"; your last reward was {history[-1]['reward']:.1f}, your best {best:.1f}" if history else "") + "):"]
    out += [f"- {x}" for x in lessons]
    return "\n".join(out)


def parse_lessons(res):
    for name, a in res.get("calls") or []:
        if name == "write_lessons" and a and isinstance(a.get("lessons"), list):
            return [str(x)[:400] for x in a["lessons"]][:MAX_LESSONS], str(a.get("note", ""))[:400]
    return None, res.get("error") or "no write_lessons call"


def lessons_from_text(text):
    """Fallback when the model wrote its lessons as a list in plain text: take the bulleted or numbered lines."""
    import re
    items = [re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", ln).strip() for ln in text.splitlines()
             if re.match(r"^\s*(?:[-*\u2022]|\d+[.)])\s+\S", ln)]
    return [x[:400] for x in items if len(x) > 15][:MAX_LESSONS] or None


def load_series(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return dict(rounds=[], lessons=[])
