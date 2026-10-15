"""The run's reward: one number (higher is better) for comparing rounds and models, computed only from what the
simulator measured. Every part is continuous, so a run that already completes all 12 transfers can still score higher
by being faster, more precise, gentler, safer or more efficient; 100 would need a perfect, instant run.

  completion  40   40 x transfers / transfers asked for (12 for the full task, 6 for --phases 1)
  speed       20   robot time per transfer, linear from 50 s (0) to 7.5 s (20), i.e. 600 s to 90 s for all 12
  precision   15   10 x mean exp(-aim error / 3 mm) over places, 5 x mean exp(-grasp offset / 3 mm) over picks
  gentleness   5   mean exp(-release height / 10 mm) over releases
  safety      10   10 x exp(-contact incidents / 3)
  efficiency  10   5 x share of actions that worked first time, 5 x turns per transfer, linear from 3.33 (0) to 0.5 (5)
  penalties        -10 per object dropped on the board, -20 per object lost off it

Contact incidents: contact events closer than 1.5 s apart count once (the monitor logs a graze on every check).

v2 also reports a model-only reward: the same, but with safety counting only contacts during actions whose geometry
the model chose (picks), not ones during motions the robot plans itself (hand-overs, parking, the guarded lowering and
backing away of a place). Robot faults then do not count against the model."""
import math

VERSION = "reward-v2"
ROBOT_PHASES = ("hand-over", "parking", "place: lowering", "place: backing away", "place: moving", "place: lining up",
                "place: re-aiming", "place: letting go")


def phase_of(what):
    return what.split("(during ")[-1].rstrip(")") if "(during " in what else ""


def incidents(events, gap=1.5):
    n, last = 0, None
    for e in sorted(events, key=lambda e: e["t"]):
        if last is None or e["t"] - last > gap:
            n += 1
        last = e["t"]
    return n


def _mean(xs, default=0.0):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else default


def compute(score, actions, turns_used, need=12):
    """score: PegTransfer.score(...) (with collision_log); actions: PegTransfer.actions; turns_used: model turns."""
    transfers = len(score["transfers"]["left->right"]) + len(score["transfers"].get("right->left", []))
    t = float(score["sim_seconds"])
    events = score.get("collision_log", {}).get("events", [])
    n_inc = incidents(events)
    n_inc_model = incidents([ev for ev in events if not phase_of(ev["what"]).startswith(ROBOT_PHASES)])
    places = [a for a in actions if a["kind"] == "place"]
    picks = [a for a in actions if a["kind"] == "pick"]
    aim = [a.get("aim_mm") for a in places if a.get("aim_mm") is not None]   # arm-half refusals have no aim
    off = [a.get("off_mm") for a in picks if a.get("off_mm") is not None]
    rel = [a.get("release_mm") for a in places if a.get("release_mm") is not None]
    parts = dict(
        completion=40 * transfers / need,
        speed=20 * min(1.0, max(0.0, (50 - t / transfers) / (50 - 7.5))) if transfers else 0.0,   # per transfer
        precision=10 * _mean([math.exp(-x / 3) for x in aim]) + 5 * _mean([math.exp(-x / 3) for x in off]),
        gentleness=5 * _mean([math.exp(-x / 10) for x in rel]),
        safety=10 * math.exp(-n_inc / 3),
        efficiency=5 * _mean([1.0 if a["ok"] else 0.0 for a in actions])
        + (5 * min(1.0, max(0.0, (40 / 12 - turns_used / transfers) / (40 / 12 - 0.5))) if transfers else 0.0),
        penalties=-10 * score.get("drops", 0) - 20 * len(score.get("lost", [])),
    )
    parts = {k: round(v, 2) for k, v in parts.items()}
    model_only = round(sum(parts.values()) - parts["safety"] + 10 * math.exp(-n_inc_model / 3), 1)
    metrics = dict(transfers=transfers, robot_s=round(t, 1), contact_incidents=n_inc, contact_incidents_model=n_inc_model,
                   turns=turns_used,
                   picks=len(picks), picks_ok=sum(a["ok"] for a in picks), places=len(places),
                   places_ok=sum(a["ok"] for a in places), places_kept=sum(bool(a.get("kept")) for a in places),
                   mean_aim_mm=round(_mean(aim, None), 1) if aim else None,
                   mean_grasp_offset_mm=round(_mean(off, None), 1) if off else None,
                   mean_release_mm=round(_mean(rel, None), 1) if rel else None,
                   drops=score.get("drops", 0), lost=len(score.get("lost", [])))
    return dict(version=VERSION, total=round(sum(parts.values()), 1), model_only=model_only, parts=parts, metrics=metrics)
