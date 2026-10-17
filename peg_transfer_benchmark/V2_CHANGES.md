# Peg transfer: changes for version 2

Found while running the frozen v1 comparison (GPT-6.1 Sol vs Claude Opus 5.5, from 2026-10-13). None of these are
applied to v1: its code stays frozen so every round runs under the same conditions. Each entry says what went wrong,
where it was seen, and the proposed change.

## Status (frozen as git tag peg-v2, 2026-10-13)

Done in v2: 4a (arm halves), 4c (longer line-up), 5 / 8 (hand-over failure explains the fix), 5b (turn budget),
6 / 6b / 6c / 7 (feedback that says where each miss was; first as exact grip points and peg centres, then replaced
before the runs by rough directions in words, see below), 9 / 10 (reflection with photos, methods not
coordinates), 11 (clumsy stand-in), 13 (model-only reward), 15a (live reward), 15c (gated checks), 1 (shifted
hand-over presentations), plus: release interlock (never let go unless within 4 mm of a peg; seat on the peg), tilted
triangles pinched square to their surface, standing triangles (> 35 degrees) reported as unrecoverable, dashboard
port fallback, effort medium by default.
Acceptance (free): clumsy stand-in 12/12, 0 drops on seeds 0 and 1 (reward 58.2 / 57.2, model-only 65.3 / 62.3);
perfect stand-in 12/12, 0 drops, seed 2 (75.9 / 85.4).
Feedback as run (tag peg-v2-vague, 2026-10-14): direction and a rough size in words only ("a little", "about a
centimetre", "well" to the left / right / higher / lower), no numbers, judged from the noisy camera-like sensor.
Clumsy stand-in with it: seed 3 12/12, seed 0 11/12 (out of 45 turns), 0 drops; 17-20 place refusals per run.

Earlier steps: final changes before the first v2 round (tag peg-v2-final, 2026-10-14): feedback gives only relative offsets
("8 mm to the left and 6 mm lower in the photo"), never positions, with seeded sensor noise that grows with the
size of the miss like a camera-based sensor (about 1 mm + 25% of the miss, at most 5 mm per axis; tag peg-v2-run); the reflection is asked to record any steady bias; a tilted held triangle is levelled before release
or not released; the clumsy stand-in follows the hand-over advice. Acceptance with relative, noisy feedback: clumsy
12/12, 0 drops on seeds 0, 1 and 3 (57.3 / 57.3 / 56.8); with the distance-dependent sensor, seeds 0 and 3:
12/12, 0 drops (55.4 / 55.3; some misses now need two corrections); perfect stand-in 12/12, 0 drops on seed 2 (75.9).
Known limitation: return-trip hand-overs can still graze a peg or a placed triangle (robot motion; excluded from the
model-only reward). See V3_CHANGES.md item 2.

## Robot / controller

1. **Hand-over clash on some grasp corners.** When the holder pinches the corner that points at the receiver, its
   wrist sits in the receiver's path; the receiver cycles through 6 presentations x 4 angles, brushing the other arm
   or nearby triangles, and often fails. (GPT round 2: 332 contact events; Opus round 2: purple could not be handed
   over.) Never seen with the stand-in, which always pinches the roomiest corner.
   *Change:* choose the presentation from the grasp corner up front (predict clearance with the kinematics before
   moving), check clearance continuously rather than after each 2-4 cm leg, and offer a re-grip when no approach works.
2. **Left arm's reach-in dips to the board on the return trip** (phase 2, once per run, every seed). The joint-swing
   guard did not fix it, so the dip comes from another step of the reach-in; the PEG_DEBUG leg trace was set up but
   not run. *Change:* finish the trace and fix that step.
3. **Contacts happen mid-move before the arm-clearance check reacts** (legs of 2-4 cm). *Change:* predictive checks
   on the planned path.

4c. **Placing refuses at 5 mm** ("could not hold the blue object precisely over that point (5 mm off)", GPT r3,
    peg (260, 240)): the controller needs the hole within 4 mm before letting go, and some reachable poses only get
    to 5 mm. *Change:* a closer final alignment (more correction steps, or re-plan the wrist turn), and feedback that
    says what to do ("hand it over and back to re-grip, or choose another peg").

## Task and instructions

4. **Arm reach is never stated.** Opus tried six times to pick a triangle in the right field with the left arm.
   *Change:* "each arm reaches only its own half of the board; recover a triangle dropped after a transfer with the
   arm on that side".
4a. **Decided for v2 (replaces #4 and #6): each arm works only its own half.** A pick or place with an arm outside its
   half is refused at once, without moving, with "the left arm works only the left field; use the right arm for
   (570, 36)"; the instructions say each arm has its own half and a triangle dropped in the right field is recovered
   with the right arm. Matches the FLS drill (hands cross only through the mid-air hand-over).
4b. **GPT hit the same reach problem in round 3:** eight picks in a row with the left arm in the right field
   (687, 179), (678, 218), ... before giving up. So #4 and #6 affect both models, not just Opus.
5. **Retrying a failed hand-over repeats the same motion.** Opus: "not much else to adjust, so I'll just attempt the
   hand-over again". *Change:* say so in the hand_over description, and add a way to set a triangle down / re-grip by
   another corner.

5b. **Turn pressure makes models undo their own progress.** GPT r3, with 7 turns left, put a triangle it was
    carrying back to the left field onto a ring peg (and missed) rather than spend two turns finishing the transfer.
    *Change:* show the budget plainly every turn ("7 turns left; a full transfer takes about 3 turns, or 1 if you
    chain calls") and say that putting a triangle back on its starting side scores nothing.

## Feedback to the model

6. **Out-of-reach picks are reported as aim errors** ("finger pads 31 mm to the side of the corner's centre line"),
   which sent Opus recalculating its scale. *Change:* "the left arm cannot reach (570, 36): it is outside that arm's
   half of the board".
6c. **Pick misses give the distance but not the direction** ("finger pads 8 mm to the side of the nearest corner's
    centre line"). Opus r3 wrote the right lesson ("shift the aim perpendicular to the centre line by that full
    amount") but could not apply it without knowing which side, so it nudged the same point by ~1 mm and missed again
    (5 misses in its first 5 turns). *Change:* give a signed offset in board coordinates, e.g. "the corner's centre
    line passes 8 mm to the right of (122.7, 30), at about (130, 31)".
7. **A missed place reports where the triangle landed, not where the peg is.** GPT aimed at (600, 40) for the peg at
   (605, 55) in round 2, wrote "(600, 40)" into a lesson, and missed at the same spot again in round 3. *Change:*
   "missed: you aimed 15.8 mm from the nearest peg's centre at (605, 55)".
6b. **Pick feedback can report a negative depth** ("the finger pads were -17 mm in from the tip"), i.e. beyond the
   tip, outside the triangle. *Change:* say "the fingers closed 17 mm beyond the corner's tip, outside the triangle".
8. **A failed hand-over gives no reason.** *Change:* say why (arm clearance with the corner held) and what would help
   (pinch a different corner).

## Reflection and lessons

9. **The reflection is text-only**, so a vision mistake cannot be re-measured. *Change:* show the reflection the
   final board photo and the photo at each failed action.
10. **Lessons store absolute coordinates, including wrong ones.** *Change:* ask for methods, not coordinates ("re-read
    each peg's centre from the grid every turn"), and ask the model to check any coordinate from a lesson against the
    current photo before using it.

## Evaluation protocol

11. **Acceptance testing used only a perfect stand-in**, so the error paths (misses, drops, wrong arm, odd corners)
    were never exercised before paid rounds. *Change:* add a "clumsy" stand-in (aim off by ~10 mm, random corners,
    drops, wrong-arm recovery) and require it to run cleanly before any paid round.
12. **Round settings drifted:** round 1 was stopped by hand after the first half, later rounds ran full length.
    *Change:* fix task length, turn limit and stopping rule before round 1 and never change them within a series.
13. **Controller faults count against the model's reward** (safety and speed in GPT round 2). *Change:* attribute
    each contact and lost second to controller or model, and report a model-only reward next to the total.
14. **One run per round is noisy** (GPT: 52.7 then 26.3, mostly hand-over luck). *Change:* 2-3 runs per round, or
    more seeds, if the budget allows.

## Dashboard

15c. **Empty passes in the checklist.** Opus r3 showed 3 of 7 checks passed with 0 transfers (drops 0, lost 0, robot
    time under 300 s), because nothing was carried and the robot barely moved. *Change:* gate the checks on work done,
    as FLS does: time passes only if every transfer is completed within 300 s ("not finished" otherwise); drops and
    lost show "n/a" until a triangle has been carried and are then judged per transfer attempted; the headline leads
    with transfers ("0 of 12"). The reward is already right (0 completion, 0 speed, near-0 efficiency).
15a. **Live reward.** Today the reward appears only when a run ends, so there is no way to tell mid-run whether a round
    is worth paying for. *Change:* after every turn, compute the reward so far (same formula, on what has happened),
    show it on the dashboard with its parts, next to that model's previous rounds and the stand-in's score, plus the
    reward per dollar spent; flag when it is trending below the model's previous round.

## Later

Bigger changes for after v2 are in [V3_CHANGES.md](V3_CHANGES.md) (both arms working at once, ...).

## Separate experiments (not v2 itself)

15. Effort `medium` vs `low` (more careful calibration; about 2-3x the cost per turn).
16. With vs without wrist-camera photos (do the models use them?).
17. Language model as planner + pi0.5 as executor (fine-tuned on stand-in demonstrations), to remove coordinate
    reading errors.
