# Peg transfer: changes for version 3

Bigger changes that wait until v2 shows the basic task runs reliably. Items come from the v1 comparison and, once
it starts, from running v2 (whose code stays frozen during its series in the same way). Each entry says what it is,
where the idea or problem came from, and what it would take.

## Robot

1. **Both arms working at once.** While the right arm places one triangle, the left arm picks the next, the way
   skilled FLS trainees overlap their hands. *From:* user suggestion during v1 (2026-10-13). *Gain:* probably 30-40%
   less robot time (better speed scores, closer to expert technique). *Needs:* two-arm motion planning with
   continuous arm-to-arm clearance checks (where most hard v1 bugs were), and a way for the model to issue
   overlapping actions (e.g. actions that start without waiting, plus a wait/status call).

2. **Receiver retries after a failed hand-over approach start from wherever the arm stopped.** The v2 trace showed
   the receiving arm "parked" at 0.15-0.18 m beside the other arm instead of its parking spot, and the retries from
   there graze pegs and placed triangles (return trip, every seed). *Tried in v2 and reverted:* forcing the arm all the
   way to its parking pose made every retry cross the whole workspace (stand-in 707 s instead of 379 s, more contacts,
   clumsy 2/12). *Needs:* a short retreat to a safe pose next to the hand-over point (above the board, clear of the
   other arm) rather than either extreme, chosen with the kinematics and checked against placed triangles.

## Task difficulty

3. **Placing precision is saturated for GPT.** v2 round 1 (2026-10-14): mean aim error 1.2 mm, grasp offset 0.5 mm,
   picks 7/7, places 6/7. The release interlock accepts anything within 4 mm of a peg and seats it, and the board photo
   has a millimetre grid, so reading coordinates is no longer what separates a good run from a bad one; the remaining
   headroom is pace (calls per turn, robot time) and hand-over corner choice. *Needs:* a harder perception task that
   still matches FLS: no grid (scale only from the board's known size), an oblique laparoscope-like camera, a tighter
   tolerance closer to the real peg clearance, occlusion by the arms, and randomised board pose per seed.

## Feedback

4. **"About a centimetre" makes models overshoot near the 4 mm line.** Opus v2 round 1, return trip, peg (260, 240):
   it aimed 255, 265, 256, 266 (x), each a 4-6 mm miss reported as "about a centimetre" (the band is 5-12 mm, and the
   sensor noise can push a 4 mm miss over 5), so it corrected by ~10 mm and swung past the peg every time, using four
   turns on one place. *Needs:* a finer word band near the tolerance ("just over the limit" for 4-6 mm), or bands
   centred on what the model will do with them (a 5 mm miss should not be described with the same word as 11 mm).

## Harness

5. **Opus sometimes sends a tool call with empty arguments.** v2 round 1 reflection (2026-10-14): twice in a row a
   `write_lessons` call arrived with no arguments although ~750 output tokens were billed, so no lessons were saved;
   the second try of the re-run worked. v2 now retries once and saves the raw reply. *Needs:* log the raw stream
   for such a call to see whether the arguments arrive under another index or field (OpenRouter streaming), and
   check how often it happens during play turns.

6. **The hand-over failure message says a retry repeats the same thing, but it does not.** Opus v2 round 2, blue
   triangle: the hand-over failed in turn 7 ("Repeating hand_over does the same thing: place it back ... and pick it
   again"), Opus spent turn 8 trying to put it back (refused, 7 mm off the peg), then retried hand_over in turn 9 and
   it worked (the arms start from slightly different poses each time). *Needs:* either make the hand-over
   deterministic for a given grip, or say "a retry may work; if it fails again, re-grip by a corner pointing away
   from the other arm".
