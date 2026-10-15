"""Offline reachability of the peg board with the YAM kinematics (no simulator; needs numpy + scipy).

For every peg, every triangle orientation and every corner: can the arm working that field
  pick   pinch the corner from above (gripper vertical), and
  place  hold the triangle by a corner, approaching it `elev` degrees below level (the grip a hand-over leaves),
         both at release height and 8 cm above it?
Prints, per peg, how many of the orientations have at least one reachable corner.

  python reach_check.py [--elevs 35,25,15,5] [--arm left|right|both]"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "core"))
from kin import Solver  # noqa: E402

GEOM = np.load(os.path.join(HERE, "..", "..", "core", "yam_geometry.npz"))
S = Solver(GEOM)
DOWN = np.array([0, 0, -1.0])
PAPER_TOP, BOARD_MM, BC = 0.001, (810.0, 300.0), (0.34, -0.305)
OBJ_RO, PINCH_D, OBJ_H, PEG_TOP = 0.070 / math.sqrt(3), 0.012, 0.020, 0.041
RING_C, RING_R = (605.0, 150.0), 95.0
FIELDS = {"left": [(x, y) for x in (130.0, 260.0) for y in (60.0, 150.0, 240.0)],
          "right": [(RING_C[0] + RING_R * math.cos(math.radians(90 + 60 * k)), RING_C[1] - RING_R * math.sin(math.radians(90 + 60 * k)))
                    for k in range(6)]}


def b2w(x, y):
    return np.array([BC[0] + (BOARD_MM[1] / 2000 - y / 1000), BC[1] - (x / 1000 - BOARD_MM[0] / 2000)])


def reach(side, p, a, c):
    for q0 in (np.zeros(6), np.array([0, 1.0, 1.0, 0, 0, 0]), np.array([0, 0.6, 1.4, -0.5, 0, 0])):
        _, pe, ae = S.solve(side, p, a, c, q0, symmetric=True, tries=6)
        if pe < 0.002 and ae < math.radians(3):
            return True
    return False


ap = argparse.ArgumentParser()
ap.add_argument("--elevs", default="35,25,15,5")
ap.add_argument("--arm", default="both")
ap.add_argument("--yaws", type=int, default=6)
args = ap.parse_args()
for side in (["left", "right"] if args.arm == "both" else [args.arm]):
    print(f"== {side} arm, {side} field")
    for elev in [float(e) for e in args.elevs.split(",")]:
        e = math.radians(elev)
        row = []
        for (x, y) in FIELDS[side]:
            c = b2w(x, y)
            n_pick = n_place = 0
            for j in range(args.yaws):
                yaw = j * (2 * math.pi / 3) / args.yaws
                pk = pl = False
                for k in range(3):
                    ang = yaw + math.pi / 2 + 2 * math.pi * k / 3
                    u = np.array([math.cos(ang), math.sin(ang), 0.0])
                    clo = np.cross(DOWN, u)
                    pt = np.array([*(c + (OBJ_RO - PINCH_D) * u[:2]), PAPER_TOP + OBJ_H - 0.004])
                    pk = pk or reach(side, pt, DOWN, clo)
                    a = math.cos(e) * -u + math.sin(e) * DOWN
                    low = np.array([pt[0], pt[1], PEG_TOP + 0.005 + OBJ_H / 2])
                    if not pl and reach(side, low, a, clo) and reach(side, low + np.array([0, 0, 0.08]), a, clo):
                        pl = True
                n_pick += pk
                n_place += pl
            row.append(f"({x:.0f},{y:.0f}) pick {n_pick}/{args.yaws} place {n_place}/{args.yaws}")
        print(f"  grip {elev:4.0f} deg: " + " | ".join(row), flush=True)
