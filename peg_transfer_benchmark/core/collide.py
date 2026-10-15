"""Collision monitor. The YAM URDF has no collision shapes, so PhysX never sees the arms touch anything, and a held
object is kinematic, so PhysX lets it pass through pegs. This checks every few sim steps, geometrically:

  arm vs arm, arm vs board, arm vs peg, arm vs object (fingertips may touch only the object they are grasping or
  holding), held object vs peg / other object / board.

The arms are clouds of surface points sampled from their meshes (scripts/setup/robot_geometry.py), placed with the
links' live poses; pegs are cylinders, the objects are three wall boxes each. A contact is a point inside (or within MARGIN of)
another body. Each new contact is reported once, with the sim time."""
import os

import numpy as np

import pegboard as PB
import yam_env as Y
from ik import GEOM

MARGIN = 0.0005          # count real overlap (half a millimetre), not near misses or sliding contact
TABLE_TOP = 0.0          # yam_env's table: a 1.0 x 1.5 m slab whose top is at z = 0
ARM_ARM = 0.004
OVERLAP = 0.0005         # a carried object counts as hitting a peg or another object only when it overlaps by this much          # two arm surface points closer than this (sample spacing ~4 mm) = touching
MAX_PTS = {"link1": 250, "link2": 400, "link3": 450, "link4": 300, "link5": 250, "gripper": 400, "tip_left": 300, "tip_right": 300}


def _bar_points(n_side=6):
    """Surface sample points of one bar box, in the bar's own frame (x along the bar)."""
    L, W, H = PB.BAR_L, PB.OBJ_W, PB.OBJ_H
    g = np.stack(np.meshgrid(np.linspace(-L / 2, L / 2, 4 * n_side), np.linspace(-W / 2, W / 2, 3), np.linspace(-H / 2, H / 2, 3)), -1).reshape(-1, 3)
    on_surface = (np.abs(np.abs(g) - [L / 2, W / 2, H / 2]) < 1e-9).any(1)
    return g[on_surface]


class Monitor:
    def __init__(self, env, task, every=2, on_contact=None):
        self.env, self.task, self.every = env, task, every
        self.on_contact = on_contact or (lambda text: None)
        rng = np.random.default_rng(0)
        self.pts = {}
        for n in PB.MONITOR_LINKS:
            p = GEOM[f"pts_{n}"]
            k = MAX_PTS[n]
            self.pts[n] = p[rng.choice(len(p), size=min(k, len(p)), replace=False)]
        self.bar_pts = _bar_points()
        self.active, self.events, self.n = set(), [], 0
        self.min_clear = {"arm_arm": np.inf}
        L, W, H = PB.BAR_L, PB.OBJ_W, PB.OBJ_H
        self.half = np.array([L / 2, W / 2, H / 2]) + MARGIN
        self.pegs = PB.board_to_world(PB.PEGS_MM)

    # ---------- geometry
    def arm_points(self, side):
        """{link: (N, 3) world points} for one arm."""
        p, q = self.task.link_views[side].get_world_poses()
        out = {}
        for j, n in enumerate(PB.MONITOR_LINKS):
            R = Y.quat_to_mat(np.asarray(q[j], float))
            out[n] = self.pts[n] @ R.T + np.asarray(p[j], float)
        return out

    def bar_frames(self, i):
        T = self.task.obj_T(i)
        frames = []
        for mid, d, _, _ in PB.BARS:
            R = np.eye(3)
            R[:, 0] = d
            R[:, 2] = [0, 0, 1]
            R[:, 1] = np.cross(R[:, 2], R[:, 0])
            frames.append((T[:3, :3] @ R, T[:3, :3] @ mid + T[:3, 3]))
        return frames

    def inside_object(self, pts, i, margin=MARGIN):
        """Boolean mask: which points are inside (or within margin of; negative = that deep inside) object i."""
        hit = np.zeros(len(pts), bool)
        half = self.half - MARGIN + margin
        for R, c in self.bar_frames(i):
            loc = (pts - c) @ R
            hit |= (np.abs(loc) <= half).all(1)
        return hit

    def held_points(self, i):
        out = []
        for R, c in self.bar_frames(i):
            out.append(self.bar_pts @ R.T + c)
        return np.vstack(out)

    # ---------- checks
    def check(self):
        self.n += 1
        if self.n % self.every:
            return
        t = self.task
        found = {}
        arms = {s: self.arm_points(s) for s in ("left", "right")}
        live = [i for i in range(t.n) if i not in t.lost]
        for side, links in arms.items():
            for n, P in links.items():
                tip = n.startswith("tip")
                low = P[P[:, 2] < PB.PEG_TOP + 0.01]
                if len(low):
                    bxy = PB.world_to_board(low[:, :2])
                    on_board = (bxy[:, 0] >= 0) & (bxy[:, 0] <= PB.BOARD_MM[0]) & (bxy[:, 1] >= 0) & (bxy[:, 1] <= PB.BOARD_MM[1])
                    if (on_board & (low[:, 2] < Y.PAPER_TOP - MARGIN)).any():
                        found[(side, n, "board")] = f"{side} {n} touched the board"
                    elif (~on_board & (low[:, 2] < TABLE_TOP - MARGIN)).any():
                        found[(side, n, "table")] = f"{side} {n} touched the table"
                    peg_h = low[low[:, 2] < PB.PEG_TOP + MARGIN]        # only points at the pegs' height can touch one
                    d = np.linalg.norm(peg_h[:, None, :2] - self.pegs[None], axis=2)
                    k = np.where((d < PB.PEG_R + MARGIN).any(0))[0] if len(peg_h) else []
                    for kk in k:
                        found[(side, n, f"peg{kk}")] = f"{side} {n} hit peg {kk + 1}"
                for i in live:
                    if tip and t.engaged[side] == i:
                        continue                                  # fingers on the object they are grasping
                    lo, hi = t.obj_T(i)[:3, 3] - 0.07, t.obj_T(i)[:3, 3] + 0.07
                    near = P[((P > lo) & (P < hi)).all(1)]
                    if len(near) and self.inside_object(near, i).any():
                        found[(side, n, f"obj{i}")] = f"{side} {n} hit the {PB.COLORS[i][0]} object"
        # arm vs arm: only link pairs whose bounding boxes come close
        best = np.inf
        for nl, Pl in arms["left"].items():
            lo_l, hi_l = Pl.min(0), Pl.max(0)
            for nr, Pr in arms["right"].items():
                lo_r, hi_r = Pr.min(0), Pr.max(0)
                if (lo_l > hi_r + 0.02).any() or (lo_r > hi_l + 0.02).any():
                    continue
                d = np.sqrt(((Pl[:, None, :] - Pr[None, :, :]) ** 2).sum(-1)).min()
                best = min(best, d)
                if d < ARM_ARM:
                    found[("arms", nl, nr)] = f"left {nl} touched right {nr}"
        self.min_clear["arm_arm"] = min(self.min_clear["arm_arm"], best)
        # held objects vs pegs, other objects, board
        for side in ("left", "right"):
            i = t.holding(side)
            if i is None:
                continue
            H = self.held_points(i)
            name = PB.COLORS[i][0]
            if (H[:, 2] < Y.PAPER_TOP - MARGIN).any():
                found[("held", i, "board")] = f"the held {name} object touched the board"
            low = H[H[:, 2] < PB.PEG_TOP]
            if len(low):
                d = np.linalg.norm(low[:, None, :2] - self.pegs[None], axis=2)
                # a carried object may slide along a peg it was resting against (lifting it off): only real overlap counts
                for kk in np.where((d < PB.PEG_R - OVERLAP).any(0))[0]:
                    found[("held", i, f"peg{kk}")] = f"the held {name} object hit peg {kk + 1}"
                    if os.environ.get("PEG_DEBUG"):
                        hit = low[d[:, kk] < PB.PEG_R - OVERLAP]
                        print(f"DBG monitor held-peg {kk + 1}: {len(hit)} pts, z {hit[:, 2].min() * 1000:.1f}..{hit[:, 2].max() * 1000:.1f}mm, "
                              f"axis dist {d[:, kk].min() * 1000:.1f}mm, obj centre {np.round(t.obj_T(i)[:3, 3] * 1000, 1)}, "
                              f"all held pts z min {H[:, 2].min() * 1000:.1f}", flush=True)
            for j in live:
                if j == i:
                    continue
                if np.linalg.norm(t.obj_T(j)[:3, 3] - t.obj_T(i)[:3, 3]) > 0.15:
                    continue
                if self.inside_object(H, j, margin=-OVERLAP).any():     # likewise for a neighbour it was resting against
                    found[("held", i, f"obj{j}")] = f"the held {name} object hit the {PB.COLORS[j][0]} object"
        new = set(found) - self.active
        for key in sorted(new, key=str):
            ev = dict(t=round(self.env.t / 60, 2), what=found[key] + (f" (during {t.doing})" if t.doing else ""))
            self.events.append(ev)
            self.on_contact(f"Collision at {ev['t']:.1f} s: {ev['what']}")
        self.active = set(found)

    def summary(self):
        return dict(count=len(self.events), events=self.events[:50],
                    min_arm_clearance_mm=None if not np.isfinite(self.min_clear["arm_arm"]) else round(1000 * self.min_clear["arm_arm"], 1))
