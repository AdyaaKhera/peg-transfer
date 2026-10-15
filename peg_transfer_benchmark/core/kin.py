"""YAM arm kinematics from the URDF (numpy + scipy), for a global IK solver: the step-by-step Jacobian controller
gets stuck when a target is far from the arm's current pose; this solves for joint angles directly, within the joint
limits and from several starting poses, and the arm then moves there along a smooth joint-space path."""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
URDF = os.path.join(P, "third_party/i2rt/i2rt/robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.urdf")


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _origin(el):
    o = el.find("origin")
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
    return T


def _axis_rot(axis, q):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(4)
    R[:3, :3] = np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K
    return R


class Chain:
    """Root (world) -> {side}_gripper through joints 1-6. The URDF's root link sits at the world origin."""

    def __init__(self, side):
        root = ET.parse(URDF).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        by_child = {j.find("child").get("link"): j for j in joints.values()}
        # walk up from the gripper to the root, collecting joints
        chain, link = [], f"{side}_gripper"
        while link in by_child:
            j = by_child[link]
            chain.append(j)
            link = j.find("parent").get("link")
        chain.reverse()
        self.steps = []          # (fixed transform, axis or None, joint index or None)
        self.lo, self.hi = [], []
        for j in chain:
            T = _origin(j)
            if j.get("type") == "revolute":
                ax = [float(v) for v in j.find("axis").get("xyz").split()]
                lim = j.find("limit")
                self.steps.append((T, ax, len(self.lo)))
                self.lo.append(float(lim.get("lower")))
                self.hi.append(float(lim.get("upper")))
            else:
                self.steps.append((T, None, None))
        self.lo, self.hi = np.array(self.lo), np.array(self.hi)
        assert len(self.lo) == 6, f"{side}: expected 6 revolute joints, found {len(self.lo)}"

    def fk(self, q):
        T = np.eye(4)
        for Tf, ax, k in self.steps:
            T = T @ Tf
            if ax is not None:
                T = T @ _axis_rot(ax, q[k])
        return T


class Solver:
    def __init__(self, geom):
        self.chains = {s: Chain(s) for s in ("left", "right")}
        self.pad = geom["pad"].astype(float)
        self.app = geom["approach"].astype(float)
        self.clo = geom["closing"].astype(float)
        self.rng = np.random.default_rng(0)

    def pad_pose(self, side, q):
        T = self.chains[side].fk(q)
        R = T[:3, :3]
        return T[:3, 3] + R @ self.pad, R @ self.app, R @ self.clo

    def _res(self, q, side, target, a_t, c_t, w_rot):
        p, a, c = self.pad_pose(side, q)
        return np.concatenate([p - target, w_rot * (a - a_t), w_rot * (c - c_t)])

    def solve(self, side, target, a_t, c_t, q0, symmetric=True, tries=8):
        """Joint angles putting the finger pads at `target` with approach a_t and closing c_t (or -c_t if symmetric).
        Returns (q, position error m, angle error rad) for the solution nearest q0 among those that converge."""
        ch = self.chains[side]
        lo, hi = ch.lo + 1e-3, ch.hi - 1e-3
        seeds = [np.clip(q0, lo, hi)] + [np.clip(q0 + self.rng.normal(0, 0.6, 6), lo, hi) for _ in range(tries // 2)] + \
                [self.rng.uniform(lo, hi) for _ in range(tries // 2)]
        best = None
        cs = [c_t, -c_t] if symmetric else [c_t]
        # warm start first (fast, and keeps the arm's configuration); random restarts only if that fails
        order = [(c, seeds[0]) for c in cs] + [(c, sd) for sd in seeds[1:] for c in cs]
        for n, (c, sd) in enumerate(order):
            r = least_squares(self._res, sd, bounds=(lo, hi), args=(side, target, a_t, c, 0.25), xtol=1e-10, ftol=1e-12, max_nfev=400)
            p, a, cc = self.pad_pose(side, r.x)
            pe = float(np.linalg.norm(p - target))
            ae = float(math.acos(np.clip(a @ a_t, -1, 1)) + math.acos(np.clip(cc @ c, -1, 1)))
            ok = pe < 0.0015 and ae < math.radians(2)
            jump = float(np.abs(r.x - q0).max())
            key = (not ok, jump if ok else pe + ae)
            if best is None or key < best[0]:
                best = (key, r.x, pe, ae)
            if ok and jump < 0.6 and n >= len(cs) - 1:
                break
        return best[1], best[2], best[3]
