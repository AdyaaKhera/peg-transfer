"""Differential IK for both YAM grippers. Controls the grasp point (the centre of the finger pads, 14 cm out from the
wrist, measured from the meshes by scripts/setup/robot_geometry.py) and the gripper's orientation: its approach axis
(wrist to fingertips) and closing axis (the direction the fingers close along)."""
import math
import os

import numpy as np

import yam_env as Y
from kin import Solver

SIDES = ("left", "right")
GEOM = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "yam_geometry.npz"))
DOWN = np.array([0, 0, -1.0])


def skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def finger_q(gap):
    """Finger joint value for a gap of `gap` metres between the fingertips (each finger moves q, gap = gap0 + 2q)."""
    return float(np.clip((gap - float(GEOM["gap0"])) / 2, 0.0, Y.FINGER_MAX))


def grasp_axes(bar_dir, lean_dir=None, lean=0.0):
    """Gripper axes for grasping a bar so the fingers close across it. With lean > 0 the gripper's body tilts that
    many radians towards `lean_dir` (horizontal), e.g. away from another gripper."""
    a = DOWN.copy()
    if lean and lean_dir is not None:
        n = np.array([lean_dir[0], lean_dir[1], 0.0])
        n /= np.linalg.norm(n)
        a = math.cos(lean) * DOWN - math.sin(lean) * n        # wrist -> fingertips: the wrist sits towards lean_dir
    d = np.asarray(bar_dir, float)
    d = d - a * (d @ a)
    c = np.cross(a, d / np.linalg.norm(d))
    return a, c / np.linalg.norm(c)


class Grippers:
    def __init__(self, env):
        self.env = env
        view = env.robot._articulation_view
        names, J = list(view.body_names), view.get_jacobians()
        off = 1 if J.shape[1] == len(names) - 1 else 0   # fixed base: the root link has no Jacobian row
        self.bi = {s: names.index(f"{s}_gripper") - off for s in SIDES}
        self.pad_local = GEOM["pad"].astype(float)
        self.approach_local = GEOM["approach"].astype(float)
        self.close_local = GEOM["closing"].astype(float)
        self.axes = {s: (DOWN.copy(), np.array([1.0, 0, 0])) for s in SIDES}
        self.exact = {s: False for s in SIDES}
        self.fingers = {s: finger_q(0.02) for s in SIDES}
        self.solver = Solver(GEOM)
        self.max_dq = 0.035            # rad per sim step when following a joint path

    # ---------------------------------------------------------------- state
    def pose(self, side):
        """(grasp point, rotation of the gripper body, gripper body position)."""
        p, q = self.env.grips[side].get_world_poses()
        R = Y.quat_to_mat(np.asarray(q[0], float))
        gp = np.asarray(p[0], float)
        return gp + R @ self.pad_local, R, gp

    def tool(self, side):
        return self.pose(side)[0]

    def current_axes(self, side):
        _, R, _ = self.pose(side)
        return R @ self.approach_local, R @ self.close_local

    def closing_axis(self, side):
        return self.current_axes(side)[1]

    def tool_frame(self, side):
        """4x4 world transform of the grasp point: x = closing axis, z = against the approach (back up the fingers)."""
        t, R, _ = self.pose(side)
        z = -(R @ self.approach_local)
        x = R @ self.close_local
        x = x - z * (x @ z)
        x /= np.linalg.norm(x)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, t
        return T

    def keep_orientation(self, side, exact=True):
        """Hold the gripper's present orientation from now on (used while carrying an object)."""
        self.axes[side] = self.current_axes(side)
        self.exact[side] = exact

    def set_axes(self, side, a, c, exact=False):
        self.axes[side], self.exact[side] = (np.asarray(a, float), np.asarray(c, float)), exact

    # ---------------------------------------------------------------- motion
    def _targets(self):
        env = self.env
        qt = env.target.copy()
        for s in SIDES:
            qt[env.fidx[s]] = self.fingers[s]
        return qt

    def step_toward(self, side, target, damping=0.05):
        env = self.env
        cols = env.idx[side]
        J = env.robot._articulation_view.get_jacobians()[0, self.bi[side]]
        Jv, Jw = J[:3][:, cols], J[3:][:, cols]
        tool, R, gp = self.pose(side)
        a, c = R @ self.approach_local, R @ self.close_local
        a_t, c_t = self.axes[side]
        if not self.exact[side]:
            # A parallel gripper is symmetric: closing along +c_t or -c_t is the same grasp. Use the one the wrist
            # (joint 6, +-120 deg) can reach; turning towards the nearer one can run into the limit and stall.
            r = float(Jw[:, 5] @ a_t) or 1e-6                     # turn about the approach axis per radian of joint 6
            q6, lo6, hi6 = env.q()[cols[5]], env.lo[cols[5]] + 0.05, env.hi[cols[5]] - 0.05
            best = None
            for cand in (c_t, -c_t):
                phi = math.atan2(float(np.cross(c, cand) @ a_t), float(c @ cand))
                q_end = q6 + phi / r
                key = (not (lo6 <= q_end <= hi6), abs(phi))
                if best is None or key < best[0]:
                    best = (key, cand)
            c_t = best[1]
        e_rot = 0.5 * (np.cross(a, a_t) + np.cross(c, c_t))
        k = 0.3
        A = np.vstack([Jv - skew(tool - gp) @ Jw, k * Jw])
        e = np.concatenate([np.asarray(target, float) - tool, k * e_rot])
        dq = np.clip(A.T @ np.linalg.solve(A @ A.T + damping ** 2 * np.eye(6), e), -0.06, 0.06)
        qt = self._targets()
        qt[cols] = env.q()[cols] + dq
        env.step(qt)
        return float(np.linalg.norm(np.asarray(target) - tool)), float(np.linalg.norm(e_rot))

    def go(self, side, target, tol=0.002, max_steps=40, damping=0.05):
        for _ in range(max_steps):
            d, r = self.step_toward(side, target, damping)
            if d < tol and r < 0.03:
                return True
        return False

    def _move_joints(self, side, q_goal, settle=0.01):
        env = self.env
        cols = env.idx[side]
        q0 = env.target[cols].copy()
        n = max(1, int(np.ceil(np.abs(q_goal - q0).max() / self.max_dq)))
        for k in range(1, n + 1):
            qt = self._targets()
            qt[cols] = q0 + (q_goal - q0) * k / n
            env.step(qt)
        for _ in range(150):           # let the drives catch up: they lag on big moves, and the next pose assumes they did
            if np.abs(env.q()[cols] - q_goal).max() < settle:
                break
            env.step(self._targets())

    def line_to(self, side, target, spacing=0.02, tol=0.006, precise=False):
        """Straight-line move of the finger pads, the orientation turning towards self.axes[side] on the way. Each
        waypoint is solved by the global IK (joint limits respected, warm-started from the previous one) and the arm
        follows that joint path; a short Jacobian correction at the end absorbs model-vs-sim differences.
        precise: finish within ~1 mm (final approach of a pick or place)."""
        env = self.env
        cols = env.idx[side]
        a0, b = self.tool(side), np.asarray(target, float)
        at0, ct0 = self.current_axes(side)
        at1, ct1 = self.axes[side]
        if not self.exact[side] and ct0 @ ct1 < 0:
            ct1 = -ct1                                         # symmetric gripper: turn the short way
        n = max(1, int(np.ceil(np.linalg.norm(b - a0) / spacing)), int(math.acos(np.clip(at0 @ at1, -1, 1)) / 0.15) + 1)
        q = env.q()[cols].copy()
        path = []
        for k in range(1, n + 1):
            f = k / n
            at = at0 * (1 - f) + at1 * f
            at /= np.linalg.norm(at)
            ct = ct0 * (1 - f) + ct1 * f
            ct = ct - at * (ct @ at)
            ct /= np.linalg.norm(ct)
            q_k, pe, ae = self.solver.solve(side, a0 + (b - a0) * f, at, ct, q, symmetric=not self.exact[side], tries=4 if k < n else 8)
            if pe > 0.01 or ae > math.radians(8):
                break                                          # beyond reach: stop at the last good waypoint
            jump = np.abs(q_k - q).max()
            if jump > 0.35:
                # a jump to another arm configuration would sweep the arm through space: go in small joint steps only
                # if the solver had to; otherwise stop here
                q_w, pe_w, _ = self.solver.solve(side, a0 + (b - a0) * f, at, ct, q, symmetric=not self.exact[side], tries=0)
                if pe_w < 0.003 and np.abs(q_w - q).max() <= 0.35:
                    q_k = q_w
                else:
                    break
            path.append(q_k)
            q = q_k
        for j, q_k in enumerate(path):
            self._move_joints(side, q_k, settle=0.004 if (precise or j == len(path) - 1) else 0.012)
        if np.linalg.norm(self.tool(side) - b) > 0.015:
            # the straight-line path stopped short: solve for the target pose directly (with restarts) and move there
            # in joint space; if even that fails, stop rather than drag the arm straight at the target
            q_t, pe, ae = self.solver.solve(side, b, at1, ct1, env.q()[cols], symmetric=not self.exact[side], tries=10)
            if pe > 0.003 or ae > math.radians(4):
                return False
            # a straight joint-space swing can take the gripper anywhere in between: trace it with the kinematics first
            # and refuse if the finger pads would dip more than 2 cm below the lower of the two ends
            q_now = env.q()[cols]
            zs = [self.solver.pad_pose(side, q_now + (q_t - q_now) * f)[0][2] for f in np.linspace(0, 1, 25)]
            if os.environ.get("PEG_DEBUG"):
                print(f"DBG line_to {side} fallback: from {np.round(self.tool(side), 3)} to {np.round(b, 3)}, pad z along swing "
                      f"min {min(zs):.3f} ends {zs[0]:.3f}/{zs[-1]:.3f} -> {'REFUSED' if min(zs) < min(zs[0], zs[-1]) - 0.02 else 'swing'}", flush=True)
            if min(zs) < min(zs[0], zs[-1]) - 0.02:
                return False
            self._move_joints(side, q_t, settle=0.004)
        if precise:
            return self.go(side, b, tol=0.001, max_steps=80, damping=0.015)
        return self.go(side, b, tol=tol, max_steps=20)

    def set_fingers(self, side, q, steps=20):
        self.fingers[side] = float(q)
        for _ in range(steps):
            self.env.step(self._targets())

    def hold(self, steps):
        for _ in range(steps):
            self.env.step(self._targets())
