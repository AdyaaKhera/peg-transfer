"""FLS peg transfer on the YAM station: the pegboard, six triangular objects (physics bodies), grasping, the three
actions the model can call (pick, hand_over, place), where each object ended up, and the score.

Board coordinates are millimetres as the model reads them off the overhead photo: x from 0 at the left edge to
810 at the right, y from 0 at the top (far from the robot) to 300 at the bottom (near the robot). The layout follows
the standard FLS / Simulab peg board: the left field has six pegs in two straight columns of three (the objects start
there), the right field six pegs on a ring, so each peg is reached from a different direction. The left arm works the
left field, the right arm the right field.

Sizes: the YAM gripper is far bigger than a laparoscopic grasper (fingers 94 mm long, 24 mm wide at the tip), so the
task is scaled to it: solid 70 mm triangles, 20 mm thick, with a triangular hole (22 mm inscribed circle) on 10 x 40 mm
pegs. A triangle is modelled as three thick walls around the hole, which is what the real ones are.

Grasping: like a surgeon, the gripper pinches a corner of the triangle, closing across it. The robot's URDF has no
collision shapes, so grasps are decided geometrically: when the fingers close on a corner (finger pads on the
corner's centre line, within GRASP_TOL, at a depth where the corner fits between them), the object moves with that
gripper (placed relative to the finger pads every step) until the fingers open; then physics takes over and it falls,
slides down a peg, or lands wherever physics puts it. Contacts that physics cannot see (robot vs anything, a held object vs pegs
and other objects) are caught by collide.Monitor."""
import math
import os

VERSION = "peg-v2"   # v1 (tag peg-v1) was frozen for the first GPT vs Opus series; v2 applies V2_CHANGES.md

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

import yam_env as Y
from ik import DOWN, finger_q, grasp_axes

BOARD_MM = (810.0, 300.0)
BOARD_CENTER = (0.34, -0.305)                                    # world xy, between and in front of the arm bases
BASES = {"left": np.array([0.0, 0.0]), "right": np.array([0.0, -0.61])}
Y.PAPER_CENTER = np.array(BOARD_CENTER)                          # the env's "sheet" is the board
BOARD_RGB = (0.80, 0.81, 0.79)
PEG_R, PEG_H = 0.005, 0.040   # the YAM finger widens to 65 mm higher up: taller pegs would always be in its way
PEG_TOP = Y.PAPER_TOP + PEG_H
OBJ_SIDE, OBJ_HOLE_R, OBJ_H, OBJ_MASS = 0.070, 0.011, 0.020, 0.030   # outer side, hole's inscribed radius, thickness
OBJ_RO = OBJ_SIDE / math.sqrt(3)                                 # centre to a corner tip
OBJ_W = OBJ_SIDE / (2 * math.sqrt(3)) - OBJ_HOLE_R              # wall thickness (9.2 mm)
OBJ_LC = (OBJ_SIDE / (2 * math.sqrt(3)) + OBJ_HOLE_R) * math.sqrt(3)   # side of the walls' centre lines
BAR_L = OBJ_LC + math.sqrt(3) * OBJ_W                           # each wall's length: long enough for sharp outer corners
# Standard FLS layout (checked for reach with the arm kinematics, scripts/setup/reach_check.py): left, two straight columns
# of three; right, a ring of six. Every peg is 25-46 cm from its arm's base.
RING_C, RING_R = (605.0, 150.0), 95.0
PEGS_MM = [(x, y) for x in (130.0, 260.0) for y in (60.0, 150.0, 240.0)] + \
          [(round(RING_C[0] + RING_R * math.cos(math.radians(90 + 60 * k)), 1),
            round(RING_C[1] - RING_R * math.sin(math.radians(90 + 60 * k)), 1)) for k in range(6)]   # 0-5 left, 6-11 right
FIELDS = {"left": (15, 390), "right": (420, 795)}
COLORS = [("red", (0.80, 0.13, 0.12)), ("orange", (0.95, 0.50, 0.08)), ("yellow", (0.93, 0.80, 0.10)),
          ("green", (0.15, 0.62, 0.22)), ("blue", (0.12, 0.35, 0.80)), ("purple", (0.50, 0.20, 0.65))]
PLACE_TOL_MM = 4.0         # v2 release interlock: a place is refused unless the point is this close to a peg's centre
SENSOR_BASE_MM, SENSOR_FRAC, SENSOR_CAP_MM = 1.0, 0.25, 5.0   # v2 sensor: error ~ 1 mm + 25% of the miss, at most 5 mm
STANDING_DEG = 35.0        # a triangle tilted more than this (standing on its edge) cannot be pinched any more
GRASP_TOL = 0.006          # finger pads within 6 mm of a corner's centre line
PINCH_D = 0.012            # where to pinch: finger pads 12 mm in from the corner's tip
PINCH_RANGE = (0.004, 0.022)   # pad depths (from the tip) that hold the corner
FINGER_REACH = 0.0126      # the 24 mm-wide fingertip touches a corner this much deeper than its pad centre
PAD_BELOW_TOP = 0.004      # pads close 4 mm below the triangle's top face, so the fingertips stay off the board
OPEN_GAP = 2 * (PINCH_RANGE[1] + FINGER_REACH) * math.tan(math.radians(30)) + 0.004   # open: any held depth fits
GRIP_SQUEEZE = 0.0006      # closed: this much tighter than the corner's width where the fingers touch it
TRANSFER = (0.31, -0.305, 0.11)   # mid-air transfer point (object centre), over the middle of the board: low enough
                                   # that the holder (gripper vertical, reach ends near 0.17 m) can rise 5 cm off it
RECV_ELEVS = [math.radians(e) for e in (35, 15)]   # receiver's approach, degrees below level, in this order: 35 reaches every
                                                         # ring peg afterwards (scripts/setup/reach_check.py); flatter ones only if 35 would clash
ARM_GAP_MIN = 0.006        # the receiver backs out of an approach that brings any part of the two arms closer than this
PRESENT_OFFSETS = [0.0, math.radians(40)]   # holder turns a free corner to the receiver, then also +40 deg (v2: fewer tries; a failed hand-over says how to fix it)
SAFE_Z = 0.12              # carry height: a held object's bottom clears the 40 mm pegs by 8 cm
PARK = {"left": (0.28, 0.17, 0.28), "right": (0.28, -0.78, 0.28)}   # beside the board on each arm's own side, out of the photo
RENDER_GAMMA = 1.6         # the renderer over-exposes colours; pre-darken materials so they look like their names
CUTOFF_S = 300.0           # the FLS peg transfer's time limit
MONITOR_LINKS = ["link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right"]


# ---------------------------------------------------------------- coordinates
def board_to_world(xy_mm):
    p = np.asarray(xy_mm, float).reshape(-1, 2)
    uv = np.stack([p[:, 0] / 1000 - BOARD_MM[0] / 2000, BOARD_MM[1] / 2000 - p[:, 1] / 1000], 1)
    return Y.paper_to_world(uv)


def world_to_board(xy):
    uv = Y.world_to_paper(np.asarray(xy, float).reshape(-1, 2))
    return np.stack([(uv[:, 0] + BOARD_MM[0] / 2000) * 1000, (BOARD_MM[1] / 2000 - uv[:, 1]) * 1000], 1)


def field_of(x_mm):
    return "left" if x_mm < BOARD_MM[0] / 2 else "right"


def mat_to_quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def corner_width(depth):
    """Width of a 60-degree corner `depth` metres in from its tip."""
    return 2 * max(0.0, depth) * math.tan(math.radians(30))


def grip_gap(pad_depth):
    return max(0.002, corner_width(pad_depth + FINGER_REACH) - GRIP_SQUEEZE)


def _bars_local():
    rc = OBJ_LC / math.sqrt(3)
    V = [np.array([rc * math.cos(math.radians(90 + 120 * k)), rc * math.sin(math.radians(90 + 120 * k)), 0.0]) for k in range(3)]
    out = []
    for k in range(3):
        a, b = V[k], V[(k + 1) % 3]
        out.append(((a + b) / 2, (b - a) / np.linalg.norm(b - a), a, b))
    return out


BARS = _bars_local()


def board_texture(px_per_mm=3.0, seed=5):
    rng = np.random.default_rng(seed)
    W, H = int(BOARD_MM[0] * px_per_mm), int(BOARD_MM[1] * px_per_mm)
    img = np.empty((H, W, 3), np.float32)
    img[:] = BOARD_RGB
    small = Image.fromarray((rng.random((H // 40 + 2, W // 40 + 2)) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
    img *= (1 + 0.025 * (np.asarray(small, np.float32) / 127.5 - 1))[..., None]
    # faint printed field outlines, like the FLS board's two halves
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    k = px_per_mm
    for x0, x1 in FIELDS.values():
        d.rectangle([x0 * k, 15 * k, x1 * k, (BOARD_MM[1] - 15) * k], outline=(150, 152, 150), width=max(1, int(k)))
    return np.asarray(im, np.float32) / 255


# ---------------------------------------------------------------- the task
class PegTransfer:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.sensor_rng = np.random.default_rng(seed + 1000)   # noise on reported offsets (same seed, same noise)
        self.held = {"left": None, "right": None}       # side -> (object index, 4x4 offset from the grasp frame)
        self.engaged = {"left": None, "right": None}    # object a gripper's fingers are allowed to touch right now
        self.on_moment = None                           # optional callback(name) at notable moments (tests, video)
        self.doing = ""                                 # the action under way, for collision reports
        self.objs = None
        self.n = len(COLORS)
        self.log = []                                   # every action's outcome
        self.lost = set()
        self.crossed = [None] * self.n                  # "left->right" etc. once it has been passed across in mid-air
        self.picked_by = [None] * self.n
        self.transfers = {"left->right": [], "right->left": []}
        self.drops = 0
        self.phase = 1
        self.actions = []                               # every action's measurements, for the reward (reward.py)
        self._last = {}

    # ---------- scene (called by YamDrawEnv before world.reset)
    def build(self, env):
        from isaacsim.core.api.objects import FixedCylinder
        from isaacsim.core.prims import RigidPrim
        from isaacsim.sensors.camera import Camera
        self.env, stage = env, env.stage
        for i, (x, y) in enumerate(PEGS_MM):
            w = board_to_world([(x, y)])[0]
            env.world.scene.add(FixedCylinder(f"/World/pegs/p{i}", name=f"peg{i}", position=np.array([w[0], w[1], Y.PAPER_TOP + PEG_H / 2]),
                                              radius=PEG_R, height=PEG_H, color=np.array([0.03, 0.03, 0.035])))
        paths = []
        start = list(range(6))                          # every object starts on a left peg
        for i, (name, rgb) in enumerate(COLORS):
            x, y = PEGS_MM[start[i]]
            w = board_to_world([(x, y)])[0]
            path = f"/World/objs/o{i}"
            xf = UsdGeom.Xform.Define(stage, path)
            prim = xf.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(OBJ_MASS)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSolverPositionIterationCountAttr(16)
            yaw = self.rng.uniform(0, 2 * math.pi / 3)
            xf.AddTranslateOp().Set(Gf.Vec3d(float(w[0]), float(w[1]), Y.PAPER_TOP + OBJ_H / 2 + 0.001))
            xf.AddOrientOp().Set(Gf.Quatf(math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))
            mat = env._material(f"obj_{name}", [c ** RENDER_GAMMA for c in rgb], 0.5, 0.0, srgb=True)
            for k, (mid, d, _, _) in enumerate(BARS):
                c = UsdGeom.Cube.Define(stage, f"{path}/bar{k}")
                cx = UsdGeom.Xformable(c)
                cx.AddTranslateOp().Set(Gf.Vec3d(*map(float, mid)))
                cx.AddRotateZOp().Set(math.degrees(math.atan2(d[1], d[0])))
                cx.AddScaleOp().Set(Gf.Vec3f(BAR_L / 2, OBJ_W / 2, OBJ_H / 2))
                UsdPhysics.CollisionAPI.Apply(c.GetPrim())
                pc = PhysxSchema.PhysxCollisionAPI.Apply(c.GetPrim())
                pc.CreateContactOffsetAttr(0.002)
                pc.CreateRestOffsetAttr(0.0)
                env._bind(c.GetPrim(), mat)
            paths.append(path)
        self.paths = paths
        self.objs = env.world.scene.add(RigidPrim(paths, name="peg_objects"))
        # overhead camera for the model's photo: orthographic, so pegs and objects sit exactly where they are
        # the arms' links, for the collision monitor (the base links don't move)
        self.link_views = {s: env.world.scene.add(RigidPrim([f"/World/yam/{s}_{n}" for n in MONITOR_LINKS], name=f"{s}_links"))
                           for s in ("left", "right")}
        self.cam = Camera("/World/board_cam", name="board_cam", resolution=(1350, 500),
                          position=np.array([*Y.PAPER_CENTER, 1.0]),
                          orientation=Y.euler_angles_to_quat(np.array([0, math.pi / 2, 0])))

    def after_reset(self):
        self.cam.initialize()
        cam = UsdGeom.Camera(self.env.stage.GetPrimAtPath("/World/board_cam"))
        cam.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)
        cam.GetHorizontalApertureAttr().Set(BOARD_MM[0] / 100 * 1.0)   # tenths of a metre: exactly the board's width
        cam.GetVerticalApertureAttr().Set(BOARD_MM[1] / 100 * 1.0)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 5.0))

    # ---------- per-step: carry held objects with their grippers
    def carry(self, grippers):
        for side, h in self.held.items():
            if h is None:
                continue
            i, off = h
            T = grippers.tool_frame(side) @ off
            self.objs.set_world_poses(T[:3, 3][None], mat_to_quat(T[:3, :3])[None], indices=[i])
            self.objs.set_velocities(np.zeros((1, 6)), indices=[i])

    # ---------- object geometry in the world
    def poses(self):
        p, q = self.objs.get_world_poses()
        return np.asarray(p, float), np.asarray(q, float)

    def obj_T(self, i):
        p, q = self.poses()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Y.quat_to_mat(q[i]), p[i]
        return T

    def bars_world(self, i):
        T = self.obj_T(i)
        out = []
        for mid, d, a, b in BARS:
            m = T[:3, :3] @ mid
            n = np.array([m[0], m[1], 0.0])
            out.append(dict(mid=m + T[:3, 3], dir=T[:3, :3] @ d, a=T[:3, :3] @ a + T[:3, 3], b=T[:3, :3] @ b + T[:3, 3],
                            out=n / max(np.linalg.norm(n), 1e-9), center=T[:3, 3]))
        return out

    def sense(self, dx, dy):
        """A measured offset (mm), like a camera-based sensor: the true one plus noise that grows with the size of the
        miss (about 1 mm plus a quarter of the distance, at most 5 mm per axis), so it is a sensor, not an oracle."""
        bound = min(SENSOR_CAP_MM, SENSOR_BASE_MM + SENSOR_FRAC * math.hypot(dx, dy))
        n = np.clip(self.sensor_rng.normal(0.0, bound / 2, 2), -bound, bound)
        return dx + float(n[0]), dy + float(n[1])

    def tilt_deg(self, i):
        z = self.obj_T(i)[:3, 2]
        return math.degrees(math.acos(min(1.0, abs(float(z[2])))))

    def corners_world(self, i):
        """The triangle's three corners: tip (outer corner, at mid-thickness), u (unit, centre -> tip, in the object's
        plane), z (the object's normal)."""
        T = self.obj_T(i)
        out = []
        for _, _, a, _ in BARS:
            u = T[:3, :3] @ (a / np.linalg.norm(a))
            out.append(dict(tip=T[:3, 3] + OBJ_RO * u, u=u, z=T[:3, 2].copy(), center=T[:3, 3].copy()))
        return out

    def nearest_corner(self, point, exclude=()):
        """(object, corner, xy distance m) for the corner centre line (tip to 30 mm in) nearest `point`."""
        best = None
        for i in range(self.n):
            if i in exclude or i in self.lost:
                continue
            for k, cor in enumerate(self.corners_world(i)):
                dist, _ = self._seg_dist_xy(np.asarray(point, float), cor["tip"] - 0.030 * cor["u"], cor["tip"])
                if best is None or dist < best[2]:
                    best = (i, k, dist)
        return best

    def _grasped_corner(self, g, side, exclude=()):
        """(object, corner, off-centre m, depth m from the tip, holds) for the corner nearest the finger pads: holds when
        the pads sit on its centre line at a depth where the corner fits between the fingers, closing across it."""
        tool, closing = g.tool(side), g.closing_axis(side)
        best = None
        for i in range(self.n):
            if i in exclude or i in self.lost:
                continue
            for k, cor in enumerate(self.corners_world(i)):
                r = tool - cor["tip"]
                depth = -float(r @ cor["u"])
                height = float(r @ cor["z"])
                lat = r + depth * cor["u"] - height * cor["z"]
                off = float(np.linalg.norm(lat))
                if best is None or off + abs(depth - PINCH_D) < best[2] + abs(best[3] - PINCH_D):
                    holds = (PINCH_RANGE[0] <= depth <= PINCH_RANGE[1] and off < GRASP_TOL and abs(height) < OBJ_H / 2 + 0.004
                             and abs(float(closing @ cor["u"])) < 0.3 and abs(float(closing @ cor["z"])) < 0.3)
                    best = (i, k, off, depth, holds)
        return best

    @staticmethod
    def _seg_dist_xy(p, a, b):
        ab, ap = (b - a)[:2], (np.asarray(p, float)[:2] - a[:2])
        t = float(np.clip(ap @ ab / max(ab @ ab, 1e-12), 0, 1))
        return float(np.linalg.norm(ap - t * ab)), t

    def attach(self, side, i, grippers):
        # The object stays an ordinary rigid body and carry() moves it with the gripper every step. (A kinematic body
        # moved this way is not redrawn: it stayed on its peg on camera and appeared to jump when released.)
        self.held[side] = (i, np.linalg.inv(grippers.tool_frame(side)) @ self.obj_T(i))
        self.engaged[side] = i
        grippers.keep_orientation(side, exact=True)    # carrying: the object turns with the gripper, so no flips

    def release(self, side):
        i = self.held[side][0]
        self.held[side] = None
        self.engaged[side] = None
        self.objs.set_velocities(np.zeros((1, 6)), indices=[i])
        return i

    def holding(self, side):
        return None if self.held[side] is None else self.held[side][0]

    # ---------- where things are
    def where(self, i):
        p = self.poses()[0][i]
        x, y = world_to_board(p[:2])[0]
        if p[2] < Y.PAPER_TOP - 0.01 or not (0 <= x <= BOARD_MM[0] and 0 <= y <= BOARD_MM[1]):
            return dict(state="off", xy=(x, y))
        d = [math.hypot(x - px, y - py) for px, py in PEGS_MM]
        k = int(np.argmin(d))
        if p[2] < Y.PAPER_TOP + 0.02 and self.peg_through_hole(i, k):
            return dict(state="peg", peg=k, field=field_of(PEGS_MM[k][0]), xy=(x, y))
        if d[k] < 45 and p[2] > 0.04:
            return dict(state="peg_top", peg=k, xy=(x, y))
        return dict(state="board", field=field_of(x), xy=(x, y))

    def peg_through_hole(self, i, k):
        """Does peg k pass through object i's triangular hole? The hole is triangular, so a peg can sit up to ~37 mm
        from the centre (in a corner): test the peg's position against the hole's three sides, not a distance."""
        T = self.obj_T(i)
        peg = board_to_world([PEGS_MM[k]])[0]
        local = T[:3, :3].T @ (np.array([peg[0], peg[1], T[2, 3]]) - T[:3, 3])
        r_in = OBJ_LC / (2 * math.sqrt(3)) - OBJ_W / 2               # the hole's inradius (to the bars' inner faces)
        for mid, _, _, _ in BARS:
            n = mid[:2] / np.linalg.norm(mid[:2])                    # outward normal of this side of the hole
            if local[:2] @ n > r_in - PEG_R + 0.0015:
                return False
        return True

    def free_pegs(self, field):
        taken = {self.where(i).get("peg") for i in range(self.n) if self.where(i)["state"] == "peg"}
        return [k for k, (x, _) in enumerate(PEGS_MM) if field_of(x) == field and k not in taken]

    def summary(self):
        lines = []
        for i, (name, _) in enumerate(COLORS):
            h = [s for s in ("left", "right") if self.holding(s) == i]
            if h:
                lines.append(f"{name}: in the {h[0]} gripper")
                continue
            w = self.where(i)
            desc = {"peg": f"on a {w.get('field')} peg", "peg_top": "resting on top of a peg (not over it)",
                    "board": f"lying on the board, {w.get('field')} field", "off": "fell off the board (lost)"}[w["state"]]
            lines.append(f"{name}: {desc}")
        return lines

    # ---------- the actions
    def lift(self, g, side):
        """Straight up to carry height (a held object's bottom at SAFE_Z)."""
        p = g.tool(side)
        z = SAFE_Z
        if self.holding(side) is not None:
            z = SAFE_Z + (p[2] - (self.obj_T(self.holding(side))[2, 3] - OBJ_H / 2))
        if p[2] < z - 0.002:
            g.line_to(side, [p[0], p[1], z])

    def park(self, g, side):
        self.lift(g, side)
        t = g.tool(side)
        if t[2] < PARK[side][2] - 0.01:
            g.line_to(side, [t[0], t[1], PARK[side][2]])        # straight up first, then across: never sweep low
        g.keep_orientation(side, exact=False)
        g.line_to(side, PARK[side])                            # across first, wrist as it is
        g.set_axes(side, DOWN, np.array([1.0, 0, 0]))
        g.line_to(side, PARK[side])                            # then turn it, at the parking spot, away from everything

    def _approach(self, g, side, point, a):
        """Come in along the gripper's (leaning) approach axis: from carry height straight down that line."""
        d = max(0.06, (SAFE_Z - point[2]) / max(0.3, -a[2]))
        start = point - a * d
        self.lift(g, side)
        g.line_to(side, start)
        g.go(side, start, max_steps=60)          # finish turning before going in
        return start

    def wrong_half(self, side, x_mm, what):
        """v2: each arm works only its own half of the board; refuse at once (no motion) and say which arm to use."""
        if field_of(x_mm) == side:
            return None
        other = "right" if side == "left" else "left"
        return (f"{what}: refused, the {side} arm works only the {side} field and x = {x_mm:g} mm is in the {other} field "
                f"(the fields meet at x = {BOARD_MM[0] / 2:.0f} mm); use the {other} arm there")

    def pick(self, g, side, x_mm, y_mm):
        self._last = {}
        refuse = self.wrong_half(side, x_mm, f"pick({side}, {x_mm:g}, {y_mm:g})")
        if refuse:
            self.actions.append(dict(kind="pick", ok=False, refused=True))
            return False, refuse
        ok, msg = self._pick(g, side, x_mm, y_mm)
        self.actions.append(dict(kind="pick", ok=ok, **self._last))
        return ok, msg

    def hand_over(self, g):
        ok, msg = self._hand_over(g)
        self.actions.append(dict(kind="hand_over", ok=ok))
        return ok, msg

    def place(self, g, x_mm, y_mm, height_mm=50.0):
        self._last = {}
        holders = [s_ for s_ in ("left", "right") if self.holding(s_) is not None]
        if len(holders) == 1:
            refuse = self.wrong_half(holders[0], x_mm, f"place({x_mm:g}, {y_mm:g})")
            if refuse:
                self.actions.append(dict(kind="place", ok=False, refused=True, aim_mm=None))
                return False, refuse + f"; the {COLORS[self.holding(holders[0])][0]} object is still in the {holders[0]} gripper"
        ok, msg = self._place(g, x_mm, y_mm, height_mm)
        aim = min(math.hypot(x_mm - px, y_mm - py) for px, py in PEGS_MM)   # requested point vs the nearest peg centre
        # (an interlock refusal keeps its aim error: the model really aimed that far from the peg)
        self.actions.append(dict(kind="place", ok=ok, aim_mm=round(aim, 1), kept="still in the" in msg, **self._last))
        return ok, msg

    def _pick(self, g, side, x_mm, y_mm):
        self.doing = "pick"
        if self.holding(side) is not None:
            return False, f"the {side} gripper is already holding the {COLORS[self.holding(side)][0]} object"
        held = {h[0] for h in self.held.values() if h}
        w = board_to_world([(x_mm, y_mm)])[0]
        near = self.nearest_corner(w, exclude=held)
        # The model's point decides where the fingers go. Like a surgeon's wrist, the gripper turns to close across
        # whichever corner is nearest that point, just below the triangle's top face.
        if near and near[2] < 2 * GRASP_TOL and self.tilt_deg(near[0]) > STANDING_DEG:
            self._last = dict(refused=True)
            return False, (f"pick({side}, {x_mm:g}, {y_mm:g}): the {COLORS[near[0]][0]} triangle there is standing on its edge "
                           f"(tilted {self.tilt_deg(near[0]):.0f} degrees) and cannot be picked up any more; leave it and go on")
        if near and near[2] < 2 * GRASP_TOL:
            cor = self.corners_world(near[0])[near[1]]
            nrm = cor["z"] if cor["z"][2] > 0 else -cor["z"]  # the triangle's upward normal
            target_obj = near[0]
            if nrm[2] > math.cos(math.radians(8)):            # lying flat: vertical, closing across the corner
                a, c = grasp_axes(cor["u"])
                point = np.array([w[0], w[1], cor["tip"][2] + OBJ_H / 2 - PAD_BELOW_TOP])
            else:                                             # v2: tilted (e.g. resting against a peg): pinch square to it
                a = -nrm
                c = np.cross(a, cor["u"])
                c /= np.linalg.norm(c)
                cen = cor["center"]
                z = cen[2] - (nrm[0] * (w[0] - cen[0]) + nrm[1] * (w[1] - cen[1])) / nrm[2]   # its mid-plane under (x, y)
                point = np.array([w[0], w[1], z]) + nrm * (OBJ_H / 2 - PAD_BELOW_TOP)
        else:
            a, c, target_obj = DOWN, g.closing_axis(side), None
            point = np.array([w[0], w[1], Y.PAPER_TOP + OBJ_H - PAD_BELOW_TOP])
        g.set_fingers(side, finger_q(OPEN_GAP), steps=10)
        g.set_axes(side, a, c)
        start = self._approach(g, side, point, a)
        dbg = os.environ.get("PICK_DEBUG")
        if dbg:
            print(f"PICKDBG point {np.round(point, 3)} start {np.round(start, 3)} tool {np.round(g.tool(side), 3)} a {np.round(a, 2)}", flush=True)
        self.engaged[side] = target_obj
        g.line_to(side, point, spacing=0.004, precise=True)
        if dbg:
            print(f"PICKDBG after descent tool {np.round(g.tool(side), 3)}", flush=True)
        pre = self._grasped_corner(g, side, exclude=held)
        g.set_fingers(side, finger_q(grip_gap(pre[3]) if pre else 0.002), steps=20)
        got = self._grasped_corner(g, side, exclude=held)
        ok = bool(got and got[4])
        if got:
            self._last = dict(off_mm=round(got[2] * 1000, 1), depth_mm=round(got[3] * 1000, 1))
        aa = g.current_axes(side)[0]
        print(f"PICK {side} at ({x_mm:g}, {y_mm:g}): pad off by {np.linalg.norm(g.tool(side) - point) * 1000:.1f} mm, "
              f"tilt {math.degrees(math.acos(np.clip(aa @ a, -1, 1))):.1f} deg, grasp {'ok' if ok else 'missed'}", flush=True)
        if ok:
            i = got[0]
            self.attach(side, i, g)
            w0 = self.where(i)
            # picking it up again on the side it was passed to (after a drop) keeps the transfer; anything else restarts it
            if not (self.crossed[i] and w0.get("field") == self.crossed[i].split("->")[1]):
                self.crossed[i] = None
            self.picked_by[i] = side
            self.straight(g, side, PEG_H + 0.03)   # exactly straight up until it is clear of the peg and its neighbours
            self.lift(g, side)
        else:
            g.set_fingers(side, finger_q(OPEN_GAP), steps=8)
            g.line_to(side, start, spacing=0.006)   # back out the way it came
        if not ok:
            self.engaged[side] = None
            why = ""
            if got:
                cors = self.corners_world(got[0])
                cor = cors[got[1]]
                gx, gy = world_to_board(pinch_point(cor))[0]
                others = [world_to_board(pinch_point(c_))[0] for k_, c_ in enumerate(cors) if k_ != got[1]]
                nrm = cor["z"] if cor["z"][2] > 0 else -cor["z"]
                tilt = math.degrees(math.acos(float(np.clip(nrm[2], -1, 1))))
                dx, dy = gx - x_mm, gy - y_mm
                dirs = " and ".join(t for t in ((f"{abs(dx):.0f} mm {'right' if dx > 0 else 'left'}" if abs(dx) >= 1 else ""),
                                                (f"{abs(dy):.0f} mm {'down' if dy > 0 else 'up'}" if abs(dy) >= 1 else "")) if t)
                # v2: relative feedback only (how far and which way), never the true position itself
                if math.hypot(dx, dy) < 3:                    # the aim was right: say what else stopped the grasp
                    why = (f"; your aim was on the {COLORS[got[0]][0]} triangle's grip point, but the fingers could not close "
                           f"across that corner" + (f" (the triangle is tilted {tilt:.0f} degrees)" if tilt > 8 else "")
                           + "; try another corner")
                else:
                    mx, my = self.sense(dx, dy)
                    why = (f"; the nearest grip point (the {COLORS[got[0]][0]} triangle's corner, 12 mm in from its tip on its "
                           f"centre line) is {offset_words(mx, my)} of where you aimed")
            return False, f"pick({side}, {x_mm:g}, {y_mm:g}): the fingers closed on nothing{why}"
        self.lift(g, side)
        return True, f"pick({side}, {x_mm:g}, {y_mm:g}): grasped the {COLORS[i][0]} object"

    def _hand_over(self, g):
        self.doing = "hand-over"
        holders = [s for s in ("left", "right") if self.holding(s) is not None]
        if len(holders) != 1:
            return False, "hand_over needs exactly one gripper holding an object"
        h = holders[0]
        r = "right" if h == "left" else "left"
        i = self.holding(h)
        toward_r = np.array([0, -1.0, 0]) if r == "right" else np.array([0, 1.0, 0])
        self.doing = "hand-over: holder lifting and turning it"
        # 1. The holder (gripper vertical, from above) lifts the object into open air above the centre of the board and
        #    turns it so a free corner points at the receiver.
        self.lift(g, h)
        cors = self.corners_world(i)
        hb = self._grasped_corner(g, h)[1]                        # the corner the holder pinches
        turns = []
        for k in range(3):                                        # turn a free corner to point at the receiver
            if k == hb:
                continue
            n = cors[k]["u"][:2]
            turns.append(math.atan2(n[0] * toward_r[1] - n[1] * toward_r[0], n @ toward_r[:2]))
        got = None
        # Try presenting each free corner to the receiver (smallest turn first): each puts the holder's wrist and elbow
        # somewhere else, so if every approach of the receiver would bring the arms too close, the next one may not.
        # v2: also move the hand-over point (5 cm towards the holder's own side, or 4 cm further from the bases), which
        # changes where both wrists sit; at most 6 presentations x 2 approach angles
        t0, t1 = sorted(turns, key=abs)[0], sorted(turns, key=abs)[-1]
        aside, ahead = -toward_r * 0.05, np.array([0.04, 0.0, 0.0])
        presents = [(t0, 0 * aside), (t0, aside), (t1, 0 * aside), (t0 + PRESENT_OFFSETS[1], 0 * aside), (t0, ahead), (t1, aside)]
        for p_try, (turn, shift) in enumerate(presents):
            self.doing = "hand-over: holder presenting it"
            a_h, c_h = g.current_axes(h)
            Rz = np.array([[math.cos(turn), -math.sin(turn), 0], [math.sin(turn), math.cos(turn), 0], [0, 0, 1]])
            if p_try:                                            # turn back by the first presentation's turn first
                Rz = Rz @ Rz_prev.T
            g.set_axes(h, Rz @ a_h, Rz @ c_h, exact=True)
            T, pad = self.obj_T(i), g.tool(h)
            hold_at = np.array(TRANSFER) + shift + Rz @ (pad - T[:3, 3])
            g.line_to(h, hold_at)
            g.go(h, hold_at, max_steps=100)
            g.hold(10)
            Rz_prev = np.array([[math.cos(turn), -math.sin(turn), 0], [math.sin(turn), math.cos(turn), 0], [0, 0, 1]])
            # 2. The receiver comes in from its own side, angled down, and pinches the corner that points at it. Its
            #    body points away from the object, so the two arms meet it from opposite sides; every move is checked
            #    for clearance between the arms and abandoned if they would come within ARM_GAP_MIN.
            cors = self.corners_world(i)
            k = max((k for k in range(3) if k != hb), key=lambda k: float(cors[k]["u"] @ toward_r))
            for elev in RECV_ELEVS:
                self.doing = "hand-over: receiver reaching in"
                self.park(g, r)                                  # every attempt starts from the folded-out pose
                self.dbg(f"receiver {r} attempt: presentation {p_try} corner {k} elev {math.degrees(elev):.0f}, parked at {np.round(g.tool(r), 3)}", side=r)
                cor = self.corners_world(i)[k]
                u = cor["u"]
                point = cor["tip"] - PINCH_D * u                  # on the corner's centre line, at mid-thickness
                a = math.cos(elev) * -u + math.sin(elev) * DOWN    # come in towards the corner, angled down
                c = np.cross(DOWN, u)
                c /= np.linalg.norm(c)                            # fingers close across the corner (horizontal)
                g.set_fingers(r, finger_q(OPEN_GAP), steps=10)
                g.set_axes(r, a, c)
                start = point - a * 0.12
                above = start + np.array([0, 0, 0.05])
                if not (self.approach_clear(g, r, g.tool(r), above, step=0.04) and self.approach_clear(g, r, above, start)):
                    continue                                     # the arms would meet on the way: next angle
                g.go(r, start, max_steps=80)
                self.engaged[r] = i
                self.doing = "hand-over: receiver closing"
                if not self.approach_clear(g, r, start, point):  # would come too close to the other arm: back out
                    g.line_to(r, start)
                    self.engaged[r] = None
                    continue
                pre = self._grasped_corner(g, r, exclude=[j for j in range(self.n) if j != i])
                g.set_fingers(r, finger_q(grip_gap(pre[3]) if pre else 0.002), steps=20)
                got = self._grasped_corner(g, r, exclude=[j for j in range(self.n) if j != i])
                if os.environ.get("PICK_DEBUG"):
                    print(f"HANDDBG corner {k} elev {math.degrees(elev):.0f} point {np.round(point, 3)} tool {np.round(g.tool(r), 3)} got {got}", flush=True)
                if got and got[0] == i and got[4] and got[1] != hb:
                    break
                g.set_fingers(r, finger_q(OPEN_GAP), steps=8)
                g.line_to(r, start)
                self.engaged[r] = None
                got = None
            if got is not None:
                break
        if got is None:
            self.park(g, r)
            return False, (f"hand_over: the {r} gripper could not get hold of the {COLORS[i][0]} object without the two arms "
                           f"touching, with the corner the {h} gripper is holding; it is still in the {h} gripper. Repeating "
                           f"hand_over does the same thing: place it back on a free {h} peg and pick it again by a corner "
                           f"that points away from the {r} arm")
        # 3. Both grippers hold it for a moment; then the holder lets go and lifts straight away.
        self.doing = "hand-over: both holding"
        self.held[h] = None
        self.attach(r, i, g)
        self.crossed[i] = f"{h}->{r}"
        g.hold(12)
        if self.on_moment:
            self.on_moment("both_holding")
        g.hold(24)
        self.doing = "hand-over: holder opening"
        u_h = self.corners_world(i)[hb]["u"]
        g.set_fingers(h, finger_q(min(OPEN_GAP, g.fingers[h] * 2 + 0.006)), steps=15)   # just enough to let go
        self.doing = "hand-over: holder backing away"
        self.dbg("retreat start", side=h, i=i)
        g.line_to(h, g.tool(h) + u_h * 0.04, spacing=0.004)                   # slide off its corner, away from both
        self.dbg("retreat after slide", side=h, i=i)
        self.straight(g, h, 0.06)                                              # then straight up
        away = -toward_r
        for _ in range(3):                                                     # then out to its own side until clear
            if self.clear_of(h, i):
                break
            g.line_to(h, g.tool(h) + away * 0.06 + np.array([0, 0, 0.02]))
        self.engaged[h] = None
        self.dbg("retreat done, parking", side=h, i=i)
        self.doing = "hand-over: holder parking"
        self.park(g, h)
        self.dbg("parked", side=h, i=i)
        return True, f"hand_over: the {COLORS[i][0]} object is now in the {r} gripper"

    def level(self, g, side):
        """Turn the gripper so the object it holds lies flat again (it is tilted after a hand-over)."""
        i = self.holding(side)
        R_o = self.obj_T(i)[:3, :3]
        if R_o[2, 2] > math.cos(math.radians(3)):
            return
        Tt = g.tool_frame(side)
        R_off = Tt[:3, :3].T @ R_o
        x = R_o[:, 0].copy()
        x[2] = 0
        x /= np.linalg.norm(x)
        z = np.array([0, 0, 1.0])
        R_lvl = np.stack([x, np.cross(z, x), z], 1)
        R_t = R_lvl @ R_off.T
        g.set_axes(side, -R_t[:, 2], R_t[:, 0], exact=True)
        g.go(side, g.tool(side) + np.array([0, 0, 0.02]), max_steps=150)

    def arms_gap(self, n=120):
        """Smallest distance (m) between any two points of the two arms (links 2 to fingertips), subsampled."""
        from ik import GEOM
        if not hasattr(self, "_gap_pts"):
            rng = np.random.default_rng(1)
            self._gap_pts = {nm: GEOM[f"pts_{nm}"][rng.choice(len(GEOM[f"pts_{nm}"]), min(n, len(GEOM[f"pts_{nm}"])), replace=False)]
                             for nm in MONITOR_LINKS if nm != "link1"}
        P = {}
        for side in ("left", "right"):
            p, q = self.link_views[side].get_world_poses()
            P[side] = np.vstack([self._gap_pts[nm] @ Y.quat_to_mat(np.asarray(q[j], float)).T + np.asarray(p[j], float)
                                 for j, nm in enumerate(MONITOR_LINKS) if nm != "link1"])
        return float(np.sqrt(((P["left"][:, None, :] - P["right"][None, :, :]) ** 2).sum(-1)).min())

    def approach_clear(self, g, side, start, point, step=0.01):
        """Move the finger pads from start to point in short legs, checking the arms' clearance after each; stop and
        return False the moment the two arms come within ARM_GAP_MIN."""
        start, point = np.asarray(start, float), np.asarray(point, float)
        n = max(1, int(np.ceil(np.linalg.norm(point - start) / step)))
        for k in range(1, n + 1):
            g.line_to(side, start + (point - start) * k / n, spacing=0.004, precise=(k == n))
            if os.environ.get("PEG_DEBUG"):
                P = self.hand_points(side)
                print(f"DBG leg {k}/{n} {side}: target {np.round(start + (point - start) * k / n, 3)} at {np.round(g.tool(side), 3)} "
                      f"hand low z {P[:, 2].min():.3f}", flush=True)
            if self.arms_gap() < ARM_GAP_MIN or self.safe_drop(side, 0.02) < 0.005:   # near the other arm, a peg or the board
                return False
        return True

    def dbg(self, tag, side=None, i=None, target=None):
        """PEG_DEBUG=1: print where things are (diagnosing contacts)."""
        if not os.environ.get("PEG_DEBUG"):
            return
        out = [f"DBG {self.env.t / 60:7.2f}s {tag}"]
        if i is not None:
            U = self.underside_points(i)
            out.append(f"obj_bottom {self.obj_bottom(i) * 1000:.1f}mm (underside min {U[:, 2].min() * 1000:.1f}, peg top {PEG_TOP * 1000:.1f})")
            if target is not None:
                out.append(f"hole off {np.linalg.norm(self.obj_T(i)[:2, 3] - target[:2]) * 1000:.1f}mm, "
                           f"nearest underside pt to peg axis {np.linalg.norm(U[:, :2] - target[:2], axis=1).min() * 1000:.1f}mm")
            if side is not None and self.holding(side) != i:
                P = self.hand_points(side)
                c = self.obj_T(i)[:3, 3]
                out.append(f"{side} hand to obj: min xy {np.linalg.norm(P[:, :2] - c[:2], axis=1).min() * 1000:.0f}mm, "
                           f"hand low z {P[:, 2].min() * 1000:.0f}mm vs obj top {(c[2] + OBJ_H / 2) * 1000:.0f}mm")
        if side is not None:
            out.append(f"arms gap {self.arms_gap() * 1000:.1f}mm")
        print(" | ".join(out), flush=True)

    def hand_points(self, side):
        """World points on the wrist, gripper and fingertips (the same meshes the collision monitor uses)."""
        from ik import GEOM
        p, q = self.link_views[side].get_world_poses()
        pts = []
        for j, n in enumerate(MONITOR_LINKS):
            if n in ("gripper", "tip_left", "tip_right", "link5", "link4"):
                R = Y.quat_to_mat(np.asarray(q[j], float))
                pts.append(GEOM[f"pts_{n}"] @ R.T + np.asarray(p[j], float))
        return np.vstack(pts)

    def straight(self, g, side, dz, step=0.002):
        """Move the finger pads straight up (dz > 0) or down (dz < 0) in small Jacobian steps, held over the same
        point. Going down, it checks the gripper's real geometry against the pegs and the board before every step and
        stops short rather than touch one. Returns how far it actually moved (m, unsigned)."""
        p0 = g.tool(side)
        moved = 0.0
        while moved < abs(dz) - 1e-4:
            rem = abs(dz) - moved
            if dz < 0:                                  # big steps while there is room, 2 mm ones near a peg
                room = self.safe_drop(side, rem)
                if room < 1e-4:
                    break
                # decelerate: at most half the room and 30% of what is left, down to `step` (the arm overshoots big steps)
                s = min(rem, max(step, 0.5 * room, 0.0) if room > step else room, max(step, 0.3 * rem))
                fine = room < 0.01 or rem < 0.02
            else:
                s, fine = min(rem, 0.01), False
            moved += s
            last = moved >= abs(dz) - 1e-4
            g.go(side, np.array([p0[0], p0[1], p0[2] + math.copysign(moved, dz)]), tol=0.0008 if (fine or last) else 0.002,
                 max_steps=30, damping=0.015)
        return moved

    def obj_bottom(self, i):
        """Height (m) of the lowest point of an object (it may be slightly tilted in the gripper)."""
        return min(min(b["a"][2], b["b"][2]) for b in self.bars_world(i)) - OBJ_H / 2

    def clear_of(self, side, i, margin=0.015):
        """True when no part of this hand is near object i: all of it above the object, or well to one side."""
        P = self.hand_points(side)
        c = self.obj_T(i)[:3, 3]
        top = max(max(b["a"][2], b["b"][2]) for b in self.bars_world(i)) + OBJ_H / 2
        return bool(P[:, 2].min() > top + margin or np.linalg.norm(P[:, :2] - c[:2], axis=1).min() > OBJ_RO + margin)

    def orient_for(self, g, side, target, drop):
        """Turn the held object about the vertical so the arm can reach the place pose. Any turn works for placing (the
        peg goes through the hole whichever way the triangle faces), so try turns every 30 degrees, nearest first to
        the one that puts the gripper on the arm's side of the peg, and keep the first one the inverse kinematics can
        reach both high above the peg and down at release height."""
        i = self.holding(side)
        o = (g.tool(side) - self.obj_T(i)[:3, 3])
        u = BASES[side] - np.asarray(target[:2])
        base = math.atan2(o[0] * u[1] - o[1] * u[0], o[:2] @ u) if np.linalg.norm(o[:2]) > 0.02 else 0.0
        a0, c0 = g.current_axes(side)
        q0 = g.env.q()[g.env.idx[side]]
        cands = []
        for t in sorted((base + math.radians(30 * k) for k in range(-6, 6)), key=lambda t: abs(math.remainder(t - base, 2 * math.pi))):
            t = math.atan2(math.sin(t), math.cos(t))
            if all(abs(math.remainder(t - t2, 2 * math.pi)) > math.radians(5) for t2 in cands):
                cands.append(t)
        for t in cands:
            Rz = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
            off = Rz @ o
            hi = np.array([target[0] + off[0], target[1] + off[1], g.tool(side)[2]])
            ok = True
            for pt in (hi, hi - np.array([0, 0, drop])):
                _, pe, ae = g.solver.solve(side, pt, Rz @ a0, Rz @ c0, q0, symmetric=False, tries=8)
                if pe > 0.002 or ae > math.radians(3):
                    ok = False
                    break
            if ok:
                if abs(t) > math.radians(5):
                    g.set_axes(side, Rz @ a0, Rz @ c0, exact=True)
                    g.line_to(side, g.tool(side))           # turn in place, high above the board
                return True
        return False

    def safe_drop(self, side, wanted):
        """How far (m) the gripper can come straight down, up to `wanted`, before any part of it (fingers, gripper, wrist)
        would touch a peg or the board. A gripper holding an object from the side sticks out 10-15 cm, over the
        neighbouring pegs, so lowering all the way to the peg it is placing on can hit the next one."""
        P = self.hand_points(side)
        i = self.holding(side)
        if i is not None:
            P = np.vstack([P, self.underside_points(i)])                       # and the object it carries
        limit = float(P[:, 2].min() - Y.PAPER_TOP - 0.003)                       # the board
        pegs = board_to_world(PEGS_MM)
        near = np.linalg.norm(P[:, None, :2] - pegs[None], axis=2) < PEG_R + 0.004  # points above some peg
        if near.any():
            limit = min(limit, float(P[near.any(1), 2].min() - PEG_TOP - 0.003))
        return max(0.0, min(wanted, limit))

    def underside_points(self, i, n=9):
        """Points along the bottom edges of object i's three walls (world)."""
        T = self.obj_T(i)
        pts = []
        for mid, d, _, _ in BARS:
            nrm = np.cross([0, 0, 1.0], d)
            for s_ in np.linspace(-1, 1, n):
                for w in (-1, 1):
                    pts.append(mid + d * s_ * BAR_L / 2 + nrm * w * OBJ_W / 2 + np.array([0, 0, -OBJ_H / 2]))
        return np.asarray(pts) @ T[:3, :3].T + T[:3, 3]

    def _place(self, g, x_mm, y_mm, height_mm=50.0):
        self.doing = "place"
        holders = [s for s in ("left", "right") if self.holding(s) is not None]
        if len(holders) != 1:
            return False, "place needs exactly one gripper holding an object"
        side = holders[0]
        i = self.holding(side)
        # v2 release interlock: never let go where no peg is under the hole (a triangle dropped onto a peg can end up
        # standing on its edge, where nothing can pick it up again). Refused at once, without moving; within the
        # tolerance the hole is lined up on the peg itself, as a surgeon would seat it.
        px, py = min(PEGS_MM, key=lambda p: math.hypot(x_mm - p[0], y_mm - p[1]))
        if math.hypot(x_mm - px, y_mm - py) > PLACE_TOL_MM:
            self._last = dict(refused=True)
            mx, my = self.sense(px - x_mm, py - y_mm)
            return False, (f"place({x_mm:g}, {y_mm:g}): refused, there is no peg within {PLACE_TOL_MM:g} mm of that point; the "
                           f"nearest peg's centre is {offset_words(mx, my)} of that point. The {COLORS[i][0]} object is "
                           f"still in the {side} gripper")
        x_mm, y_mm = px, py
        target = board_to_world([(x_mm, y_mm)])[0]
        self.lift(g, side)
        self.level(g, side)
        h = float(np.clip(height_mm, 2.0, 60.0)) / 1000
        if not self.orient_for(g, side, target, self.obj_bottom(i) - (PEG_TOP + h)):
            return False, (f"place({x_mm:g}, {y_mm:g}): the {side} arm cannot reach that peg with the {COLORS[i][0]} object "
                           f"held this way; it is still in the {side} gripper")
        T, pad = self.obj_T(i), g.tool(side)
        off = pad - T[:3, 3]                                   # finger pads relative to the object's centre (its hole)
        self.doing = "place: moving over the peg"
        g.line_to(side, [target[0] + off[0], target[1] + off[1], pad[2]])
        self.doing = "place: lining up"
        for _ in range(6):                                     # line the hole up while still high above the pegs
            err = target[:2] - self.obj_T(i)[:2, 3]
            if np.linalg.norm(err) < 0.0015:
                break
            g.go(side, g.tool(side) + np.array([err[0], err[1], 0.0]), tol=0.001, max_steps=120, damping=0.015)
        h_note = ""
        for attempt in range(3):
            drop = self.obj_bottom(i) - (PEG_TOP + h)          # lower until its lowest point is height_mm above the peg top
            self.doing = "place: lowering"
            self.dbg(f"place lower start drop {drop * 1000:.1f}mm", i=i, target=target)
            safe = self.straight(g, side, -drop)               # stops where the gripper would touch another peg
            self.dbg(f"place lower end safe {safe * 1000:.1f}mm", i=i, target=target)
            err = target[:2] - self.obj_T(i)[:2, 3]
            if np.linalg.norm(err) <= 0.004 or attempt == 2:
                break
            # off target down among the pegs: never correct sideways down there; go back up, correct, lower again
            self.doing = "place: re-aiming"
            self.straight(g, side, safe)
            for _ in range(3):
                err = target[:2] - self.obj_T(i)[:2, 3]
                if np.linalg.norm(err) < 0.0015:
                    break
                g.go(side, g.tool(side) + np.array([err[0], err[1], 0.0]), tol=0.001, max_steps=120, damping=0.015)
        if safe < drop - 0.001:
            h_note = f" (as low as the gripper could go without touching another peg; {height_mm:g} mm was asked)"
            h = h + (drop - safe)
        if np.linalg.norm(err) > 0.004:
            # the arm can't hold it precisely over this point: keep it rather than drop it (a robot limit, not a miss)
            self.lift(g, side)
            return False, (f"place({x_mm:g}, {y_mm:g}): the {side} arm could not hold the {COLORS[i][0]} object precisely "
                           f"over that point ({np.linalg.norm(err) * 1000:.0f} mm off); it is still in the {side} gripper")
        if self.tilt_deg(i) > 6:                               # v2: a tilted triangle slides off the peg top: level it first
            self.level(g, side)
            if self.tilt_deg(i) > 6:
                self.lift(g, side)
                return False, (f"place({x_mm:g}, {y_mm:g}): not released, the {COLORS[i][0]} object is held at "
                               f"{self.tilt_deg(i):.0f} degrees and would slide off the peg; hand it over and back to re-grip it. "
                               f"It is still in the {side} gripper")
        self.doing = "place: letting go"
        self._last = dict(release_mm=round(h * 1000, 1))
        self.release(side)
        self.engaged[side] = i                                 # the open fingers are still around it
        g.set_fingers(side, finger_q(OPEN_GAP), steps=12)
        a = g.axes[side][0]
        self.doing = "place: backing away"
        g.line_to(side, g.tool(side) - a * 0.10, spacing=0.006)
        self.engaged[side] = None
        g.keep_orientation(side, exact=False)
        g.hold(45)                                             # let it settle
        w = self.where(i)
        name = COLORS[i][0]
        if w["state"] == "peg":
            moved = self.crossed[i]
            if moved and moved.endswith(w["field"]):
                if i not in self.transfers[moved]:
                    self.transfers[moved].append(i)
                return True, (f"place({x_mm:g}, {y_mm:g}): released {h * 1000:.0f} mm above the peg top{h_note}; the {name} "
                              f"object is on a {w['field']} peg")
            note = " (without a mid-air transfer, so it does not count)"
            return True, f"place({x_mm:g}, {y_mm:g}): the {name} object is on a {w['field']} peg{note}"
        self.drops += 1
        if w["state"] == "off":
            self.lost.add(i)
            return False, f"place({x_mm:g}, {y_mm:g}): missed the peg; the {name} object fell off the board and is lost"
        where = "is resting on top of a peg" if w["state"] == "peg_top" else f"fell onto the board near ({w['xy'][0]:.0f}, {w['xy'][1]:.0f})"
        px, py = min(PEGS_MM, key=lambda p: math.hypot(x_mm - p[0], y_mm - p[1]))
        mx, my = self.sense(px - x_mm, py - y_mm)
        return False, (f"place({x_mm:g}, {y_mm:g}): missed the peg; the {name} object {where}. The nearest peg's centre is "
                       f"{offset_words(mx, my)} of where you aimed")

    # ---------- score
    def score(self, sim_seconds, phases, collisions=0):
        lr, rl = len(self.transfers["left->right"]), len(self.transfers["right->left"])
        on_right = sum(1 for i in range(self.n) if self.where(i).get("state") == "peg" and self.where(i)["field"] == "right")
        # v2: checks are gated on work done (no empty passes): drops and lost mean nothing until something was
        # carried, and the time limit only passes for a finished task (as in FLS)
        carried = any(a_["kind"] == "pick" and a_["ok"] for a_ in self.actions)
        finished = lr + (rl if phases == 2 else 0) >= 6 * phases
        checks = {
            "across": dict(label="Left → right", value=lr, text=f"{lr} / 6", pass_=lr >= 6),
            "drops": dict(label="Drops", value=self.drops, text=str(self.drops) if carried else "n/a (nothing carried)",
                          pass_=carried and self.drops == 0),
            "lost": dict(label="Lost off board", value=len(self.lost), text=str(len(self.lost)) if carried else "n/a (nothing carried)",
                         pass_=carried and not self.lost),
            "time": dict(label="Robot time", value=sim_seconds,
                         text=f"{sim_seconds:.0f} s" + ("" if finished else " (task not finished)"),
                         pass_=finished and sim_seconds <= CUTOFF_S),
            "collisions": dict(label="Collisions", value=collisions, text=str(collisions), pass_=collisions == 0),
        }
        if phases == 2:
            checks["back"] = dict(label="Right → left", value=rl, text=f"{rl} / 6", pass_=rl >= 6)
        done = lr + (rl if phases == 2 else 0)
        need = 6 * phases
        checks["complete"] = dict(label="Transfers", value=done, text=f"{done} / {need}", pass_=done >= need)
        out = {k: dict(label=v["label"], value=v["value"], text=v["text"], **{"pass": bool(v["pass_"])}) for k, v in checks.items()}
        return dict(checks=out, transfers=dict(self.transfers), on_right=on_right, drops=self.drops, lost=sorted(self.lost),
                    sim_seconds=sim_seconds, n_pass=sum(v["pass"] for v in out.values()),
                    **{"pass": all(v["pass"] for v in out.values())})

    # ---------- the model's photo
    def photo(self, width=1350):
        """Orthographic overhead photo cropped to the board: pixel (x, y) maps linearly to board (x_mm, y_mm)."""
        for _ in range(30):
            self.env.world.step(render=True)
            img = Y.rgb(self.cam)
            if img.max() > 0:
                break
        for _ in range(4):
            self.env.world.step(render=True)
        img = Y.rgb(self.cam)
        return np.asarray(Image.fromarray(img).resize((width, int(width * BOARD_MM[1] / BOARD_MM[0])), Image.LANCZOS))


def with_board_grid(img):
    """Grid every 10 mm (faint) with labels every 50 mm, over a photo of exactly the board."""
    im = Image.fromarray(img).convert("RGBA")
    w, h = im.size
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    try:
        font = ImageFont.load_default(size=max(12, w // 80))
    except TypeError:
        font = ImageFont.load_default()
    for x in range(0, int(BOARD_MM[0]) + 1, 10):
        px = x / BOARD_MM[0] * (w - 1)
        dr.line([(px, 0), (px, h)], fill=(30, 50, 140, 110 if x % 50 == 0 else 35), width=1)
        if x % 50 == 0 and 0 < x < BOARD_MM[0]:
            dr.text((px + 2, 2), str(x), fill=(20, 30, 110, 255), font=font)
    for y in range(0, int(BOARD_MM[1]) + 1, 10):
        py = y / BOARD_MM[1] * (h - 1)
        dr.line([(0, py), (w, py)], fill=(30, 50, 140, 110 if y % 50 == 0 else 35), width=1)
        if y % 50 == 0 and 0 < y < BOARD_MM[1]:
            dr.text((2, py + 1), str(y), fill=(20, 30, 110, 255), font=font)
    return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


# ---------------------------------------------------------------- the model's side
TOOLS = [
    dict(type="function", name="pick",
         description="Move one gripper to a point and close its fingers there. Aim at a point on one corner's centre line "
                     "(the line from the tip of the corner to the triangle's centre), about 12 mm in from the tip: the gripper "
                     "turns to pinch that corner and lifts the object.",
         parameters=dict(type="object", additionalProperties=False, required=["arm", "x_mm", "y_mm", "note"], properties=dict(
             arm=dict(type="string", enum=["left", "right"]),
             x_mm=dict(type="number"), y_mm=dict(type="number"),
             note=dict(type="string", description="one short sentence for the operator")))),
    dict(type="function", name="hand_over",
         description="Mid-air transfer: the gripper holding an object brings it to the centre and the other gripper takes it.",
         parameters=dict(type="object", additionalProperties=False, required=["note"], properties=dict(
             note=dict(type="string", description="one short sentence for the operator")))),
    dict(type="function", name="place",
         description="Lower the held object so its hole is centred over the point (x_mm, y_mm), down until its underside is "
                     "height_mm above the top of the peg, then let go: from there it slides down the peg. Give a peg's centre.",
         parameters=dict(type="object", additionalProperties=False, required=["x_mm", "y_mm", "height_mm", "note"], properties=dict(
             x_mm=dict(type="number"), y_mm=dict(type="number"),
             height_mm=dict(type="number", description="how far above the top of the peg to let go (2-60 mm). Lower is gentler: "
                                                       "a few mm means it slides onto the peg instead of being dropped"),
             note=dict(type="string", description="one short sentence for the operator")))),
    dict(type="function", name="done",
         description="Finish the task.",
         parameters=dict(type="object", additionalProperties=False, required=["note"], properties=dict(
             note=dict(type="string", description="one short sentence for the operator")))),
]


def system_prompt(turns, phases, max_calls):
    back = (" When all six are on the right, transfer them back the same way: pick with the right arm, hand over in "
            "mid-air, place on a free left peg with the left arm.") if phases == 2 else ""
    return f"""You are performing the FLS (Fundamentals of Laparoscopic Surgery) peg transfer task with a two-armed robot.
The board has 12 pegs: in the left field six pegs in two straight columns of three, in the right field six pegs on a
ring. There are six coloured solid triangles, each with a small hole through the middle. All six start on the left pegs.

Task: move each object from the left field to a free peg in the right field. For every object: pick it up with the
LEFT arm, transfer it in mid-air to the right arm (hand_over), then place it over a free right peg with the right arm.
Placing without a mid-air transfer does not count.{back} Objects you drop on the board may be picked up again;
objects that fall off the board are lost.

Each arm works only its own half of the board: the left arm the left field (x below {BOARD_MM[0] / 2:.0f} mm), the right
arm the right field. A pick or place outside an arm's half is refused. Objects cross only through the mid-air
hand_over. To recover an object dropped in the right field after its transfer, pick it with the right arm and place
it on a free right peg (it still counts).

Coordinates: millimetres on the board in the overhead photo. x runs from 0 (left edge) to {BOARD_MM[0]:.0f} (right edge),
y from 0 (top edge, far from the robot) to {BOARD_MM[1]:.0f} (bottom edge). The photo has a faint grid line every 10 mm, labelled every 50 mm.
Pegs are the small dark circles (10 mm across); each object is a solid triangle with 70 mm sides and a small hole.
To pick, pinch a corner: aim at a point on the line from the corner's tip to the triangle's centre, about 12 mm in
from the tip. More than about 6 mm to the side of that line, or nearer than 4 mm / farther than 22 mm from the tip,
and the fingers miss. The open fingers need room on both sides of the corner: pick a corner that does not point at a
peg or another triangle. To place, give the exact centre of a free peg: the object's hole is lowered over that point
and released. The hole is only a little wider than the peg: more than about 4 mm off and the object lands on top of
the peg instead of around it.

Handle the objects the way a surgeon would in the exam:
- Place gently. Do not throw or drop objects onto the pegs from a height: lower each one until it is just above the
  top of the peg (height_mm of a few millimetres) and let it slide down the peg.
- Avoid every collision. Grippers must not hit pegs, the board, other objects or each other, and a carried object
  must not hit pegs or other objects. Every collision is reported to you and counts against you.

Each turn you get a fresh overhead photo. After every successful pick and hand_over, the camera on the wrist of the
arm now holding the object takes a photo, and you see those with the next turn's photo (the last three at most): use
them to check how the object sits in the gripper. If you are unsure about a grasp, you may end the turn after pick to
look before handing over; otherwise work efficiently: call pick, hand_over and place for one object together in a
single reply (up to {max_calls} calls per turn; they run in order, and the first one that fails ends the turn).
You have {turns} turns. Call done when the task is complete.

When an action misses, the reply says roughly which way the target was, as a person would judge it by eye:
"a little" (under about 5 mm), "about a centimetre", or "well" (more than about a centimetre) to the left or right,
higher or lower in the photo. It gives no numbers, and the judgement is itself approximate: look at the photo again and
decide how far to move. It never gives the position itself: correct your own reading
of the photo by that offset. If your misses keep pointing the same way, your reading of the grid is biased."""


def turn_prompt(turn, turns, task, results, views=()):
    lines = [f"Turn {turn} of {turns}. The first image is the overhead photo of the board now."]
    if views:
        lines.append("Then, from last turn: " + "; ".join(f"image {k + 2}: {c}" for k, c in enumerate(views)) + ".")
    if results:
        lines.append("Last turn:")
        lines += [f"- {r}" for r in results]
    lines.append(f"Transferred so far: {len(task.transfers['left->right'])} left->right"
                 + (f", {len(task.transfers['right->left'])} right->left" if task.phase == 2 else "") + ".")
    lines.append(f"Turns left after this one: {turns - turn}. A full transfer (pick, hand_over, place) takes 1 turn if you "
                 f"chain the three calls in one reply, 3 if you do not. Putting an object back on the side it started "
                 f"from scores nothing.")
    lines.append("Objects: " + "; ".join(task.summary()) + ".")
    return "\n".join(lines)


def finger_clearance(task, i, cor, point, gap=OPEN_GAP):
    """Predicted gap (m) between the open fingers and the nearest peg, the board or another object, for a vertical
    pinch of corner `cor` of object i at `point`."""
    from ik import GEOM
    a, c = grasp_axes(cor["u"])
    R = np.stack([-c, np.cross(a, -c), a], 1)          # gripper frame: approach = +z, closing = -x (yam_geometry.npz)
    gp = point - R @ GEOM["pad"]
    pts = []
    for side in ("left", "right"):
        T, ax = GEOM[f"tipT_{side}"], GEOM[f"tipaxis_{side}"]
        local = GEOM[f"pts_tip_{side}"] @ T[:3, :3].T + T[:3, 3] + ax * finger_q(gap)
        pts.append(local @ R.T + gp)
    P = np.vstack(pts)
    low = P[P[:, 2] < PEG_TOP]
    pegs = board_to_world(PEGS_MM)
    d_peg = np.linalg.norm(low[:, None, :2] - pegs[None], axis=2).min() - PEG_R if len(low) else 1.0
    others = [task.obj_T(j)[:2, 3] for j in range(task.n) if j != i and j not in task.lost]
    d_obj = min((float(np.linalg.norm(P[:, :2] - o, axis=1).min()) - OBJ_RO for o in others), default=1.0)
    return float(min(d_peg, d_obj, P[:, 2].min() - Y.PAPER_TOP))


def size_word(d):
    """A rough size, as a person judging by eye would say it (no numbers)."""
    return "a little" if d < 5 else "about a centimetre" if d < 12 else "well"


def offset_words(dx, dy):
    """v2 feedback: direction and a rough size only, never millimetres: "a little to the right and about a centimetre
    lower in the photo" (x right, y down the photo)."""
    parts = []
    if abs(dx) >= 1.5:
        parts.append(f"{size_word(abs(dx))} to the {'right' if dx > 0 else 'left'}")
    if abs(dy) >= 1.5:
        parts.append(f"{size_word(abs(dy))} {'lower' if dy > 0 else 'higher'} in the photo")
    return " and ".join(parts) or "very close"


def pinch_point(cor):
    """Where to aim a pick at a corner: on its centre line, PINCH_D in from the tip (xy)."""
    return (cor["tip"] - PINCH_D * cor["u"])[:2]


class ReplayClient:
    """Replays a recorded run's tool calls turn by turn ($0, no API): the same moves on the same board, to reproduce
    and fix what the robot did with them. Its own turn's results are not fed back, so a replay can drift if the robot
    now behaves differently (e.g. a place that failed before succeeds)."""
    model = "replay"

    def __init__(self, turns_json):
        import json
        self.turns, self.turn = json.load(open(turns_json)), 0

    def call(self, instructions, user_text, images, tools, on_event=lambda k, t: None, should_stop=lambda: False):
        import time
        self.turn += 1
        if self.turn > len(self.turns):
            calls = [("done", {})]
            text = "The recording has no more turns."
        else:
            rec = self.turns[self.turn - 1]
            calls = [(n, dict(a or {})) for n, a in (rec.get("calls") or [])]
            text = f"Replaying turn {self.turn} of the recording. " + (rec.get("reasoning") or "")[:400]
        on_event("reasoning", text)
        return dict(tool=calls[0][0] if calls else None, args=calls[0][1] if calls else None, calls=calls, reasoning=text,
                    text="", usage=None, error=None if calls else "no calls recorded", cancelled=False, seconds=0.0)


class ClumsyClient:
    """v2 acceptance test ($0): the stand-in, made to fail the way the real models did, to exercise every error path.
    It aims every pick and place about 8 mm off (a steady bias, like Opus), sometimes reaches into the other field
    with the wrong arm (like both models), and sometimes pinches the corner pointing at the receiver (the hand-over
    clash GPT met). Like a model that reads its feedback, it uses the exact position from a failure message for its
    next action of that kind. A sound v2 lets it finish despite all of this."""
    model = "clumsy"

    def __init__(self, task, seed=0, bias_mm=(8.0, -6.0), p_wrong_arm=0.3, p_bad_corner=0.3):
        import random
        self.inner, self.task = MockClient(task, words_per_second=200), task
        self.rng, self.bias = random.Random(seed), bias_mm
        self.p_wrong_arm, self.p_bad_corner = p_wrong_arm, p_bad_corner
        self.turn = 0

    def call(self, instructions, user_text, images, tools, on_event=lambda k, t: None, should_stop=lambda: False):
        import re
        self.turn += 1
        res = self.inner.call(instructions, user_text, images, tools, on_event, should_stop)
        def corrected(kind):
            """The last aimed point of this kind, moved by the offset the reply reported (relative feedback only)."""
            for line in user_text.splitlines():
                m = re.search(kind + r"\((?:\w+, )?([-\d.]+), ([-\d.]+)\):.*?(?:grip point \(.*?\) is|peg's centre is) (.*?) of (?:where you aimed|that point)", line)
                if m:
                    x, y, words = float(m.group(1)), float(m.group(2)), m.group(3)
                    for w, d in re.findall(r"(a little|about a centimetre|well) (?:to the )?(right|left|lower|higher)", words):
                        n = {"a little": 3.0, "about a centimetre": 10.0, "well": 16.0}[w]
                        x += n if d == "right" else -n if d == "left" else 0
                        y += n if d == "lower" else -n if d == "higher" else 0
                    return [(x, y)]
            return []
        grip = corrected("pick")
        peg = corrected("place")
        if "Repeating hand_over does the same thing" in user_text:
            h = [s_ for s_ in ("left", "right") if self.task.holding(s_) is not None]
            free = self.task.free_pegs(h[0]) if h else []
            if free:                                          # follow the advice: put it back, re-grip next turn
                px, py = PEGS_MM[free[0]]
                calls = [("place", dict(x_mm=px, y_mm=py, height_mm=5, note="Put it back to re-grip by another corner."))]
                res.update(calls=calls, tool="place", args=calls[0][1])
                on_event("reasoning", " [clumsy: following the hand-over advice: put it back to re-grip]")
                return res
        out, notes = [], []
        for name, a in res.get("calls") or []:
            a = dict(a or {})
            if name == "pick":
                if grip:                                       # use the simulator's correction
                    a["x_mm"], a["y_mm"] = float(grip[0][0]), float(grip[0][1]); grip = []; notes.append("using the reported grip point")
                else:
                    if self.rng.random() < self.p_bad_corner:  # pinch the corner pointing at the receiver
                        near = self.task.nearest_corner(board_to_world([(a["x_mm"], a["y_mm"])])[0])
                        if near:
                            to_r = np.array([0, -1.0, 0]) if a["arm"] == "left" else np.array([0, 1.0, 0])
                            cor = max(self.task.corners_world(near[0]), key=lambda c: float(c["u"] @ to_r))
                            a["x_mm"], a["y_mm"] = (round(float(v), 1) for v in world_to_board(pinch_point(cor))[0])
                            notes.append("chose the corner facing the other arm")
                    a["x_mm"] = round(a["x_mm"] + self.bias[0], 1); a["y_mm"] = round(a["y_mm"] + self.bias[1], 1)
                    if self.rng.random() < self.p_wrong_arm:
                        a["arm"] = "right" if a["arm"] == "left" else "left"; notes.append("used the wrong arm")
            elif name == "place":
                if peg:
                    a["x_mm"], a["y_mm"] = float(peg[0][0]), float(peg[0][1]); peg = []; notes.append("using the reported peg centre")
                else:
                    a["x_mm"] = round(a["x_mm"] + self.bias[0], 1); a["y_mm"] = round(a["y_mm"] + self.bias[1], 1)
            out.append((name, a))
        text = (res.get("reasoning") or "") + (" [clumsy: " + "; ".join(notes) + "]" if notes else "")
        on_event("reasoning", text[-200:])
        res.update(calls=out, reasoning=text)
        if out:
            res["tool"], res["args"] = out[0]
        return res


class MockClient:
    """Stand-in for the model ($0): each turn moves the next object (pinch its roomiest corner, hand over, place on a
    free peg), reading positions from the simulator. Token counts are typical of a real call, so the dashboard shows what
    GPT would have cost."""
    model = "mock"

    def __init__(self, task, words_per_second=30):
        self.task, self.wps, self.turn = task, words_per_second, 0

    def call(self, instructions, user_text, images, tools, on_event=lambda k, t: None, should_stop=lambda: False):
        import time
        t0, task = time.time(), self.task
        self.turn += 1
        src, dst = ("left", "right") if task.phase == 1 else ("right", "left")
        if task.holding(src) is not None or task.holding(dst) is not None:
            free = task.free_pegs(dst)
            i = task.holding(src) if task.holding(src) is not None else task.holding(dst)
            px, py = PEGS_MM[free[self.turn % len(free)]] if free else (0, 0)   # vary the peg if one keeps failing
            calls = ([("hand_over", dict(note="Try the transfer again."))] if task.holding(src) is not None else []) + \
                [("place", dict(x_mm=px, y_mm=py, height_mm=5, note=f"Lower it onto the peg at ({px:.0f}, {py:.0f})."))]
            return self._reply(f"The {COLORS[i][0]} object is still in a gripper; I'll finish moving it.", calls, should_stop, on_event, t0)
        todo = [i for i in range(task.n) if i not in task.lost and task.holding("left") != i and task.holding("right") != i
                and task.where(i).get("field") == src and i not in task.transfers[f"{src}->{dst}"]]
        dropped = [i for i in range(task.n) if i not in task.lost and task.where(i)["state"] == "board"
                   and task.tilt_deg(i) <= STANDING_DEG
                   and task.where(i).get("field") == dst and task.crossed[i] == f"{src}->{dst}"]
        free = task.free_pegs(dst)
        if dropped and free:
            i = dropped[0]
            cor = max(task.corners_world(i), key=lambda c: finger_clearance(task, i, c, np.array([*pinch_point(c), c["tip"][2] + OBJ_H / 2 - PAD_BELOW_TOP])))
            p = world_to_board(pinch_point(cor))[0]
            px, py = PEGS_MM[free[0]]
            name = COLORS[i][0]
            text = f"The {name} object was dropped in the {dst} field after its transfer. I'll pick it up with the {dst} arm and put it on the free peg at ({px:.0f}, {py:.0f})."
            calls = [("pick", dict(arm=dst, x_mm=round(float(p[0]), 1), y_mm=round(float(p[1]), 1), note=f"Recover the {name} object.")),
                     ("place", dict(x_mm=px, y_mm=py, height_mm=5, note=f"Put it on the peg at ({px:.0f}, {py:.0f})."))]
        elif not todo or not free:
            text = f"All objects are on the {dst} pegs. The task is complete."
            calls = [("done", dict(note="All transfers complete."))]
        if not dropped and todo and free:
            i = todo[0]
            peg = board_to_world([PEGS_MM[task.where(i)["peg"]]])[0] if task.where(i)["state"] == "peg" else task.obj_T(i)[:2, 3]
            def room(c):                                     # the corner with the most room around the fingers
                return finger_clearance(task, i, c, np.array([*pinch_point(c), c["tip"][2] + OBJ_H / 2 - PAD_BELOW_TOP]))
            cor = max(task.corners_world(i), key=room)
            p = world_to_board(pinch_point(cor))[0]
            px, py = PEGS_MM[free[0]]
            name = COLORS[i][0]
            text = (f"The {name} triangle sits on a {src} peg; I'll pinch the corner with the most room around it, at about ({p[0]:.0f}, {p[1]:.0f}). "
                    f"I'll pick it there with the {src} arm, pass it to the {dst} arm in mid-air, and drop it over the "
                    f"free {dst} peg at ({px:.0f}, {py:.0f}).")
            calls = [("pick", dict(arm=src, x_mm=round(float(p[0]), 1), y_mm=round(float(p[1]), 1), note=f"Pick the {name} object.")),
                     ("hand_over", dict(note=f"Pass it to the {dst} arm.")),
                     ("place", dict(x_mm=px, y_mm=py, height_mm=5, note=f"Lower it onto the peg at ({px:.0f}, {py:.0f})."))]
        return self._reply(text, calls, should_stop, on_event, t0)

    def _reply(self, text, calls, should_stop, on_event, t0):
        import time
        sent = ""
        for w in text.split(" "):
            if should_stop():
                return dict(tool=None, args=None, calls=[], reasoning=sent, text="", usage=None, error=None, cancelled=True, seconds=time.time() - t0)
            on_event("reasoning", w + " ")
            sent += w + " "
            time.sleep(1 / self.wps)
        usage = dict(input=4300 + 150 * self.turn, cached=0 if self.turn == 1 else 1500, output=1100, reasoning=760)
        return dict(tool=calls[0][0], args=calls[0][1], calls=calls, reasoning=sent, text="", usage=usage, error=None,
                    cancelled=False, seconds=time.time() - t0)
