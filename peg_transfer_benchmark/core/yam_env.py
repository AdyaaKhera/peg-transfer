"""YAM bimanual scene for Isaac Sim 6.0. Import after SimulationApp is created.

Adapted from art_benchmarks/core/yam_env.py (via an earlier skin-marking benchmark) and given `tool="none"` (both
grippers free, no pen), views of both grippers and a `build_extra(env)` hook for task objects. Earlier: `sheet=` (a skin pad instead of paper, shown
through the 2D ink texture), a surgical-marker pen and a configurable overhead-camera field of view. The paint
palette/brush code is unused here."""
import math
import os
import sys

import numpy as np

if os.environ.get("SIM_PYLIBS") and os.environ["SIM_PYLIBS"] not in sys.path:
    sys.path.insert(0, os.environ["SIM_PYLIBS"])

import imageio.v2 as imageio
import omni.usd
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.prims import RigidPrim, SingleArticulation, SingleXFormPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera

SIDES = ("left", "right")
ARM_JOINTS = {s: [f"{s}_joint{i}" for i in range(1, 7)] for s in SIDES}
FINGER_JOINTS = {s: [f"{s}_joint7", f"{s}_joint8"] for s in SIDES}
FINGER_MAX = 0.047                        # metres, fully open

PAPER_CENTER = np.array([0.30, -0.305])   # world xy, in front of and between the arm bases
PAPER_SIZE = np.array([0.297, 0.21])      # A4 landscape: (u width, v height)
PAPER_TOP = 0.001
PARK_TIP = (0.25, -0.56, 0.18)            # brush tip parked beside the paper, outside the overhead camera's view
PEN_REACH = 0.10                          # pen tip sits 10 cm beyond the fingertips
PEN_LENGTH = 0.14
INK_TOL = 0.002                           # tip within 2 mm of the paper leaves ink
INK_SPACING = 0.0008
BRUSH_WIDTH = 0.003                       # rendered paint line width, metres
DEFAULT_INK = ("ink", (0.03, 0.06, 0.35))

# Brush + palette mode: paint only comes from dipping the brush tip into a well.
BRUSH_CAPACITY = 0.30                     # metres of stroke one dip can paint before the brush runs dry
PALETTE_Y = -0.50                         # palette row sits just right of the paper (paper u = +0.195)
PALETTE_X = (0.23, 0.265, 0.30, 0.335, 0.37)
WELL_R = 0.013
WELL_TOP = 0.012                          # paint surface height in the wells
DIP_TOL = 0.004                           # tip within 4 mm of the paint surface (or below) loads the brush
DRY_BRISTLE = (0.80, 0.72, 0.55)
FADE_LEVELS = 4                           # load quantisation for rendering fading paint
BRUSH_MAX_WIDTH = 0.010                   # pressing the brush harder splays the bristles: 3 mm light .. 10 mm hard


def configure_sheet(size):
    """Use a sheet of another size (width, height in metres) instead of A4, e.g. a skin pad. Call before creating
    the env: the paper helpers below read PAPER_SIZE at call time."""
    global PAPER_SIZE
    PAPER_SIZE = np.asarray(size, float)


def brush_width(tip_z):
    """Painted width for a brush tip at height tip_z: 3 mm touching lightly, widening ~2.3 mm per mm of press."""
    return float(np.clip(BRUSH_WIDTH + 2.3 * max(0.0, PAPER_TOP - tip_z), BRUSH_WIDTH, BRUSH_MAX_WIDTH))


def paper_to_world(uv):
    """Paper coords (u = right, v = up the page, metres, origin at paper centre) -> world xy."""
    uv = np.atleast_2d(uv)
    return np.stack([PAPER_CENTER[0] + uv[:, 1], PAPER_CENTER[1] - uv[:, 0]], axis=1)


def world_to_paper(xy):
    xy = np.atleast_2d(xy)
    return np.stack([PAPER_CENTER[1] - xy[:, 1], xy[:, 0] - PAPER_CENTER[0]], axis=1)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def look_at(pos, target):
    d = np.asarray(target, float) - np.asarray(pos, float)
    return euler_angles_to_quat(np.array([0, math.atan2(-d[2], math.hypot(d[0], d[1])), math.atan2(d[1], d[0])]))


def rgb(cam):
    a = cam.get_rgba()
    if a is None or a.size == 0:
        w, h = cam.get_resolution()
        return np.zeros((h, w, 3), np.uint8)
    return np.ascontiguousarray(a[:, :, :3]).astype(np.uint8)


def label(img, text, size=None):
    im = Image.fromarray(img)
    if size:
        im = im.resize(size)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 7 * len(text) + 6, 14], fill=(0, 0, 0))
    d.text((3, 1), text, fill=(255, 255, 255))
    return np.asarray(im)


def paint_strength(load):
    """How opaque paint is at a given brush load (a nearly dry brush leaves a faint trace)."""
    return 0.35 + 0.65 * min(1.0, max(0.0, load))


def render_canvas(strokes, colors, guides=(), px_per_m=1600, ss=3, loads=None, widths=None):
    """Top-down render of the paper: strokes painted in their colours over faint tinted guides [(rgb, strokes)].
    `loads` (per-point brush load per stroke) fades paint as the brush dries.
    Supersampled `ss`x then downscaled, so brush strokes are anti-aliased."""
    k = px_per_m * ss
    W, H = (PAPER_SIZE * k).astype(int)
    im = Image.new("RGB", (int(W), int(H)), (247, 245, 238))
    d = ImageDraw.Draw(im)
    bw = max(1, int(round(BRUSH_WIDTH * k)))

    def rgb255(c, fade=0.0):
        return tuple(int(255 * ((1 - fade) * x + fade)) for x in c)

    def paint(s, fill, w):
        p = [((u + PAPER_SIZE[0] / 2) * k, (PAPER_SIZE[1] / 2 - v) * k) for u, v in s]
        if len(p) > 1:
            d.line(p, fill=fill, width=w, joint="curve")
        for x, y in (p[0], p[-1]) if p else ():  # round caps
            d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=fill)
    for c, group in guides:
        for s in group:
            paint(s, rgb255(c, 0.72), max(1, bw // 2))
    for i, (s, c) in enumerate(zip(strokes, colors)):
        if not len(s):
            continue
        if loads is None and widths is None:
            paint(s, rgb255(c), bw)
            continue
        q = np.minimum(FADE_LEVELS - 1, (np.asarray(loads[i]) * FADE_LEVELS).astype(int)) if loads else np.full(len(s), FADE_LEVELS - 1)
        wq = np.round(np.asarray(widths[i]) * 1000).astype(int) if widths else np.full(len(s), round(BRUSH_WIDTH * 1000))
        start = 0
        for j in range(1, len(s) + 1):  # one polyline per run of equal fade level and width
            if j == len(s) or q[j] != q[start] or wq[j] != wq[start]:
                paint(s[max(0, start - 1):j], rgb255(c, 1 - paint_strength((q[start] + 1) / FADE_LEVELS)),
                      max(1, int(round(wq[start] / 1000 * k))))
                start = j
    return np.asarray(im.resize((int(W // ss), int(H // ss)), Image.LANCZOS))


class YamDrawEnv:
    def __init__(self, usd_path, cam_axes=("ros",), hero=True, hero_res=(1280, 720), paper_cam=True, palette=None,
                 paint_texture=False, policy_cams=True, render_every=1, sheet=None, pen_width=BRUSH_WIDTH,
                 paper_cam_fov=42.0, tool="pen", build_extra=None, physics_dt=1 / 60):
        """palette: [(name, rgb)] puts a paint palette on the table and swaps the pen for a brush that only
        paints after being dipped in a well. paint_texture: simulate the paint in 2D (paint_sim.Canvas: brush-shaped
        marks, layered colour) and show it as the paper's texture, instead of the older 3D point marks.
        policy_cams=False skips the robot's top/wrist cameras (only VLA policies need them) and render_every=N
        renders one physics step in N; both make painting runs several times faster.
        sheet: dict(size=(w, h) m, base=HxWx3 float image at px_per_mm, px_per_mm, drape=rgb or None) replaces the A4
        paper with another surface (a skin pad) shown through the paint texture; the pen becomes a surgical marker
        drawing pen_width-wide lines. paper_cam_fov: the overhead camera's horizontal field of view (degrees)."""
        self.sheet, self.tool = sheet, tool
        if sheet is not None:
            configure_sheet(sheet["size"])
        self.pen_width, self.paper_cam_fov = pen_width, paper_cam_fov
        self.ink_spacing = 0.0004 if sheet is not None else INK_SPACING   # a 1.2 mm marker needs dense stamps
        self.tex_gamma = 1.0   # extra gamma on the uploaded texture: the renderer over-exposes it (>1 darkens mid-tones)
        self.render_every = max(1, int(render_every))
        self.palette = list(palette or [])
        self.paint = None
        self._paint_texture = bool(paint_texture and (self.palette or sheet is not None))
        self.world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=physics_dt)
        self.stage = omni.usd.get_context().get_stage()
        self.world.scene.add_default_ground_plane(z_position=-0.75)
        self._add_lights()
        self.mats = {
            "black": self._material("black", (0.015, 0.015, 0.015), 0.45, 0.3),
            "metal": self._material("metal", (0.55, 0.56, 0.58), 0.3, 0.9),
            "pen": self._material("pen", (0.08, 0.12, 0.45), 0.4, 0.0),
            "wood": self._material("wood", (0.55, 0.36, 0.18), 0.6, 0.0),
            "bristle": self._material("bristle", DRY_BRISTLE, 0.8, 0.0, srgb=True),
            "tray": self._material("tray", (0.93, 0.93, 0.95), 0.35, 0.0),
            "marker_body": self._material("marker_body", (0.92, 0.92, 0.94), 0.4, 0.0, srgb=True),
        }
        self.world.scene.add(FixedCuboid("/World/table", name="table", position=np.array([0.2, -0.305, -0.02]),
                                         scale=np.array([1.0, 1.5, 0.04]), color=np.array([0.42, 0.36, 0.30])))
        self.world.scene.add(FixedCuboid("/World/paper", name="paper",
                                         position=np.array([*PAPER_CENTER, PAPER_TOP / 2]),
                                         scale=np.array([PAPER_SIZE[1], PAPER_SIZE[0], PAPER_TOP]),
                                         color=np.array([0.95, 0.95, 0.93])))

        if sheet is not None and sheet.get("drape"):  # surgical drape under the pad (visual only)
            drape = UsdGeom.Cube.Define(self.stage, "/World/drape")
            dxf = UsdGeom.Xformable(drape)
            dxf.AddTranslateOp().Set(Gf.Vec3d(float(PAPER_CENTER[0]), float(PAPER_CENTER[1]), 0.0002))
            dxf.AddScaleOp().Set(Gf.Vec3f(float(PAPER_SIZE[1]) / 2 + 0.06, float(PAPER_SIZE[0]) / 2 + 0.08, 0.0002))
            self._bind(drape.GetPrim(), self._material("drape", sheet["drape"], 0.9, 0.0, srgb=True))
        self.wells = {}
        if self.palette:
            self._setup_palette()
        if self._paint_texture:
            self._setup_paper_texture()
        add_reference_to_stage(usd_path, "/World/yam")
        self._tune_robot()
        if self.tool == "pen":
            self._setup_pen()
        root = next(str(p.GetPath()) for p in self.stage.Traverse()
                    if str(p.GetPath()).startswith("/World/yam") and p.HasAPI(UsdPhysics.ArticulationRootAPI))
        self.robot = self.world.scene.add(SingleArticulation(root, name="yam"))
        self.grips = {s: self.world.scene.add(RigidPrim(f"/World/yam/{s}_gripper", name=f"{s}_grip_view")) for s in SIDES}
        self.grip = self.grips["right"]
        if build_extra is not None:
            build_extra(self)   # task objects: added before world.reset() so their physics views initialise

        self.cams = {}
        for n in (("top", "left", "right") if policy_cams else ()):
            for ax in cam_axes:
                key = n if len(cam_axes) == 1 else f"{n}_{ax}"
                self.cams[key] = (Camera(f"/World/yam/{n}_camera/cam_{ax}", name=f"cam_{n}_{ax}",
                                         resolution=(640, 360), translation=np.zeros(3),
                                         orientation=np.array([1.0, 0, 0, 0])), ax)
        self.hero = None
        if hero:
            pos = np.array([0.95, -0.85, 0.55])
            self.hero = Camera("/World/hero_cam", name="hero", resolution=hero_res, position=pos,
                               orientation=look_at(pos, [0.22, -0.32, 0.05]))
        self.paper_cam = None
        if paper_cam:
            # straight down over the paper: image up = +v, image right = +u
            pos = np.array([*PAPER_CENTER, 0.50])
            self.paper_cam = Camera("/World/paper_cam", name="paper_cam", resolution=(1280, 720), position=pos,
                                    orientation=euler_angles_to_quat(np.array([0, math.pi / 2, 0])))
        self.inks, self.ink, self._prims = {}, None, {}

        self.world.reset()
        for cam, ax in self.cams.values():
            cam.initialize()
            cam.set_local_pose(np.zeros(3), np.array([1.0, 0, 0, 0]), camera_axes=ax)
            cam.set_clipping_range(0.005, 20.0)
            cam.set_focal_length(cam.get_horizontal_aperture() / (2 * math.tan(math.radians(87) / 2)))  # D405 ~87° HFOV
        if self.hero:
            self.hero.initialize()
            self.hero.set_clipping_range(0.01, 20.0)  # USD default near plane is 1 m, which clips the arms
            self.hero.set_focal_length(self.hero.get_horizontal_aperture() / (2 * math.tan(math.radians(60) / 2)))
        if self.paper_cam:
            self.paper_cam.initialize()
            self.paper_cam.set_clipping_range(0.01, 5.0)
            self.paper_cam.set_focal_length(self.paper_cam.get_horizontal_aperture() / (2 * math.tan(math.radians(self.paper_cam_fov) / 2)))

        names = self.robot.dof_names
        self.idx = {s: [names.index(j) for j in ARM_JOINTS[s]] for s in SIDES}
        self.fidx = {s: [names.index(j) for j in FINGER_JOINTS[s]] for s in SIDES}
        props = self.robot.dof_properties
        self.lo, self.hi = np.array(props["lower"]), np.array(props["upper"])
        self.home = self.robot.get_joint_positions().copy()
        self.target = self.home.copy()
        self.t, self.writer, self.video_target, self.ink_enabled = 0, None, None, True
        self.panel = None   # optional () -> (title, text): shown in the video in place of the painting tile
        self.timelapse, self.timelapse_every, self.timelapse_caption = None, 12, ""
        if self.palette:
            for name, color in self.palette:
                self.inks[name] = dict(color=tuple(color))
        else:
            self.set_ink(*DEFAULT_INK)
        self.reset_ink()
        for _ in range(10):
            self.step()

    # ---------- scene setup ----------
    def _add_lights(self):
        UsdLux.DomeLight.Define(self.stage, "/World/Lights/dome").CreateIntensityAttr(500.0)
        sun = UsdLux.DistantLight.Define(self.stage, "/World/Lights/sun")
        sun.CreateIntensityAttr(1500.0)
        sun.CreateAngleAttr(2.0)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 25.0, 0.0))

    def _material(self, name, color, roughness, metallic, emissive=0.0, srgb=False):
        """srgb=True: `color` is a display (sRGB) colour, converted to the linear value the renderer expects,
        so paint looks like its nominal colour on camera instead of washed out."""
        if srgb:
            color = [float(c) ** 2.2 for c in color]
        path = f"/World/Looks/{name}"
        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, f"{path}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        if emissive:
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[emissive * c for c in color]))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def _bind(self, prim, mat):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)

    def _tune_robot(self):
        for p in self.stage.Traverse():
            if not str(p.GetPath()).startswith("/World/yam"):
                continue
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                # position-controlled arms with gravity compensation, like the real YAM controller
                PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr().Set(True)
                self._bind(p, self.mats["metal"] if "tip" in p.GetName() else self.mats["black"])
            if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint):
                angular = p.IsA(UsdPhysics.RevoluteJoint)
                drive = UsdPhysics.DriveAPI.Apply(p, "angular" if angular else "linear")
                drive.CreateStiffnessAttr().Set(100.0 if angular else 5000.0)  # angular gains are per degree
                drive.CreateDampingAttr().Set(2.0 if angular else 100.0)
                drive.CreateMaxForceAttr().Set(1000.0 if angular else 200.0)

    def _setup_paper_texture(self):
        """A quad just above the paper whose texture is the simulated canvas (updated as paint goes down)."""
        from paint_sim import Canvas
        self._canvas_kw = (dict(px_per_mm=self.sheet.get("px_per_mm", 3.0), size=tuple(PAPER_SIZE), base=self.sheet.get("base"))
                           if self.sheet is not None else dict(px_per_mm=3.0))
        self.paint = Canvas(**self._canvas_kw)
        cx, cy, z = PAPER_CENTER[0], PAPER_CENTER[1], PAPER_TOP + 0.0002
        hu, hv = PAPER_SIZE[0] / 2, PAPER_SIZE[1] / 2
        # corners in paper (u, v) -> world (x = cx + v, y = cy - u); st: image top row is v = +hv
        corners = [(-hu, -hv, 0.0, 0.0), (hu, -hv, 1.0, 0.0), (hu, hv, 1.0, 1.0), (-hu, hv, 0.0, 1.0)]
        mesh = UsdGeom.Mesh.Define(self.stage, "/World/paper_paint")
        mesh.CreatePointsAttr([Gf.Vec3f(cx + v, cy - u, z) for u, v, _, _ in corners])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set([Gf.Vec2f(a, b) for _, _, a, b in corners])
        try:
            import omni.ui as ui
            self._tex_provider = ui.DynamicTextureProvider("paper_canvas")
            tex_file, self._tex_dir = "dynamic://paper_canvas", None
        except Exception as e:  # no omni.ui: fall back to rewriting a PNG under a new name each update
            print("dynamic texture unavailable, using files:", e)
            import tempfile
            self._tex_provider, self._tex_dir = None, tempfile.mkdtemp(prefix="paper_canvas_")
            tex_file = os.path.join(self._tex_dir, "canvas_0.png")
            Image.fromarray(self.paint.image()).save(tex_file)
        mat = UsdShade.Material.Define(self.stage, "/World/Looks/paper_paint")
        sh = UsdShade.Shader.Define(self.stage, "/World/Looks/paper_paint/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        reader = UsdShade.Shader.Define(self.stage, "/World/Looks/paper_paint/st")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(self.stage, "/World/Looks/paper_paint/tex")
        tex.CreateIdAttr("UsdUVTexture")
        self._tex_file_input = tex.CreateInput("file", Sdf.ValueTypeNames.Asset)
        self._tex_file_input.Set(tex_file)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        self._bind(mesh.GetPrim(), mat)
        self._tex_version, self._tex_count = -1, 0
        self._upload_paint()

    def _upload_paint(self):
        if self.paint is None or self.paint.version == self._tex_version:
            return
        img = self.paint.image()
        if self._tex_provider is not None:
            # dynamic textures are read as linear (the sourceColorSpace hint is ignored), so linearise the sRGB image
            lin = (np.power(img.astype(np.float32) / 255.0, 2.2 * self.tex_gamma) * 255 + 0.5).astype(np.uint8)
            rgba = np.concatenate([lin, np.full(img.shape[:2] + (1,), 255, np.uint8)], -1)
            self._tex_provider.set_data_array(rgba, [img.shape[1], img.shape[0]])
        else:
            self._tex_count += 1
            path = os.path.join(self._tex_dir, f"canvas_{self._tex_count}.png")
            Image.fromarray(img).save(path)
            self._tex_file_input.Set(path)
            old = os.path.join(self._tex_dir, f"canvas_{self._tex_count - 3}.png")
            if os.path.exists(old):
                os.remove(old)
        self._tex_version = self.paint.version

    def _setup_palette(self):
        """Visual-only palette tray with one paint well per colour (the brush is virtual, so no collisions)."""
        assert len(self.palette) <= len(PALETTE_X), "palette has more colours than wells"
        tray = UsdGeom.Cube.Define(self.stage, "/World/palette/tray")
        xf = UsdGeom.Xformable(tray)
        xf.AddTranslateOp().Set(Gf.Vec3d(float(np.mean(PALETTE_X)), PALETTE_Y, 0.003))
        xf.AddScaleOp().Set(Gf.Vec3f((PALETTE_X[-1] - PALETTE_X[0]) / 2 + 0.022, 0.024, 0.003))  # cube is 2 units
        self._bind(tray.GetPrim(), self.mats["tray"])
        for (name, color), x in zip(self.palette, PALETTE_X):
            safe = "".join(ch if ch.isalnum() else "_" for ch in name)
            # white dish (top 1 mm below the paint) holding a blob of paint whose top is the paint surface
            paint = self._material(f"paint_{safe}", color, 0.25, 0.0, srgb=True)
            for part, r, top, mat in (("dish", WELL_R + 0.004, WELL_TOP - 0.001, self.mats["tray"]),
                                      ("paint", WELL_R, WELL_TOP, paint)):
                c = UsdGeom.Cylinder.Define(self.stage, f"/World/palette/{safe}_{part}")
                c.CreateRadiusAttr(r)
                c.CreateHeightAttr(top - 0.006)
                c.CreateAxisAttr("Z")
                UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(x, PALETTE_Y, (top + 0.006) / 2))
                self._bind(c.GetPrim(), mat)
            self.wells[name] = dict(xy=np.array([x, PALETTE_Y]), color=tuple(color))

    def _setup_pen(self):
        def xf(path):
            return UsdGeom.Xformable(self.stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        g = xf("/World/yam/right_gripper")
        mid = (np.array(xf("/World/yam/right_tip_left").ExtractTranslation()) +
               np.array(xf("/World/yam/right_tip_right").ExtractTranslation())) / 2
        axis = mid - np.array(g.ExtractTranslation())
        axis /= np.linalg.norm(axis)
        inv = g.GetInverse()
        self.pen_tip_local = np.array(inv.Transform(Gf.Vec3d(*(mid + axis * PEN_REACH))))
        a = np.array(inv.TransformDir(Gf.Vec3d(*axis)))
        self.pen_axis_local = a / np.linalg.norm(a)
        UsdGeom.Xform.Define(self.stage, "/World/pen")

        def part(name, geom, radius, length, z0, mat):  # along the tool axis; the tip is the Xform origin, +z points out
            g = geom.Define(self.stage, f"/World/pen/{name}")
            g.CreateRadiusAttr(radius)
            g.CreateHeightAttr(length)
            g.CreateAxisAttr("Z")
            UsdGeom.Xformable(g).AddTranslateOp().Set(Gf.Vec3d(0, 0, z0 - length / 2))
            self._bind(g.GetPrim(), mat)
        if self.palette:
            part("bristles", UsdGeom.Cone, 0.0045, 0.018, 0.0, self.mats["bristle"])      # apex at the tip
            part("ferrule", UsdGeom.Cylinder, 0.0048, 0.012, -0.016, self.mats["metal"])
            part("handle", UsdGeom.Cylinder, 0.004, PEN_LENGTH - 0.02, -0.026, self.mats["wood"])
        elif self.sheet is not None:  # surgical skin marker: white barrel, coloured fine tip (shaded by set_ink)
            part("tip", UsdGeom.Cone, 0.0035, 0.010, 0.0, self.mats["pen"])
            part("body", UsdGeom.Cylinder, 0.0055, PEN_LENGTH - 0.01, -0.01, self.mats["marker_body"])
        else:
            part("body", UsdGeom.Cylinder, 0.005, PEN_LENGTH, 0.0, self.mats["pen"])
        self.pen = SingleXFormPrim("/World/pen", name="pen")

    # ---------- pen + ink ----------
    def grip_pose(self):
        p, q = self.grip.get_world_poses()
        return np.asarray(p[0], float), np.asarray(q[0], float)

    def pen_tip(self):
        p, q = self.grip_pose()
        return p + quat_to_mat(q) @ self.pen_tip_local

    def pen_axis(self):
        _, q = self.grip_pose()
        return quat_to_mat(q) @ self.pen_axis_local

    def _update_pen(self):
        tip, axis = self.pen_tip(), self.pen_axis()
        r = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*axis)).GetQuat()
        self.pen.set_world_pose(tip, np.array([r.GetReal(), *r.GetImaginary()]))
        return tip

    def set_ink(self, name, color):
        """Switch paint colour (a new pen). Strokes are tagged with the ink that drew them.
        Pen mode only: with a palette, colour only changes by dipping the brush."""
        self.inks.setdefault(name, dict(color=tuple(color)))
        self.ink = name
        self._down = False  # a colour change always starts a new stroke
        self._shade("pen", [0.8 * c for c in color])

    def _shade(self, mat, color):
        sh = UsdShade.Shader(self.stage.GetPrimAtPath(f"/World/Looks/{mat}/Shader"))
        sh.GetInput("diffuseColor").Set(Gf.Vec3f(*[float(c) ** 2.2 for c in color]))  # sRGB -> linear

    def _ink_prim(self, name, level):
        """One Points prim per (colour, fade level), so paint fades as the brush dries."""
        key = (name, level)
        if key not in self._prims:
            safe = "".join(ch if ch.isalnum() else "_" for ch in name) + f"_{level}"
            a = paint_strength((level + 1) / FADE_LEVELS)
            color = [a * c + (1 - a) * 0.95 for c in self.inks[name]["color"]]
            prim = UsdGeom.Points.Define(self.stage, f"/World/ink/{safe}")
            self._bind(prim.GetPrim(), self._material(f"ink_{safe}", color, 0.9, 0.0, srgb=True))
            self._prims[key] = prim
        return self._prims[key]

    def _brush_color(self):
        if self.ink is None or self.load <= 0:
            return DRY_BRISTLE
        a = paint_strength(self.load)
        return [a * c + (1 - a) * d for c, d in zip(self.inks[self.ink]["color"], DRY_BRISTLE)]

    def reset_ink(self):
        self.strokes, self.stroke_inks, self.stroke_loads, self.stroke_widths, self.dips = [], [], [], [], []
        self._down, self.min_tip_z, self._ink_dirty = False, np.inf, True
        if self.paint is not None:  # a fresh sheet
            from paint_sim import Canvas
            self.paint = Canvas(**self._canvas_kw)
            self._tex_version = -1
        if self.palette:  # the brush starts clean and dry
            self.ink, self.load = None, 0.0
            self._shade("bristle", DRY_BRISTLE)
        else:
            self.load = 1.0

    def strokes_of(self, ink):
        return [s for s, k in zip(self.strokes, self.stroke_inks) if k == ink]

    def _dip(self, tip):
        """Brush tip in a well: load it with that paint (replacing whatever colour it held)."""
        for name, w in self.wells.items():
            if np.linalg.norm(tip[:2] - w["xy"]) < WELL_R and tip[2] <= WELL_TOP + DIP_TOL:
                if self.ink != name or self.load < 0.999:
                    if not self.dips or self.dips[-1][1] != name or self.t - self.dips[-1][0] > 30:
                        self.dips.append((self.t, name))
                    if self.paint is not None:
                        self.paint.end_stroke()
                    self.ink, self.load, self._down = name, 1.0, False
                    self._shade("bristle", self._brush_color())
                return

    def _update_ink(self, tip):
        self.min_tip_z = min(self.min_tip_z, float(tip[2]))
        if self.wells and self.ink_enabled:
            self._dip(tip)
        u, v = world_to_paper(tip[:2])[0]
        down = (self.ink_enabled and self.ink is not None and self.load > 0 and tip[2] <= PAPER_TOP + INK_TOL
                and abs(u) <= PAPER_SIZE[0] / 2 and abs(v) <= PAPER_SIZE[1] / 2)
        if down:
            if not self._down:
                self.strokes.append([])
                self.stroke_inks.append(self.ink)
                self.stroke_loads.append([])
                self.stroke_widths.append([])
                if self.paint is not None:
                    self.paint.begin_stroke(self.inks[self.ink]["color"])
            s, ld, wd = self.strokes[-1], self.stroke_loads[-1], self.stroke_widths[-1]
            width = brush_width(tip[2]) if self.palette else self.pen_width
            d = math.dist(s[-1], (u, v)) if s else self.ink_spacing
            if d >= self.ink_spacing:
                # fill the gap so a fast-moving pen still leaves a continuous line
                u0, v0 = s[-1] if s else (u, v)
                n = max(1, int(math.ceil(d / self.ink_spacing)))  # never leave a gap wider than the spacing
                for k in range(1, n + 1):
                    if self.palette:
                        self.load = max(0.0, self.load - self.ink_spacing / BRUSH_CAPACITY)
                        if self.load <= 0:
                            break
                    s.append((float(u0 + (u - u0) * k / n), float(v0 + (v - v0) * k / n)))
                    ld.append(self.load)
                    wd.append(width)
                    if self.paint is not None:
                        self.paint.stamp(*s[-1], width, self.load)
                self._ink_dirty = True
                if self.palette and self.t % 10 == 0:
                    self._shade("bristle", self._brush_color())
            if self.load <= 0:
                self._shade("bristle", DRY_BRISTLE)
        if self.paint is not None and self._down and not down:
            self.paint.end_stroke()  # brush lifted (or ran dry / left the paper): the mark is set
        self._down = down
        if self._ink_dirty and self.t % 5 == 0:
            self._flush_ink()

    def _flush_ink(self):
        if self.paint is not None:  # 2D paint: refresh the paper texture instead of point marks
            self._upload_paint()
            self._ink_dirty = False
            return
        groups, gw = {}, {}
        for s, name, ld, wd in zip(self.strokes, self.stroke_inks, self.stroke_loads, self.stroke_widths):
            for p, l, w in zip(s, ld, wd):
                key = (name, min(FADE_LEVELS - 1, int(l * FADE_LEVELS)))
                groups.setdefault(key, []).append(p)
                gw.setdefault(key, []).append(w)
        names = list(self.inks)
        for key in set(groups) | set(self._prims):
            pts = groups.get(key, [])
            xy = paper_to_world(np.array(pts)) if pts else np.zeros((0, 2))
            z = PAPER_TOP + 0.0003 + 0.00005 * names.index(key[0])  # later colours sit on top
            prim = self._ink_prim(*key)
            prim.CreatePointsAttr().Set([Gf.Vec3f(float(x), float(y), z) for x, y in xy])
            prim.CreateWidthsAttr().Set([float(w) for w in gw.get(key, [])])
        self._ink_dirty = False

    # ---------- stepping + policy I/O ----------
    def step(self, q_target=None, render=True):
        if q_target is not None:
            self.target = np.clip(q_target, self.lo, self.hi)
        self.robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=self.target))
        # decide on the counter value this step will have, so the frame checks below (after t += 1) line up with it
        render = render and (self.t + 1) % self.render_every == 0
        self.world.step(render=render)
        self.t += 1
        if self.tool == "pen":
            self._update_ink(self._update_pen())
        if self.writer is not None and render and self.t % max(2, self.render_every) == 0:
            self.writer.append_data(self.video_frame())
        if self.timelapse is not None and render and self.t % self.timelapse_every == 0:
            self.timelapse.append_data(self.timelapse_frame())

    def q(self):
        return self.robot.get_joint_positions()

    def state14(self):
        q, out = self.q(), []
        for s in SIDES:
            out += list(q[self.idx[s]]) + [float(np.mean(q[self.fidx[s]]) / FINGER_MAX)]
        return np.array(out, np.float32)

    def action14_to_target(self, a):
        t = self.target.copy()
        for k, s in enumerate(SIDES):
            t[self.idx[s]] = a[7 * k:7 * k + 6]
            t[self.fidx[s]] = np.clip(a[7 * k + 6], 0, 1) * FINGER_MAX
        return t

    def images(self):
        return {n: rgb(c) for n, (c, _) in self.cams.items()}

    # ---------- outputs ----------
    def canvas(self, target_strokes=None, target_layers=None, px_per_m=1600, ss=3):
        if self.paint is not None:  # 2D paint simulation: the sheet exactly as painted
            return self.paint.image()
        guides = list(target_layers or []) + ([((1.0, 0.45, 0.45), target_strokes)] if target_strokes else [])
        return render_canvas(self.strokes, [self.inks[k]["color"] for k in self.stroke_inks], guides, px_per_m, ss,
                             loads=self.stroke_loads if self.palette else None,
                             widths=self.stroke_widths if self.palette else None)

    def panel_strip(self, W, H=112):
        """A black strip with the panel's title and text, the text word-wrapped; long text scrolls a line every 3 s."""
        import textwrap
        title, text = self.panel()
        im = Image.new("RGB", (W, H), (12, 14, 18))
        d = ImageDraw.Draw(im)
        try:
            fb, ft = ImageFont.load_default(size=17), ImageFont.load_default(size=15)
        except TypeError:  # old Pillow
            fb = ft = ImageFont.load_default()
        d.text((10, 6), title, fill=(120, 200, 255), font=fb)
        lines = [l for para in (text or "").split("\n") for l in (textwrap.wrap(para, 150) or [])]
        shown = 4
        if len(lines) > shown:
            k = int(self.t / 60 / 3) % (len(lines) - shown + 1)
            lines = lines[k:k + shown]
        for j, l in enumerate(lines):
            d.text((10, 30 + 20 * j), l, fill=(225, 228, 232), font=ft)
        return np.asarray(im)

    def video_frame(self):
        if self.panel is not None:   # hero / panel strip / camera row
            hero = rgb(self.hero) if self.hero else np.zeros((720, 1280, 3), np.uint8)
            W = hero.shape[1]
            imgs = self.images()
            cams = [c for c in ("top", "left", "right") if c in imgs]
            tw = W // max(1, len(cams))
            th = (tw * 9 // 16) // 8 * 8
            row = np.hstack([label(imgs[c], f"{c} cam", (tw, th)) for c in cams])
            row = np.pad(row, ((0, 0), (0, W - row.shape[1]), (0, 0)))
            text = f"t={self.t / 60:5.1f}s" + (f"  {self.caption}" if getattr(self, "caption", "") else "")
            return np.vstack([label(hero, text), self.panel_strip(W), row])
        hero = rgb(self.hero) if self.hero else np.zeros((720, 1280, 3), np.uint8)
        W = hero.shape[1]
        imgs = self.images()
        n = len(imgs) + (1 if self.paper_cam else 0) + 1
        tw = W // n
        th = tw * 9 // 16
        tiles = [label(imgs[c], f"{c} cam", (tw, th)) for c in ("top", "left", "right") if c in imgs]
        if self.paper_cam:
            tiles.append(label(rgb(self.paper_cam), "paper cam", (tw, th)))
        tiles.append(label(self.canvas(*(self.video_target or (None, None)), px_per_m=800, ss=1),
                           "painting" if self.paint is not None else "drawing", (tw, th)))
        row = np.hstack(tiles)
        row = np.pad(row, ((0, 0), (0, W - row.shape[1]), (0, 0)))
        text = f"t={self.t / 60:5.1f}s" + (f"  {self.caption}" if getattr(self, "caption", "") else "")
        return np.vstack([label(hero, text), row])

    def start_video(self, path, target_strokes=None, target_layers=None):
        self.video_target = (target_strokes, target_layers)
        fps = 60 / max(2, self.render_every)  # one frame per rendered step (physics runs at 60 Hz): real-time playback
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=9, macro_block_size=8,
                                         ffmpeg_params=["-preset", "slow", "-pix_fmt", "yuv420p"])

    def timelapse_frame(self):
        """1920x540: hero view | overhead canvas camera, with a caption bar."""
        hero = Image.fromarray(rgb(self.hero)).resize((960, 540)) if self.hero else Image.new("RGB", (960, 540))
        paper = Image.fromarray(rgb(self.paper_cam)).resize((960, 540)) if self.paper_cam else Image.new("RGB", (960, 540))
        im = Image.new("RGB", (1920, 540))
        im.paste(hero, (0, 0))
        im.paste(paper, (960, 0))
        d = ImageDraw.Draw(im)
        try:
            font = ImageFont.load_default(size=22)
        except TypeError:  # old Pillow
            font = ImageFont.load_default()
        text = self.timelapse_caption or ""
        d.rectangle([0, 0, 1920, 34], fill=(0, 0, 0))
        d.text((12, 6), text[:150], fill=(255, 255, 255), font=font)
        return np.asarray(im)

    def start_timelapse(self, path, every=12):
        self.timelapse_every = every
        self.timelapse = imageio.get_writer(path, fps=30, codec="libx264", quality=9, macro_block_size=8,
                                            ffmpeg_params=["-preset", "slow", "-pix_fmt", "yuv420p"])

    def stop_timelapse(self):
        if self.timelapse is not None:
            self.timelapse.close()
            self.timelapse = None

    def paper_view(self, width=1000, hide_robot=False):
        """Overhead photo cropped to exactly the paper: pixel (x, y) maps linearly to paper (u, v), with x to the
        right and y down the page. Returns (image, height)."""
        img = self.paper_photo(hide_robot=hide_robot)
        h, w = img.shape[:2]
        f = (w / 2) / math.tan(math.radians(self.paper_cam_fov) / 2)  # paper_cam focal length in pixels
        k = f / (0.50 - PAPER_TOP)                            # pixels per metre on the paper plane
        box = (w / 2 - k * PAPER_SIZE[0] / 2, h / 2 - k * PAPER_SIZE[1] / 2,
               w / 2 + k * PAPER_SIZE[0] / 2, h / 2 + k * PAPER_SIZE[1] / 2)
        height = int(round(width * PAPER_SIZE[1] / PAPER_SIZE[0]))
        return np.asarray(Image.fromarray(img).crop(box).resize((width, height), Image.LANCZOS)), height

    def paper_photo(self, hide_robot=False):
        """High-res in-sim photo of the paper from the overhead camera. During a run, park the arm out of view first
        (Oracle.park) instead of hiding it: once the robot has been hidden and shown again, Isaac renders it in the
        URDF's own colours for the rest of the run. hide_robot=True is for the final photo only, after the video."""
        if not self.paper_cam:
            return None
        self._flush_ink()
        hide = [UsdGeom.Imageable(self.stage.GetPrimAtPath(p)) for p in ("/World/yam", "/World/pen")] if hide_robot else []
        for h in hide:
            h.MakeInvisible()
        img = None
        for _ in range(30):  # annotators lag a few frames behind the render
            self.world.step(render=True)
            img = rgb(self.paper_cam)
            if img.max() > 0:
                break
        for _ in range(5):
            self.world.step(render=True)
        img = rgb(self.paper_cam)
        for h in hide:
            h.MakeVisible()
        return img

    def stop_video(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None