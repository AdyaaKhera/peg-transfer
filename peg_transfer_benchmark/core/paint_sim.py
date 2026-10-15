"""Copied from art_benchmarks/core/paint_sim.py (fd95744), plus a sheet size and base image (the skin pad).

2D paint simulation of the sheet: soft round brush stamps, width from pressure, opacity from how loaded the brush
is, and layered colour (later paint covers earlier paint). Plain numpy, used both inside Isaac Sim (the paper's
texture) and offline by the stroke planner.

Paper coordinates (u right, v up, metres, origin at the paper centre) map to canvas pixels (x right, y down)."""
import numpy as np
from scipy.ndimage import gaussian_filter

PAPER_SIZE = (0.297, 0.21)
PAPER_RGB = np.array([0.97, 0.965, 0.94], np.float32)
BASE_OPACITY = 0.92            # a fully loaded brush nearly, but not completely, covers what is underneath


def paint_strength(load):
    """Opacity multiplier for a brush holding `load` (0..1) of a full dip: a nearly dry brush leaves a faint trace."""
    return 0.35 + 0.65 * min(1.0, max(0.0, load))


class Canvas:
    def __init__(self, px_per_mm=3.0, seed=0, size=None, base=None):
        """size: sheet (width, height) in metres (default A4 landscape). base: optional HxWx3 float image of the bare
        sheet (e.g. a skin pad with its lesion), at px_per_mm, instead of plain paper."""
        self.k = px_per_mm * 1000.0                                   # pixels per metre
        self.size = tuple(size) if size is not None else PAPER_SIZE
        self.W, self.H = int(round(self.size[0] * self.k)), int(round(self.size[1] * self.k))
        self.base = np.empty((self.H, self.W, 3), np.float32)
        self.base[:] = PAPER_RGB if base is None else np.asarray(base, np.float32)[:self.H, :self.W]
        rng = np.random.default_rng(seed)
        # paper/bristle grain: paint coverage varies a little from pixel to pixel
        self.grain = (0.88 + 0.12 * np.clip(gaussian_filter(rng.random((self.H, self.W)), 1.2) * 4 - 1.5, 0, 1)).astype(np.float32)
        self.alpha, self.color, self.version = None, None, 0

    def copy(self):
        """A planning copy: same paint on it so far (any stroke in progress is left out)."""
        c = Canvas.__new__(Canvas)
        c.k, c.W, c.H, c.size = self.k, self.W, self.H, self.size
        c.base, c.grain = self.base.copy(), self.grain
        c.alpha, c.color, c.version = None, None, 0
        return c

    # ---------- coordinates ----------
    def uv_to_px(self, u, v):
        return (u + self.size[0] / 2) * self.k, (self.size[1] / 2 - v) * self.k

    def px_to_uv(self, x, y):
        return x / self.k - self.size[0] / 2, self.size[1] / 2 - y / self.k

    # ---------- painting ----------
    def begin_stroke(self, rgb):
        self.alpha = np.zeros((self.H, self.W), np.float32)
        self.color = np.asarray(rgb, np.float32)

    def stamp(self, u, v, width, load=1.0):
        """One brush footprint at (u, v): a soft disc `width` metres across. Within a stroke, overlapping stamps don't
        stack up (the stroke's coverage is the max), like one continuous brush mark."""
        if self.alpha is None:
            return
        x, y = self.uv_to_px(u, v)
        r = max(1.0, width / 2 * self.k)
        x0, x1 = int(max(0, np.floor(x - r - 1))), int(min(self.W, np.ceil(x + r + 2)))
        y0, y1 = int(max(0, np.floor(y - r - 1))), int(min(self.H, np.ceil(y + r + 2)))
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        soft = np.clip((r - d) / max(1.0, 0.25 * r) + 0.5, 0, 1)        # ~1 px soft edge on small brushes, wider on big
        a = BASE_OPACITY * paint_strength(load) * soft * self.grain[y0:y1, x0:x1]
        np.maximum(self.alpha[y0:y1, x0:x1], a, out=self.alpha[y0:y1, x0:x1])
        self.version += 1

    def end_stroke(self):
        if self.alpha is not None:
            a = self.alpha[..., None]
            self.base = self.base * (1 - a) + self.color * a
            self.alpha = None
            self.version += 1

    def paint_polyline(self, pts_uv, rgb, width, load=1.0, spacing=0.0008):
        """Whole stroke in one call (used by the planner's preview)."""
        pts = np.asarray(pts_uv, float).reshape(-1, 2)
        self.begin_stroke(rgb)
        if len(pts) == 1:
            self.stamp(*pts[0], width, load)
        for p, q in zip(pts[:-1], pts[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(q - p) / spacing)))
            for t in np.linspace(0, 1, n + 1):
                self.stamp(*(p + (q - p) * t), width, load)
        self.end_stroke()

    # ---------- output ----------
    def rgb_float(self):
        if self.alpha is None:
            return self.base
        a = self.alpha[..., None]
        return self.base * (1 - a) + self.color * a

    def image(self):
        """uint8 HxWx3 of the paper as it looks now (including the stroke being painted)."""
        return (np.clip(self.rgb_float(), 0, 1) * 255 + 0.5).astype(np.uint8)
