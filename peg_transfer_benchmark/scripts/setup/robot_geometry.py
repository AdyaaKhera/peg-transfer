"""The YAM URDF has visual meshes but no collision shapes, so in Isaac the arms pass through everything. This script
turns each link's mesh into surface sample points (in that link's frame) for the collision monitor, and measures the
fingers: where the finger pads are, and how wide the gap between them is for a given finger joint value.

Writes core/yam_geometry.npz. Plain numpy; run once on the login node:
  python3 peg_transfer_benchmark/scripts/setup/robot_geometry.py"""
import math
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
URDF = os.path.join(P, "third_party/i2rt/i2rt/robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.urdf")
OUT = os.path.join(HERE, "..", "..", "core", "yam_geometry.npz")
LINKS = ["base", "link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right"]
SPACING = 0.004          # about one sample point per 4 mm of surface


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def origin(el):
    o = el.find("origin")
    xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rpy_mat(*rpy), xyz
    return T


def read_stl(path):
    data = open(path, "rb").read()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        v = [list(map(float, l.split()[1:4])) for l in data.decode(errors="ignore").splitlines() if l.strip().startswith("vertex")]
        return np.array(v).reshape(-1, 3, 3)
    n = struct.unpack("<I", data[80:84])[0]
    arr = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return arr["v"].reshape(-1, 3, 3).astype(float)


def sample_surface(tris, spacing, rng):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = max(50, int(area.sum() / spacing ** 2))
    idx = rng.choice(len(tris), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])
    return np.vstack([pts, tris.reshape(-1, 3)[::max(1, len(tris) // 200)]])   # plus some vertices (sharp corners)


def main():
    rng = np.random.default_rng(0)
    root = ET.parse(URDF).getroot()
    base_dir = os.path.dirname(URDF)
    out = {}
    for name in LINKS:
        link = root.find(f"link[@name='right_{name}']")
        pts = []
        for vis in link.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            tris = read_stl(os.path.normpath(os.path.join(base_dir, mesh.get("filename"))))
            scale = np.array([float(s) for s in mesh.get("scale", "1 1 1").split()])
            T = origin(vis)
            tris = (tris * scale) @ T[:3, :3].T + T[:3, 3]
            pts.append(sample_surface(tris, SPACING, rng))
        out[f"pts_{name}"] = np.vstack(pts) if pts else np.zeros((0, 3))
        print(f"{name}: {len(out[f'pts_{name}'])} points, extent {np.ptp(out[f'pts_{name}'], 0).round(3) if len(out[f'pts_{name}']) else '-'} m")
    # fingers: tip frames in the gripper frame at joint value q (prismatic along each joint's axis)
    for side, j in (("left", "right_joint7"), ("right", "right_joint8")):
        jt = root.find(f"joint[@name='{j}']")
        T = origin(jt)
        ax = np.array([float(v) for v in jt.find("axis").get("xyz").split()])
        out[f"tipT_{side}"], out[f"tipaxis_{side}"] = T, T[:3, :3] @ ax       # axis in the gripper frame
    tl = out["pts_tip_left"] @ out["tipT_left"][:3, :3].T + out["tipT_left"][:3, 3]
    tr = out["pts_tip_right"] @ out["tipT_right"][:3, :3].T + out["tipT_right"][:3, 3]
    closing = out["tipaxis_left"] / np.linalg.norm(out["tipaxis_left"])       # left tip moves along this as q grows
    mid0 = (tl.mean(0) + tr.mean(0)) / 2
    gp = np.zeros(3)                                                           # gripper origin
    approach = mid0 - gp - closing * ((mid0 - gp) @ closing)
    approach /= np.linalg.norm(approach)
    # fingertip ends: the furthest points along the approach direction; pads: 7 mm back from the ends
    end = max((tl @ approach).max(), (tr @ approach).max())
    pad_depth = 0.007
    near_l = tl[(tl @ approach) > end - 0.015]
    near_r = tr[(tr @ approach) > end - 0.015]
    gap0 = float((near_l @ closing).min() - (near_r @ closing).max())   # inner faces at q = 0 (sign: + is open)
    pad = (near_l.mean(0) + near_r.mean(0)) / 2
    pad = pad - approach * ((pad @ approach) - (end - pad_depth))
    out.update(approach=approach, closing=closing, pad=pad, gap0=np.array(gap0), end=np.array(end))
    print(f"approach (gripper frame) {approach.round(3)}, closing {closing.round(3)}")
    print(f"finger pad centre {pad.round(4)} m, fingertip end {end:.4f} m along approach")
    print(f"gap between fingertips at q=0: {gap0 * 1000:.1f} mm; each finger moves along its axis, so gap(q) = gap0 + 2q")
    np.savez_compressed(OUT, **out)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
