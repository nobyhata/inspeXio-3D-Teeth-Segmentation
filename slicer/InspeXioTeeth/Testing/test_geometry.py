"""Synthetic tests for InspeXioTeethLib.geometry (run with: python -m pytest Testing/test_geometry.py)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from InspeXioTeethLib import geometry as g

S = 0.1                                            # mm per voxel
SHAPE = (160, 120, 140)                            # k, j, i
M = np.array([[-S, 0, 0, 5.0], [0, -S, 0, 3.0], [0, 0, S, -2.0], [0, 0, 0, 1]])  # LPS-like flips + offset


def grid():
    k, j, i = np.indices(SHAPE)
    ijk1 = np.stack([i, j, k, np.ones_like(i)], -1).astype(float)
    return (ijk1 @ M.T)[..., :3]


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


def root(X, apex, axis, length, r_apex, r_top):
    d = X - apex
    t = d @ axis
    lat = np.linalg.norm(d - t[..., None] * axis, axis=-1)
    r = r_apex + (r_top - r_apex) * np.clip(t / length, 0, 1)
    return (t >= 0) & (t <= length) & (lat <= r)


def molar(X, shift=0.0):
    """Two-rooted tooth: crown block on top (+S in RAS), two tilted roots below."""
    c = np.array([5.0 - 7.0 + shift, 3.0 - 6.0, -2.0 + 11.0])     # crown centre
    crown = np.all(np.abs(X - c) <= [1.6, 1.4, 1.0], -1)
    a1, a2 = unit([0.25, 0.0, 1.0]), unit([-0.2, 0.1, 1.0])          # apex -> crown
    apex1 = c + [-0.9, 0, -1.0] - 6.0 * a1
    apex2 = c + [0.9, 0, -1.0] - 6.0 * a2
    m = crown | root(X, apex1, a1, 6.3, 0.25, 0.7) | root(X, apex2, a2, 6.3, 0.25, 0.7)
    return m, (apex1, a1), (apex2, a2)


def test_plan_two_roots():
    X = grid()
    m, (ap1, a1), (ap2, a2) = molar(X)
    for ap, a in ((ap1, a1), (ap2, a2)):
        p = g.plan_root(m, M, apex_hint=ap + [0, 0, 0.3], resect_mm=3.0, axis_len_mm=4.0)
        assert np.linalg.norm(np.array(p["apex"]) - ap) < 0.25, p["apex"]
        ang = np.degrees(np.arccos(abs(np.dot(p["axis"], a))))
        assert ang < 3.0, ang
        assert np.dot(p["axis"], a) > 0
        exp_c = ap + 3.0 * a
        assert np.linalg.norm(np.array(p["plane_center"]) - exp_c) < 0.3


def test_plan_single_root_auto():
    X = grid()
    a = unit([0.1, -0.15, 1.0]); ap = np.array([-2.0, -3.0, 4.0])
    m = root(X, ap, a, 9.0, 0.3, 1.5)
    p = g.plan_root(m, M, resect_mm=3.0)
    assert np.linalg.norm(np.array(p["apex"]) - ap) < 0.25
    assert np.degrees(np.arccos(abs(np.dot(p["axis"], a)))) < 3.0


def make_trial(cut_level, cut_tilt_deg, cut_neighbour=False):
    X = grid()
    m, (ap1, a1), _ = molar(X)
    nb, _, _ = molar(X, shift=4.5)                  # neighbouring tooth
    bone = np.all(np.abs(X - [-2.0 + 2.25, -3.0, 4.5]) <= [6.0, 2.5, 3.5], -1) & ~m & ~nb
    post = np.full(SHAPE, 34700.0)
    post[bone] = 37500; post[m | nb] = 38800
    # tilted cut plane through the root
    tilt = np.radians(cut_tilt_deg)
    perp = unit(np.cross(a1, [0, 1, 0]))
    n = np.cos(tilt) * a1 + np.sin(tilt) * perp
    centre = ap1 + cut_level * a1
    apical = ((X - centre) @ n) < 0
    near = np.linalg.norm((X - ap1) - ((X - ap1) @ a1)[..., None] * a1, axis=-1) < 1.2
    post[m & apical & near] = 34700
    # bone window: a channel from the -A (posterior? here -y) surface to the root
    entry = np.array([centre[0], -3.0 - 2.5, centre[2]])
    chan = (np.linalg.norm((X - entry)[..., [0, 2]], axis=-1) < 1.0) & (X[..., 1] <= centre[1]) & bone
    post[chan] = 34700
    if cut_neighbour:
        post[nb & (np.linalg.norm(X - (ap1 + [4.5, 0, 2.0]), axis=-1) < 0.8)] = 34700
    return m, nb, bone, post, ap1, a1, entry


def test_evaluate_cut():
    m, nb, bone, post, ap1, a1, entry = make_trial(3.4, 12.0)
    plan = g.plan_root(m, M, apex_hint=ap1, resect_mm=3.0, axis_len_mm=4.0)
    tau = g.auto_threshold(post, m)
    r = g.evaluate_cut({"t46": m, "t47": nb}, "t46", [plan], post, M, tau, bone=bone, entry_plan=entry)
    root = r["roots"][0]
    assert r["wrong_tooth"] == []
    assert abs(root["level_mm"] - 3.4) < 0.25, root["level_mm"]
    assert abs(root["angle_deg"] - 12.0) < 3.0, root["angle_deg"]
    assert root["apex_removed"]
    assert root["apical_remnant_mm3"] < 0.02, root["apical_remnant_mm3"]
    assert r["entry_error_mm"] < 0.6, r["entry_error_mm"]


def test_wrong_tooth_flagged():
    m, nb, bone, post, ap1, a1, entry = make_trial(3.0, 0.0, cut_neighbour=True)
    plan = g.plan_root(m, M, apex_hint=ap1, resect_mm=3.0, axis_len_mm=4.0)
    r = g.evaluate_cut({"t46": m, "t47": nb}, "t46", [plan], post, M, g.auto_threshold(post, m))
    assert r["wrong_tooth"] == ["t47"]
    assert abs(r["roots"][0]["angle_deg"]) < 4.0     # ~1 mm root at 0.1 mm voxels: ~5 deg resolution


def test_two_root_plan_one_root_cut():
    X = grid()
    m, (ap1, a1), (ap2, a2) = molar(X)
    post = np.where(m, 38800.0, 34700.0)
    rng = np.random.default_rng(0)
    post += rng.normal(0, 300, SHAPE)              # scanner noise
    post[:12] = 0; post[:, :12] = 0                # zero padding from resampling
    post[m & (((X - ap1) @ a1) < 2.5) & (np.linalg.norm(X - ap1, axis=-1) < 3.5)] = 34700
    plans = [g.plan_root(m, M, apex_hint=ap, resect_mm=3.0, axis_len_mm=4.0) for ap in (ap1, ap2)]
    r = g.evaluate_cut({"t46": m}, "t46", plans, post, M, g.auto_threshold(post, m))
    r1, r2 = r["roots"]
    assert abs(r1["level_mm"] - 2.5) < 0.25 and r1["angle_deg"] < 4, r1
    assert r2["level_mm"] is None and not r2["apex_removed"], r2


def test_one_voxel_misregistration():
    m, nb, bone, post, ap1, a1, entry = make_trial(3.4, 12.0)
    post = np.roll(post, 1, axis=2)                 # residual registration error of one voxel
    plan = g.plan_root(m, M, apex_hint=ap1, resect_mm=3.0, axis_len_mm=4.0)
    r = g.evaluate_cut({"t46": m, "t47": nb}, "t46", [plan], post, M, g.auto_threshold(post, m), bone=bone,
                       entry_plan=entry)
    root = r["roots"][0]
    assert r["wrong_tooth"] == [], r["removed_mm3"]
    assert abs(root["level_mm"] - 3.4) < 0.3, root["level_mm"]
    assert abs(root["angle_deg"] - 12.0) < 4.0, root["angle_deg"]


def test_pulp_canal_not_counted():
    """A dark canal inside every tooth (dark before and after) must not count as removed."""
    X = grid()
    m, (ap1, a1), _ = molar(X)
    nb, (bp1, b1), _ = molar(X, shift=4.5)
    pre = np.full(SHAPE, 34700.0); pre[m | nb] = 38800
    for ap, a in ((ap1, a1), (bp1, b1)):
        pre[root(X, ap, a, 6.3, 0.08, 0.12)] = 35500    # canal, ~2 voxels wide
    post = pre.copy()
    post[m & (((X - ap1) @ a1) < 3.0) & (np.linalg.norm(X - ap1, axis=-1) < 3.5)] = 34700
    plan = g.plan_root(m, M, apex_hint=ap1, resect_mm=3.0, axis_len_mm=4.0)
    tau = g.auto_threshold(post, m)
    r = g.evaluate_cut({"t46": m, "t47": nb}, "t46", [plan], post, M, tau, pre=pre)
    assert r["removed_mm3"]["t47"] == 0.0 and r["wrong_tooth"] == [], r["removed_mm3"]
    assert abs(r["roots"][0]["level_mm"] - 3.0) < 0.25, r["roots"][0]


if __name__ == "__main__":
    for f in [test_pulp_canal_not_counted, test_one_voxel_misregistration, test_two_root_plan_one_root_cut, test_plan_two_roots, test_plan_single_root_auto, test_evaluate_cut, test_wrong_tooth_flagged]:
        f(); print("ok", f.__name__)
