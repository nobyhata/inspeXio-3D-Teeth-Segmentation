"""
geometry.py - apicoectomy planning and post-cut scoring on segmented teeth.

Plain numpy + SimpleITK, no Slicer imports, so it can be tested outside Slicer.

Conventions
  mask arrays are indexed [k, j, i] (the order slicer.util.arrayFromVolume returns)
  M is the 4x4 IJK->RAS matrix of the grid the arrays live on (numpy array)
  every point, distance and angle returned is in RAS millimetres / degrees

Planning (per root)
  apex       most apical point of the root
  axis       unit vector along the apical part of the root, pointing from the apex toward the crown
  plane      resection plane: normal = axis, centre on the root, resect_mm from the apex

Scoring (one trial = one post-cut scan registered to the pre-op grid)
  removed tooth  = pre-op tooth mask where the post-cut scan is now dark (below tau)
  cut face       = remaining tooth voxels that touch the removed part
  level          = distance of the cut-face centroid from the planned apex, along the axis (target resect_mm)
  angle          = angle between the cut-face normal and the axis (0 deg = perpendicular cut)
  wrong tooth    = any non-target tooth that lost more than min_removed_mm3
  bone window    = removed bone on the outer bone surface; its centre is compared with the planned entry point
"""
import numpy as np
import SimpleITK as sitk


# ---------------------------------------------------------------- basics
def kji_to_ras(kji, M):
    kji = np.atleast_2d(np.asarray(kji, float))
    ijk1 = np.c_[kji[:, ::-1], np.ones(len(kji))]
    return (ijk1 @ np.asarray(M, float).T)[:, :3]


def ras_to_kji(ras, M):
    ras = np.atleast_2d(np.asarray(ras, float))
    inv = np.linalg.inv(np.asarray(M, float))
    ijk = (np.c_[ras, np.ones(len(ras))] @ inv.T)[:, :3]
    return ijk[:, ::-1]


def voxel_mm3(M):
    return float(abs(np.linalg.det(np.asarray(M, float)[:3, :3])))


def voxel_size(M):
    return float(voxel_mm3(M) ** (1 / 3))


def mask_points(mask, M):
    return kji_to_ras(np.argwhere(mask), M)


def pca(points):
    c = points.mean(0)
    w, v = np.linalg.eigh(np.cov((points - c).T))
    order = np.argsort(w)[::-1]
    return c, v[:, order].T, w[order]          # centre, axes (rows, major first), variances


def _img(a):
    return sitk.GetImageFromArray(np.ascontiguousarray(a, np.uint8))


def _arr(img):
    return sitk.GetArrayFromImage(img).astype(bool)


def dilate(mask, r=1):
    return _arr(sitk.BinaryDilate(_img(mask), [r] * 3, sitk.sitkBall))


def opening(mask, r=1):
    return _arr(sitk.BinaryMorphologicalOpening(_img(mask), [r] * 3, sitk.sitkBall))


def largest_component(mask):
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(_img(mask)))
    return sitk.GetArrayFromImage(cc) == 1


def remove_small(mask, min_voxels):
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(_img(mask)), int(min_voxels))
    return sitk.GetArrayFromImage(cc) > 0


def erode(mask, r=1):
    return _arr(sitk.BinaryErode(_img(mask), [r] * 3, sitk.sitkBall))


def removed_part(tooth, dark, min_voxels, open_radius=1):
    """Part of a tooth that is now dark, ignoring thin rims along its surface.

    A residual registration error of a voxel makes a thin dark rim along every tooth surface. Only dark
    regions that survive an opening and reach the tooth's core (the tooth eroded by one voxel) count.
    """
    r = np.asarray(tooth, bool) & dark
    if open_radius:
        r = opening(r, open_radius)
    if not r.any():
        return r
    lab = sitk.GetArrayFromImage(sitk.ConnectedComponent(_img(r)))
    core = erode(tooth, 1) & (lab > 0)
    ids, counts = np.unique(lab[core], return_counts=True)
    keep = ids[(ids > 0) & (counts >= min_voxels)]
    return np.isin(lab, keep)


def component_containing(mask, kji):
    lab = sitk.GetArrayFromImage(sitk.ConnectedComponent(_img(mask)))
    v = lab[tuple(np.round(kji).astype(int))]
    return lab == v if v else np.zeros_like(mask, bool)


# ---------------------------------------------------------------- planning
def plan_root(mask, M, apex_hint=None, resect_mm=3.0, axis_len_mm=None, iterations=4):
    """Apex, root axis and resection plane of one root.

    mask        the tooth segment (crown + roots), [k, j, i]
    apex_hint   RAS point near the apex of the root to plan (needed for multi-rooted teeth);
                None: single-rooted tooth, the apical end is found automatically (the narrower end)
    axis_len_mm length of root, measured from the apex, used to fit the axis (default: min(5 mm, 40 % of
                the tooth length))
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        raise ValueError("empty segment")
    s = voxel_size(M)
    idx = np.argwhere(mask)
    P = kji_to_ras(idx, M)
    c, axes, _ = pca(P)
    a = axes[0]
    t = (P - c) @ a
    length = float(t.max() - t.min())
    if apex_hint is None:
        # the apical end of a single root is the narrower end of the tooth
        band = 0.15 * length
        radial = np.linalg.norm((P - c) - np.outer(t, a), axis=1)
        r_lo, r_hi = radial[t < t.min() + band].mean(), radial[t > t.max() - band].mean()
        if r_hi < r_lo:
            a, t = -a, -t
        apex = P[t < t.min() + s].mean(0)
    else:
        apex = P[np.argmin(np.linalg.norm(P - np.asarray(apex_hint, float), axis=1))]
        if (c - apex) @ a < 0:
            a = -a
    L = float(axis_len_mm or min(5.0, 0.4 * length))
    for _ in range(iterations):
        near = np.linalg.norm(P - apex, axis=1) < L
        sub = np.zeros_like(mask)
        sub[tuple(idx[near].T)] = True
        apex_kji = idx[np.argmin(np.linalg.norm(P - apex, axis=1))]
        sub = component_containing(sub, apex_kji)  # keeps one root at a furcation
        Q = kji_to_ras(np.argwhere(sub), M)
        if len(Q) < 10:
            break
        qc, qaxes, _ = pca(Q)
        a = qaxes[0] if (qc - apex) @ qaxes[0] > 0 else -qaxes[0]
        tq = (Q - apex) @ a
        apex = Q[tq < tq.min() + s].mean(0)
    # resection plane: centred on the root cross-section at resect_mm
    tq = (Q - apex) @ a
    slab = Q[np.abs(tq - resect_mm) < s]
    centre = slab.mean(0) if len(slab) > 3 else apex + resect_mm * a
    if len(slab) > 3:
        rad = np.linalg.norm((slab - centre) - np.outer((slab - centre) @ a, a), axis=1)
        diameter = float(2 * rad.max())
    else:
        diameter = float("nan")
    return dict(apex=apex.tolist(), axis=a.tolist(), plane_center=centre.tolist(), plane_normal=a.tolist(),
                resect_mm=float(resect_mm), axis_len_mm=L, section_diameter_mm=diameter,
                tooth_length_mm=length)


def plan_entry(solid_mask, M, target_ras, side_hint_ras=None):
    """Planned bone-entry point: the outer surface point of bone/teeth closest to target_ras.

    solid_mask     bone (and teeth) of the pre-op scan, [k, j, i]
    side_hint_ras  optional point on the side the surgeon approaches from (e.g. buccal): only surface
                   points on that side of the target are considered
    """
    solid = np.asarray(solid_mask, bool)
    surface = dilate(solid, 1) & ~solid            # air voxels touching the solid
    P = mask_points(surface, M)
    tgt = np.asarray(target_ras, float)
    if side_hint_ras is not None:
        d = np.asarray(side_hint_ras, float) - tgt
        keep = (P - tgt) @ d > 0
        if keep.any():
            P = P[keep]
    return P[np.argmin(np.linalg.norm(P - tgt, axis=1))].tolist()


# ---------------------------------------------------------------- scoring
def auto_threshold(post, tooth_mask, air_mask=None):
    """Midway between the tooth and air grey values of the post-cut scan.

    Air is the most common grey value outside the teeth (the scan background inside the field of view);
    exact zeros, the padding of a resampled scan, are ignored.
    """
    post = np.asarray(post)
    tooth = float(np.median(post[tooth_mask]))
    if air_mask is not None and air_mask.any():
        vals = post[air_mask]
    else:
        vals = post[::2, ::2, ::2][~dilate(tooth_mask, 2)[::2, ::2, ::2]]
    vals = vals[vals != 0]
    lo, hi = np.percentile(vals, [1, 99])
    hist, edges = np.histogram(vals[(vals >= lo) & (vals <= hi)], bins=256)
    air = float((edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2)
    return (tooth + air) / 2


def _plane_fit(points, reject_mm=None, rounds=2):
    """Least-squares plane; optionally refit without points farther than reject_mm from it."""
    c, axes, var = pca(points)
    for _ in range(rounds if reject_mm else 0):
        keep = np.abs((points - c) @ axes[2]) < reject_mm
        if keep.sum() < 10 or keep.all():
            break
        points = points[keep]
        c, axes, var = pca(points)
    return c, axes[2], var


def evaluate_cut(teeth, target, plans, post, M, tau, bone=None, entry_plan=None, min_removed_mm3=0.1,
                 open_radius=1, pre=None):
    """Score one trial.

    teeth     {name: pre-op mask}; every tooth segment, so a cut into the wrong tooth is found
    target    name of the tooth that should be resected
    plans     list of plan_root() results for the target's roots
    post      post-cut scan resampled onto the pre-op grid, [k, j, i]
    tau       grey value below which a voxel counts as removed (see auto_threshold)
    bone      optional pre-op bone mask, for the bone-window position
    pre       optional pre-op scan on the same grid: a voxel then only counts as removed if it was clearly
              tooth/bone before (so the pulp canal and partial-volume edges, dark in both scans, never count)
    """
    v = voxel_mm3(M)
    s = voxel_size(M)
    dark = np.asarray(post) < tau
    if pre is not None:
        pre = np.asarray(pre)
        solid_level = float(np.median(pre[np.asarray(teeth[target], bool)]))
        dark = dark & (pre > (tau + max(solid_level, tau)) / 2)
    out = dict(threshold=float(tau), removed_mm3={}, wrong_tooth=[], roots=[])
    removed = {}
    min_vox = max(20, int(0.005 / v))
    for name, m in teeth.items():
        r = removed_part(m, dark, min_vox, open_radius)
        removed[name] = r
        out["removed_mm3"][name] = float(r.sum() * v)
        if name != target and r.sum() * v > min_removed_mm3:
            out["wrong_tooth"].append(name)
    R = removed[target]
    T = np.asarray(teeth[target], bool)
    remaining = T & ~R & ~dark                    # dark rims from residual misregistration are not tooth
    # thin slivers along the removed part (bone seen through a one-voxel misregistration) are not tooth either
    body = opening(remaining, 1) if open_radius else remaining
    body = largest_component(body) if body.any() else body
    face = body & dilate(R, 1)
    F = mask_points(face, M)
    rem_idx = np.argwhere(remaining)
    rem_P = kji_to_ras(rem_idx, M)
    if len(F) == 0:
        out["note"] = "no cut found on the target tooth"
    # assign each cut-face voxel to the root whose planned axis line is closest
    def dist_to_axis(X, p):
        d = X - np.asarray(p["apex"])
        a = np.asarray(p["axis"])
        return np.linalg.norm(d - np.outer(d @ a, a), axis=1)
    owner = np.argmin(np.stack([dist_to_axis(F, p) for p in plans]), 0) if len(F) else np.array([], int)
    for n, p in enumerate(plans):
        apex, a = np.asarray(p["apex"]), np.asarray(p["axis"])
        res = dict(root=n + 1, planned_level_mm=p["resect_mm"])
        Fn = F[owner == n]
        # apical remnant: tooth left on the apical side of the planned plane, near this root
        d = rem_P - apex
        tr = d @ a
        lat = np.linalg.norm(d - np.outer(tr, a), axis=1)
        near_root = lat < max(p.get("section_diameter_mm", 2.0) or 2.0, 4 * s)
        if len(Fn) >= 10:
            q, nrm, var = _plane_fit(Fn, reject_mm=2 * s)
            level = float((q - apex) @ a)
            res.update(cut_center=q.tolist(), cut_normal=nrm.tolist(), level_mm=level,
                       level_error_mm=level - p["resect_mm"],
                       angle_deg=float(np.degrees(np.arccos(min(1.0, abs(nrm @ a))))),
                       lateral_offset_mm=float(np.linalg.norm((q - apex) - level * a)),
                       center_error_mm=float(np.linalg.norm(q - np.asarray(p["plane_center"]))),
                       face_rms_mm=float(np.sqrt(var[2])), face_voxels=int(len(Fn)))
            # tooth left on the apical side of the fitted cut plane
            up = nrm if nrm @ a > 0 else -nrm
            res["apical_remnant_mm3"] = float((((rem_P - q) @ up < -2 * s) & near_root).sum() * v)
        else:
            res.update(note="root not cut", level_mm=None)
            res["apical_remnant_mm3"] = float(((tr < p["resect_mm"]) & near_root).sum() * v)
        # apex removed: most of the tooth within two voxels of the planned apex is gone
        ball = np.linalg.norm(kji_to_ras(np.argwhere(T), M) - apex, axis=1) < 2 * s
        tidx = np.argwhere(T)[ball]
        res["apex_removed"] = bool(len(tidx) and R[tuple(tidx.T)].mean() > 0.5)
        out["roots"].append(res)
    if bone is not None:
        B = np.asarray(bone, bool)
        rb = removed_part(B, dark, min_vox, open_radius)
        solid = B
        for m in teeth.values():
            solid = solid | np.asarray(m, bool)
        air = ~solid
        shell = B & dilate(air, 2)
        window = rb & shell
        out["bone_removed_mm3"] = float(rb.sum() * v)
        if window.any():
            W = mask_points(window, M)
            w = W.mean(0)
            out["window_center"] = w.tolist()
            area = window.sum() * s * s / 2.0      # shell is about two voxels thick
            out["window_diameter_mm"] = float(2 * np.sqrt(area / np.pi))
            if entry_plan is not None:
                out["entry_error_mm"] = float(np.linalg.norm(w - np.asarray(entry_plan, float)))
    return out
