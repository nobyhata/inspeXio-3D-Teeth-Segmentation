"""
nninteractive_teeth.py - segment individual teeth (crown + root) in an inspeXio micro-CT volume with
nnInteractive (MIC-DKFZ), driven from 3D Slicer's Python console.

nnInteractive is a promptable 3D segmentation model: given a box or a few points on an object it
segments that object in 3D. It was trained on CT, MRI, PET and 3D microscopy, so unlike CBCT-only dental
models it does not depend on clinical grey values. One 2D box per tooth (drawn on a vertical plane
through the tooth, from crown tip to root apex) is enough to separate the tooth from the embedding base.

Setup once (Slicer Python console):
    slicer.util.pip_install("nninteractive")     # also needs the PyTorch extension + torch with CUDA
The model weights (CC BY-NC-SA 4.0, non-commercial) download automatically on first use.

Edit the settings, then:
    exec(open(r"C:\\path\\to\\nninteractive_teeth.py").read())
"""
import os, time, traceback
import numpy as np
import slicer, vtk, ctk

# ---- settings -------------------------------------------------------------------------------
VOLUME_PATH = r"C:\path\to\scan\noring_0000\scan_noring.nhdr"
SLICES      = (0, 600)      # slice range (full-resolution k) holding the teeth, crowns to root apices
DOWNSAMPLE  = 2             # 2 -> 0.089 mm voxels: fits a 12 GB GPU and keeps the root outlines
# One prompt per tooth, in voxel coordinates of the downsampled volume, [[k0,k1],[j0,j1],[i0,i1]].
# A 2D box has size 1 along one axis: a vertical plane through the tooth centre, from the crown tip
# to just past the root apex (the end of the dark outline around the root). The box length matters:
# nnInteractive takes the box as the extent of the object, so a box that runs past the apex pulls the
# segmentation into the base.
TEETH = {                   # example (molar, box only) and a tooth that needed negative points
    "tooth_01": [[36, 246], [71, 72], [103, 233]],
    "tooth_04": {"box": [[25, 236], [318, 402], [176, 177]],
                 "exclude": [[180, 316, 176], [200, 400, 176], [242, 360, 176], [150, 318, 165],
                             [205, 402, 190], [65, 406, 176], [215, 330, 165]]},
}
# A tooth can also be {"box": [...], "exclude": [[k,j,i], ...], "include": [[k,j,i], ...]}: extra negative
# points remove a spill into the base, positive points add a missed part. Points are applied after the box.
TEETH_JSON  = None          # optional .json file with the same {name: box or dict} mapping (overrides TEETH)
GROW_STEP   = None          # nnInteractive fills the box it is given, so set each box to end just past the root apex
EXPORT_DIR  = None          # folder for one STL per tooth (None: no export)
LOG = None                  # optional path of a text log
# -----------------------------------------------------------------------------------------------

def log(*a):
    msg = " ".join(str(x) for x in a); print(msg)
    if LOG:
        with open(LOG, "a") as f: f.write(msg + "\n")

try:
    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    from nnInteractive import model_management as mm

    name = os.path.splitext(os.path.basename(VOLUME_PATH))[0]
    vol = slicer.util.getFirstNodeByClassByName("vtkMRMLScalarVolumeNode", name) or slicer.util.loadVolume(VOLUME_PATH)

    # 1. downsample the teeth slab by block averaging (also reduces noise)
    t0 = time.time(); d = DOWNSAMPLE
    full = slicer.util.arrayFromVolume(vol)                     # [k, j, i]
    k0, k1 = SLICES; nk = (k1 - k0) // d
    nj, ni = full.shape[1] // d, full.shape[2] // d
    low_arr = np.empty((nk, nj, ni), np.float32)
    for k in range(nk):
        blk = full[k0 + k * d:k0 + (k + 1) * d, :nj * d, :ni * d].astype(np.float32)
        low_arr[k] = blk.reshape(d, nj, d, ni, d).mean((0, 2, 4))
    m = vtk.vtkMatrix4x4(); vol.GetIJKToRASMatrix(m)
    origin = m.MultiplyPoint(((d - 1) / 2, (d - 1) / 2, k0 + (d - 1) / 2, 1))[:3]
    for r in range(3):
        for c in range(3): m.SetElement(r, c, m.GetElement(r, c) * d)
    low = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{name}_x{d}")
    low.SetIJKToRASMatrix(m); low.SetOrigin(origin)
    slicer.util.updateVolumeFromArray(low, low_arr)
    low.CreateDefaultDisplayNodes(); low.GetDisplayNode().AutoWindowLevelOff(); low.GetDisplayNode().SetWindowLevel(9900, 38100)
    log("downsampled", low_arr.shape, low.GetSpacing(), f"{time.time() - t0:.0f}s")

    # 2. nnInteractive session
    t0 = time.time()
    model_dir = mm.ensure_model_available(mm.get_default_model_id())
    session = nnInteractiveInferenceSession(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                            use_torch_compile=False, verbose=False, do_autozoom=True)
    session.initialize_from_trained_model_folder(str(model_dir))
    session.set_image(low_arr[None])
    target = np.zeros(low_arr.shape, np.uint8)
    session.set_target_buffer(target)
    log("model ready", model_dir, f"{time.time() - t0:.0f}s")

    # 3. one box per tooth
    if TEETH_JSON:
        import json; TEETH = json.load(open(TEETH_JSON))
    seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "teeth_nnInteractive")
    seg.CreateDefaultDisplayNodes(); seg.SetReferenceImageGeometryParameterFromVolumeNode(low)
    for tooth, spec in TEETH.items():
        if not isinstance(spec, dict): spec = {"box": spec}
        t0 = time.time(); box = [list(b) for b in spec["box"]]
        while True:
            session.reset_interactions(); target[:] = 0
            session.add_bbox_interaction(box, include_interaction=True)
            for pt in spec.get("exclude", []): session.add_point_interaction(pt, include_interaction=False)
            for pt in spec.get("include", []): session.add_point_interaction(pt, include_interaction=True)
            ks = np.nonzero(target.any((1, 2)))[0]
            # the root apex reached the end of the box: the box was too short, extend it
            if GROW_STEP and len(ks) and ks.max() >= box[0][1] - 2 and box[0][1] < target.shape[0] - 1:
                box[0][1] = min(box[0][1] + GROW_STEP, target.shape[0] - 1); continue
            break
        # keep only the largest connected piece (drops small specks the model sometimes adds)
        import SimpleITK as sitk
        cc = sitk.RelabelComponent(sitk.ConnectedComponent(sitk.GetImageFromArray(target)))
        mask = (sitk.GetArrayFromImage(cc) == 1).astype(np.uint8)
        sid = seg.GetSegmentation().AddEmptySegment(tooth, tooth)
        slicer.util.updateSegmentBinaryLabelmapFromArray(mask, seg, sid, low)
        target[:] = mask
        vol_mm3 = float(target.sum()) * np.prod(low.GetSpacing())
        log(tooth, "box", box, "slices", (int(ks.min()), int(ks.max())) if len(ks) else None,
            f"volume {vol_mm3:.0f} mm3", f"{time.time() - t0:.1f}s")

    # 4. show (and export)
    seg.GetSegmentation().SetConversionParameter("Smoothing factor", "0.3")
    seg.CreateClosedSurfaceRepresentation()
    if EXPORT_DIR:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        slicer.modules.segmentations.logic().ExportSegmentsClosedSurfaceRepresentationToFiles(
            EXPORT_DIR, seg, None, "STL", True, 1.0, False)
        log("STL files written to", EXPORT_DIR)
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalView)
    slicer.util.setSliceViewerLayers(background=low, fit=True)
    tv = slicer.app.layoutManager().threeDWidget(0).threeDView()
    tv.lookFromAxis(ctk.ctkAxesWidget.Anterior); tv.resetFocalPoint(); tv.resetCamera()
    log("done")
except Exception:
    log(traceback.format_exc())
