"""
InspeXio Teeth - 3D Slicer module for tooth segmentation and apicoectomy planning/scoring on Shimadzu
inspeXio micro-CT scans.

  1. Scan       load an inspeXio .cb folder and make a downsampled working volume of the teeth slab
  2. Segment    one tooth at a time with nnInteractive: a box on a plane through the tooth plus optional
                include/exclude points; prompts can be saved and re-run as a batch
  3. Plan       per root: apex, root axis and the resection plane at a set distance from the apex
                (3 mm by default), plus the planned bone-entry point when a bone segment is given
  4. Score      register a post-cut scan to the pre-op scan and measure the trial: resection level and
                angle, apex removed, cut into a wrong tooth, bone-window position; one CSV row per root

nnInteractive model weights are CC BY-NC-SA 4.0 (non-commercial).
"""
import csv, datetime, json, os, time, traceback

import numpy as np
import ctk, qt, slicer, vtk
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleLogic,
                                           ScriptedLoadableModuleTest, ScriptedLoadableModuleWidget)

TOOTH_ATTR = "InspeXioTeeth.Tooth"
ROLE_ATTR = "InspeXioTeeth.Role"          # box / include / exclude / plan
PROMPT_FORMAT = "inspexio-teeth-prompts/2"
PLAN_FORMAT = "inspexio-teeth-plan/1"


class InspeXioTeeth(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "InspeXio Teeth"
        parent.categories = ["Segmentation"]
        parent.dependencies = []
        parent.contributors = ["Nobuhiko Hata (Brigham and Women's Hospital)"]
        parent.helpText = (
            "Tooth segmentation (nnInteractive) and apicoectomy planning and scoring for Shimadzu inspeXio "
            "micro-CT scans. See <a href='https://github.com/nobyhata/inspeXio-3D-Teeth-Segmentation'>"
            "the repository</a>.")
        parent.acknowledgementText = (
            "Segmentation uses nnInteractive (Isensee et al., DKFZ); its model weights are licensed "
            "CC BY-NC-SA 4.0.")


# =============================================================================================== logic
class InspeXioTeethLogic(ScriptedLoadableModuleLogic):
    def __init__(self, log=print):
        ScriptedLoadableModuleLogic.__init__(self)
        self.log = log
        self._segmenter = None
        self._segmenterKey = None
        self.plans = {}          # tooth -> {"roots": [...], "entry": [...] or None, ...}

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def ijkToRas(volume):
        m = vtk.vtkMatrix4x4()
        volume.GetIJKToRASMatrix(m)
        return slicer.util.arrayFromVTKMatrix(m)

    @staticmethod
    def segmentMask(segNode, segId, volume):
        return slicer.util.arrayFromSegmentBinaryLabelmap(segNode, segId, volume).astype(bool)

    @staticmethod
    def segmentNames(segNode):
        s = segNode.GetSegmentation()
        return [(s.GetNthSegmentID(i), s.GetNthSegment(i).GetName()) for i in range(s.GetNumberOfSegments())]

    # ---------------------------------------------------------------- 1. scan
    def loadInspeXio(self, folder):
        from InspeXioTeethLib import cb2nrrd
        stacks = list(cb2nrrd.find_stacks(folder))
        if not stacks:
            raise ValueError(f"no numbered .cb slices under {folder}")
        stack = next((s for s in stacks if "noring" in os.path.basename(s).lower()), stacks[0])
        info = cb2nrrd.analyse(stack)
        path = cb2nrrd.convert(info, "header", None, False, False)
        self.log(f"loading {path} ({len(info['slices'])} slices, {info['pixel']} mm)")
        return slicer.util.loadVolume(path)

    def makeWorkingVolume(self, volume, k0=0, k1=None, d=2, name=None):
        """Block-average the slab k0..k1 by d (also reduces noise) into a new float volume."""
        full = slicer.util.arrayFromVolume(volume)
        k1 = full.shape[0] if not k1 or k1 > full.shape[0] else k1
        nk, nj, ni = (k1 - k0) // d, full.shape[1] // d, full.shape[2] // d
        low = np.empty((nk, nj, ni), np.float32)
        for k in range(nk):
            blk = full[k0 + k * d:k0 + (k + 1) * d, :nj * d, :ni * d].astype(np.float32)
            low[k] = blk.reshape(d, nj, d, ni, d).mean((0, 2, 4))
        m = vtk.vtkMatrix4x4()
        volume.GetIJKToRASMatrix(m)
        origin = m.MultiplyPoint(((d - 1) / 2, (d - 1) / 2, k0 + (d - 1) / 2, 1))[:3]
        for r in range(3):
            for c in range(3):
                m.SetElement(r, c, m.GetElement(r, c) * d)
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name or f"{volume.GetName()}_x{d}")
        node.SetIJKToRASMatrix(m)
        node.SetOrigin(origin)
        slicer.util.updateVolumeFromArray(node, low)
        node.CreateDefaultDisplayNodes()
        lo, hi = np.percentile(low[::4, ::4, ::4], [1, 99.9])
        node.GetDisplayNode().AutoWindowLevelOff()
        node.GetDisplayNode().SetWindowLevel(float(hi - lo), float((hi + lo) / 2))
        node.SetAttribute("InspeXioTeeth.Source", volume.GetID())
        node.SetAttribute("InspeXioTeeth.Downsample", str(d))
        self.log(f"working volume {node.GetName()}: {low.shape}, {node.GetSpacing()[0]:.4f} mm")
        return node

    # ---------------------------------------------------------------- 2. prompts
    def promptNodes(self, tooth, create=False):
        nodes = {}
        for role, cls in (("box", "vtkMRMLMarkupsROINode"), ("include", "vtkMRMLMarkupsFiducialNode"),
                          ("exclude", "vtkMRMLMarkupsFiducialNode")):
            n = None
            for cand in slicer.util.getNodesByClass(cls):
                if cand.GetAttribute(TOOTH_ATTR) == tooth and cand.GetAttribute(ROLE_ATTR) == role:
                    n = cand
                    break
            if n is None and create:
                n = slicer.mrmlScene.AddNewNodeByClass(cls, f"{tooth}_{role}")
                n.SetAttribute(TOOTH_ATTR, tooth)
                n.SetAttribute(ROLE_ATTR, role)
                n.CreateDefaultDisplayNodes()
                dn = n.GetDisplayNode()
                if role == "include":
                    dn.SetSelectedColor(0.1, 0.8, 0.1); dn.SetColor(0.1, 0.8, 0.1)
                elif role == "exclude":
                    dn.SetSelectedColor(0.9, 0.1, 0.1); dn.SetColor(0.9, 0.1, 0.1)
                if role != "box":
                    dn.SetPointLabelsVisibility(False)
            nodes[role] = n
        return nodes

    def promptTeeth(self):
        names = set()
        for n in slicer.util.getNodesByClass("vtkMRMLMarkupsROINode"):
            if n.GetAttribute(TOOTH_ATTR) and n.GetAttribute(ROLE_ATTR) == "box":
                names.add(n.GetAttribute(TOOTH_ATTR))
        return sorted(names)

    def roiToBox(self, roi, volume, plane="auto"):
        """ROI (RAS) -> half-open voxel box [[k0,k1],[j0,j1],[i0,i1]] flattened to one plane."""
        from InspeXioTeethLib.geometry import ras_to_kji
        b = [0.0] * 6
        roi.GetRASBounds(b)
        corners = np.array([[x, y, z] for x in b[0:2] for y in b[2:4] for z in b[4:6]])
        u = ras_to_kji(corners, self.ijkToRas(volume))
        lo, hi = np.floor(u.min(0) + 0.5 + 1e-4).astype(int), np.floor(u.max(0) + 0.5 + 1e-4).astype(int)
        hi = np.maximum(hi, lo + 1)
        axis = {"axial": 0, "coronal": 1, "sagittal": 2}.get(plane)
        if axis is None:
            axis = int(np.argmin(hi - lo))
        c = (lo[axis] + hi[axis] - 1) // 2
        lo[axis], hi[axis] = c, c + 1
        if sum((hi - lo) > 1) < 2:
            raise ValueError("the box must span two directions (draw it on a slice through the tooth)")
        return [[int(lo[x]), int(hi[x])] for x in range(3)]

    def pointsKji(self, node, volume):
        from InspeXioTeethLib.geometry import ras_to_kji
        if node is None or node.GetNumberOfControlPoints() == 0:
            return []
        P = np.array([node.GetNthControlPointPositionWorld(i) for i in range(node.GetNumberOfControlPoints())])
        return np.round(ras_to_kji(P, self.ijkToRas(volume))).astype(int).tolist()

    def savePrompts(self, path, plane="auto"):
        teeth = {}
        for tooth in self.promptTeeth():
            n = self.promptNodes(tooth)
            b = [0.0] * 6
            n["box"].GetRASBounds(b)
            pts = lambda node: ([list(node.GetNthControlPointPositionWorld(i))
                                 for i in range(node.GetNumberOfControlPoints())] if node else [])
            teeth[tooth] = {"box_ras": [b[0:2], b[2:4], b[4:6]], "include_ras": pts(n["include"]),
                            "exclude_ras": pts(n["exclude"])}
        with open(path, "w") as f:
            json.dump({"_format": PROMPT_FORMAT, "plane": plane, "teeth": teeth}, f, indent=1)
        self.log(f"saved prompts for {len(teeth)} teeth to {path}")

    def loadPrompts(self, path, volume):
        """Load a v2 (RAS) prompt file, or a legacy file of voxel boxes on this working volume."""
        from InspeXioTeethLib.geometry import kji_to_ras
        data = json.load(open(path))
        M = self.ijkToRas(volume)
        legacy = data.get("_format") != PROMPT_FORMAT
        teeth = data if legacy else data["teeth"]
        for tooth, spec in teeth.items():
            if tooth.startswith("_"):
                continue
            nodes = self.promptNodes(tooth, create=True)
            for role in ("include", "exclude"):
                nodes[role].RemoveAllControlPoints()
            if legacy:
                spec = spec if isinstance(spec, dict) else {"box": spec}
                (k0, k1), (j0, j1), (i0, i1) = spec["box"]
                c = kji_to_ras([[k0 - 0.5, j0 - 0.5, i0 - 0.5], [k1 - 0.5, j1 - 0.5, i1 - 0.5]], M)
                lo, hi = c.min(0), c.max(0)
                inc = kji_to_ras(spec.get("include", []), M) if spec.get("include") else []
                exc = kji_to_ras(spec.get("exclude", []), M) if spec.get("exclude") else []
            else:
                bx = np.array(spec["box_ras"])
                lo, hi = bx[:, 0], bx[:, 1]
                inc, exc = spec.get("include_ras", []), spec.get("exclude_ras", [])
            roi = nodes["box"]
            roi.SetCenter(((lo + hi) / 2).tolist())
            roi.SetSize((hi - lo).tolist())
            for role, P in (("include", inc), ("exclude", exc)):
                for p in P:
                    nodes[role].AddControlPoint(vtk.vtkVector3d(*[float(x) for x in p]))
        self.log(f"loaded prompts for {len([t for t in teeth if not t.startswith('_')])} teeth from {path}")
        return [t for t in teeth if not t.startswith("_")]

    # ---------------------------------------------------------------- 2. segment
    def segmenter(self, volume):
        key = (volume.GetID(), volume.GetImageData().GetMTime())
        if self._segmenter is None or self._segmenterKey != key:
            from InspeXioTeethLib.nni import ToothSegmenter
            self._segmenter = None
            self.log("starting nnInteractive (first time: model download) ...")
            slicer.app.processEvents()
            self._segmenter = ToothSegmenter(slicer.util.arrayFromVolume(volume), log=self.log)
            self._segmenterKey = key
        return self._segmenter

    def segmentTooth(self, volume, segNode, tooth, plane="auto"):
        nodes = self.promptNodes(tooth)
        if nodes["box"] is None:
            raise ValueError(f"{tooth}: no box")
        box = self.roiToBox(nodes["box"], volume, plane)
        inc, exc = self.pointsKji(nodes["include"], volume), self.pointsKji(nodes["exclude"], volume)
        t0 = time.time()
        mask = self.segmenter(volume).segment(box, inc, exc)
        segNode.SetReferenceImageGeometryParameterFromVolumeNode(volume)
        seg = segNode.GetSegmentation()
        sid = seg.GetSegmentIdBySegmentName(tooth)
        if not sid:
            sid = seg.AddEmptySegment(tooth, tooth)
        slicer.util.updateSegmentBinaryLabelmapFromArray(mask.astype(np.uint8), segNode, sid, volume)
        vol = float(mask.sum()) * float(np.prod(volume.GetSpacing()))
        self.log(f"{tooth}: box {box}, {len(inc)} include / {len(exc)} exclude points, "
                 f"{vol:.2f} mm3, {time.time() - t0:.1f} s")
        return sid

    def exportStl(self, segNode, folder):
        os.makedirs(folder, exist_ok=True)
        segNode.GetSegmentation().SetConversionParameter("Smoothing factor", "0.3")
        segNode.CreateClosedSurfaceRepresentation()
        slicer.modules.segmentations.logic().ExportSegmentsClosedSurfaceRepresentationToFiles(
            folder, segNode, None, "STL", True, 1.0, False)
        self.log(f"STL files written to {folder}")

    # ---------------------------------------------------------------- 3. plan
    def computePlan(self, volume, segNode, toothId, hintsNode=None, resect_mm=3.0, axis_len_mm=0.0,
                    boneId=None, sideNode=None):
        from InspeXioTeethLib import geometry as g
        M = self.ijkToRas(volume)
        tooth = segNode.GetSegmentation().GetSegment(toothId).GetName()
        mask = self.segmentMask(segNode, toothId, volume)
        hints = [None]
        if hintsNode is not None and hintsNode.GetNumberOfControlPoints():
            hints = [list(hintsNode.GetNthControlPointPositionWorld(i))
                     for i in range(hintsNode.GetNumberOfControlPoints())]
        roots = [g.plan_root(mask, M, apex_hint=h, resect_mm=resect_mm, axis_len_mm=axis_len_mm or None)
                 for h in hints]
        entry = None
        if boneId:
            solid = self.segmentMask(segNode, boneId, volume)
            for sid, _ in self.segmentNames(segNode):
                if sid != boneId:
                    solid |= self.segmentMask(segNode, sid, volume)
            side = list(sideNode.GetNthControlPointPositionWorld(0)) if (
                sideNode is not None and sideNode.GetNumberOfControlPoints()) else None
            target = np.mean([r["plane_center"] for r in roots], 0)
            entry = g.plan_entry(solid, M, target, side)
        plan = dict(_format=PLAN_FORMAT, tooth=tooth, volume=volume.GetName(), roots=roots, entry=entry,
                    created=datetime.datetime.now().isoformat(timespec="seconds"))
        self.plans[tooth] = plan
        self.showPlan(plan)
        for n, r in enumerate(roots, 1):
            self.log(f"{tooth} root {n}: apex {np.round(r['apex'], 2).tolist()}, axis "
                     f"{np.round(r['axis'], 3).tolist()}, plane at {r['resect_mm']:.1f} mm, section "
                     f"{r['section_diameter_mm']:.2f} mm")
        if entry:
            self.log(f"{tooth} planned entry {np.round(entry, 2).tolist()}")
        return plan

    def _planNode(self, cls, name, tooth):
        n = slicer.util.getFirstNodeByClassByName(cls, name)
        if n is None:
            n = slicer.mrmlScene.AddNewNodeByClass(cls, name)
            n.CreateDefaultDisplayNodes()
        n.SetAttribute(TOOTH_ATTR, tooth)
        n.SetAttribute(ROLE_ATTR, "plan")
        n.RemoveAllControlPoints()
        dn = n.GetDisplayNode()
        if cls != "vtkMRMLMarkupsFiducialNode":          # planes and lines: no handles or name labels
            dn.SetPropertiesLabelVisibility(False)
            dn.SetPointLabelsVisibility(False)
            dn.SetGlyphScale(1.0)
        n.SetLocked(True)
        return n

    @staticmethod
    def setPlane(node, centre, normal, size):
        normal = np.asarray(normal, float) / np.linalg.norm(normal)
        u = np.cross(normal, [1, 0, 0] if abs(normal[0]) < 0.9 else [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        c = np.asarray(centre, float)
        try:
            node.SetPlaneType(node.PlaneType3Points)
        except Exception:
            pass
        for p in (c, c + u * size / 2, c + v * size / 2):
            node.AddControlPoint(vtk.vtkVector3d(*p))
        try:
            node.SetSizeMode(node.SizeModeAbsolute)
            node.SetPlaneBounds(-size / 2, size / 2, -size / 2, size / 2)
        except Exception:
            pass

    def showPlan(self, plan):
        tooth = plan["tooth"]
        apexes = self._planNode("vtkMRMLMarkupsFiducialNode", f"plan_{tooth}_apex", tooth)
        apexes.GetDisplayNode().SetSelectedColor(1.0, 0.85, 0.0)
        for n, r in enumerate(plan["roots"], 1):
            i = apexes.AddControlPoint(vtk.vtkVector3d(*r["apex"]))
            apexes.SetNthControlPointLabel(i, f"apex {n}")
            line = self._planNode("vtkMRMLMarkupsLineNode", f"plan_{tooth}_root{n}_axis", tooth)
            a = np.asarray(r["axis"])
            line.AddControlPoint(vtk.vtkVector3d(*r["apex"]))
            line.AddControlPoint(vtk.vtkVector3d(*(np.asarray(r["apex"]) + a * max(r["axis_len_mm"], 2 * r["resect_mm"]))))
            line.GetDisplayNode().SetSelectedColor(1.0, 0.85, 0.0)
            plane = self._planNode("vtkMRMLMarkupsPlaneNode", f"plan_{tooth}_root{n}_plane", tooth)
            d = r["section_diameter_mm"]
            self.setPlane(plane, r["plane_center"], r["plane_normal"], 2.5 * d if d == d and d > 0 else 3.0)
            plane.GetDisplayNode().SetSelectedColor(0.0, 0.8, 1.0)
            plane.GetDisplayNode().SetOpacity(0.5)
        if plan.get("entry"):
            e = self._planNode("vtkMRMLMarkupsFiducialNode", f"plan_{tooth}_entry", tooth)
            i = e.AddControlPoint(vtk.vtkVector3d(*plan["entry"]))
            e.SetNthControlPointLabel(i, "entry")
            e.GetDisplayNode().SetSelectedColor(1.0, 0.4, 0.0)

    def savePlan(self, tooth, path):
        with open(path, "w") as f:
            json.dump(self.plans[tooth], f, indent=1)
        self.log(f"plan for {tooth} saved to {path}")

    def loadPlan(self, path):
        plan = json.load(open(path))
        if plan.get("_format") != PLAN_FORMAT:
            raise ValueError("not an InspeXio Teeth plan file")
        self.plans[plan["tooth"]] = plan
        self.showPlan(plan)
        self.log(f"loaded plan for {plan['tooth']} ({len(plan['roots'])} roots)")
        return plan

    # ---------------------------------------------------------------- 4. score
    def registerAndResample(self, post, working, register=True, init="Off", segNode=None):
        """Post-cut scan -> pre-op working grid (rigid BRAINSFit, then BRAINSResample).

        With a segmentation, the registration only looks at the teeth (and bone) and a margin around them.
        """
        d = int(working.GetAttribute("InspeXioTeeth.Downsample") or 1)
        moving = post
        dims = post.GetImageData().GetDimensions()
        if d > 1 and post.GetSpacing()[0] < working.GetSpacing()[0] * 0.75:
            moving = self.makeWorkingVolume(post, 0, dims[2], d, name=f"{post.GetName()}_x{d}")
        tfm = None
        if register:
            from InspeXioTeethLib import geometry as g
            tfm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", f"{post.GetName()}_to_preop")
            p = dict(fixedVolume=working.GetID(), movingVolume=moving.GetID(), linearTransform=tfm.GetID(),
                     transformType="Rigid", initializeTransformMode=init, samplingPercentage=0.2,
                     numberOfIterations=3000, minimumStepLength=0.0001)
            maskNode = None
            if segNode is not None and segNode.GetSegmentation().GetNumberOfSegments():
                solid = np.zeros(slicer.util.arrayFromVolume(working).shape, bool)
                for sid, _ in self.segmentNames(segNode):
                    solid |= self.segmentMask(segNode, sid, working)
                margin = max(2, int(round(1.0 / working.GetSpacing()[0])))   # about 1 mm
                solid = g.dilate(solid, margin)
                maskNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "registration_mask")
                m = vtk.vtkMatrix4x4(); working.GetIJKToRASMatrix(m); maskNode.SetIJKToRASMatrix(m)
                slicer.util.updateVolumeFromArray(maskNode, solid.astype(np.uint8))
                p.update(fixedBinaryVolume=maskNode.GetID(), maskProcessingMode="ROI")
            self.log("registering post-cut scan (rigid) ...")
            slicer.app.processEvents()
            cli = slicer.cli.runSync(slicer.modules.brainsfit, None, p)
            if maskNode is not None:
                slicer.mrmlScene.RemoveNode(maskNode)
            if cli.GetStatus() & cli.ErrorsMask:
                raise RuntimeError("registration failed: " + cli.GetErrorText())
        out = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"{post.GetName()}_on_preop")
        p = dict(inputVolume=moving.GetID(), referenceVolume=working.GetID(), outputVolume=out.GetID(),
                 pixelType="float", interpolationMode="Linear")
        if tfm:
            p["warpTransform"] = tfm.GetID()
        cli = slicer.cli.runSync(slicer.modules.brainsresample, None, p)
        if cli.GetStatus() & cli.ErrorsMask:
            raise RuntimeError("resampling failed: " + cli.GetErrorText())
        return out, tfm

    def scoreTrial(self, working, segNode, toothId, postOnGrid, boneId=None, tau=0.0, trial=None, csvPath=None):
        from InspeXioTeethLib import geometry as g
        tooth = segNode.GetSegmentation().GetSegment(toothId).GetName()
        if tooth not in self.plans:
            raise ValueError(f"no plan for {tooth}: compute or load one first")
        plan = self.plans[tooth]
        M = self.ijkToRas(working)
        teeth = {name: self.segmentMask(segNode, sid, working)
                 for sid, name in self.segmentNames(segNode) if sid != boneId}
        bone = self.segmentMask(segNode, boneId, working) if boneId else None
        post = slicer.util.arrayFromVolume(postOnGrid)
        if not tau:
            air = None
            if bone is not None:
                solid = bone.copy()
                for m in teeth.values():
                    solid |= m
                air = ~g.dilate(solid, 2)
            tau = g.auto_threshold(post, teeth[tooth], air)
        pre = slicer.util.arrayFromVolume(working)
        r = g.evaluate_cut(teeth, tooth, plan["roots"], post, M, tau, bone=bone, entry_plan=plan.get("entry"),
                           pre=pre)
        self.showResult(tooth, r)
        rows = self.resultRows(tooth, r, trial or {}, postOnGrid.GetName())
        if csvPath:
            new = not os.path.exists(csvPath)
            with open(csvPath, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                if new:
                    w.writeheader()
                w.writerows(rows)
            self.log(f"{len(rows)} row(s) appended to {csvPath}")
        return r, rows

    def showResult(self, tooth, r):
        for n, root in enumerate(r["roots"], 1):
            if root.get("cut_center"):
                pl = self._planNode("vtkMRMLMarkupsPlaneNode", f"result_{tooth}_root{n}_cut", tooth)
                self.setPlane(pl, root["cut_center"], root["cut_normal"], 3.0)
                pl.GetDisplayNode().SetSelectedColor(1.0, 0.1, 0.1)
                pl.GetDisplayNode().SetOpacity(0.5)
        if r.get("window_center"):
            w = self._planNode("vtkMRMLMarkupsFiducialNode", f"result_{tooth}_window", tooth)
            i = w.AddControlPoint(vtk.vtkVector3d(*r["window_center"]))
            w.SetNthControlPointLabel(i, "window")
            w.GetDisplayNode().SetSelectedColor(1.0, 0.1, 0.1)

    @staticmethod
    def resultRows(tooth, r, trial, postName):
        rows = []
        fmt = lambda x: "" if x is None else (round(x, 3) if isinstance(x, float) else x)
        for root in r["roots"]:
            wrong = ";".join(r["wrong_tooth"])
            entry = r.get("entry_error_mm")
            checks = trial.get("checks")
            ok = (not wrong) and (entry is None or entry <= 2.0) and (checks in (None, "") or int(checks) <= 2)
            rows.append({
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "trial": trial.get("trial", ""), "operator": trial.get("operator", ""),
                "condition": trial.get("condition", ""), "checks": fmt(checks), "time_s": fmt(trial.get("time_s")),
                "tooth": tooth, "root": root["root"], "wrong_tooth": wrong,
                "planned_level_mm": fmt(root["planned_level_mm"]), "level_mm": fmt(root.get("level_mm")),
                "level_error_mm": fmt(root.get("level_error_mm")), "angle_deg": fmt(root.get("angle_deg")),
                "lateral_offset_mm": fmt(root.get("lateral_offset_mm")), "apex_removed": root["apex_removed"],
                "apical_remnant_mm3": fmt(root.get("apical_remnant_mm3")),
                "removed_target_mm3": fmt(r["removed_mm3"][tooth]), "bone_removed_mm3": fmt(r.get("bone_removed_mm3")),
                "window_diameter_mm": fmt(r.get("window_diameter_mm")), "entry_error_mm": fmt(entry),
                "pass_2mm_2checks": ok, "threshold": fmt(r["threshold"]), "post_scan": postName,
                "note": root.get("note", r.get("note", ""))})
        return rows


# =============================================================================================== widget
class InspeXioTeethWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = InspeXioTeethLogic(log=self.log)
        L = self.layout

        # ---- 1. scan
        box1 = ctk.ctkCollapsibleButton(); box1.text = "1. Scan"; L.addWidget(box1)
        f1 = qt.QFormLayout(box1)
        self.scanDir = ctk.ctkPathLineEdit(); self.scanDir.filters = ctk.ctkPathLineEdit.Dirs
        self.scanDir.settingKey = "InspeXioTeeth/scanDir"
        f1.addRow("inspeXio folder:", self.scanDir)
        b = qt.QPushButton("Load scan"); b.clicked.connect(self.onLoadScan); f1.addRow(b)
        self.scanSel = self.nodeCombo("vtkMRMLScalarVolumeNode"); f1.addRow("Scan volume:", self.scanSel)
        row = qt.QHBoxLayout()
        self.k0 = qt.QSpinBox(); self.k0.setRange(0, 100000); self.k0.setValue(0)
        self.k1 = qt.QSpinBox(); self.k1.setRange(0, 100000); self.k1.setValue(0)
        self.k1.setSpecialValueText("last")
        self.ds = qt.QSpinBox(); self.ds.setRange(1, 4); self.ds.setValue(2)
        for w, lab in ((self.k0, "slices"), (self.k1, "to"), (self.ds, "downsample")):
            row.addWidget(qt.QLabel(lab)); row.addWidget(w)
        f1.addRow(row)
        b = qt.QPushButton("Make working volume"); b.clicked.connect(self.onMakeWorking); f1.addRow(b)
        self.workSel = self.nodeCombo("vtkMRMLScalarVolumeNode"); f1.addRow("Working volume:", self.workSel)

        # ---- 2. segment
        box2 = ctk.ctkCollapsibleButton(); box2.text = "2. Segment teeth (nnInteractive)"; L.addWidget(box2)
        f2 = qt.QFormLayout(box2)
        self.segSel = self.nodeCombo("vtkMRMLSegmentationNode", add=True); f2.addRow("Segmentation:", self.segSel)
        self.tooth = qt.QComboBox(); self.tooth.editable = True
        self.tooth.lineEdit().setPlaceholderText("tooth name, e.g. tooth_46")
        f2.addRow("Tooth:", self.tooth)
        self.plane = qt.QComboBox(); self.plane.addItems(["auto", "sagittal", "coronal", "axial"])
        self.plane.toolTip = "plane the 2D box is flattened to (auto: the thinnest side of the box)"
        f2.addRow("Box plane:", self.plane)
        row = qt.QHBoxLayout()
        for text, role in (("Draw box", "box"), ("+ include point", "include"), ("- exclude point", "exclude")):
            b = qt.QPushButton(text); b.clicked.connect(lambda _=None, r=role: self.onPlace(r)); row.addWidget(b)
        f2.addRow(row)
        row = qt.QHBoxLayout()
        b = qt.QPushButton("Segment tooth"); b.clicked.connect(self.onSegmentTooth); row.addWidget(b)
        b = qt.QPushButton("Segment all"); b.clicked.connect(self.onSegmentAll); row.addWidget(b)
        b = qt.QPushButton("Clear points"); b.clicked.connect(self.onClearPoints); row.addWidget(b)
        f2.addRow(row)
        row = qt.QHBoxLayout()
        b = qt.QPushButton("Load prompts..."); b.clicked.connect(self.onLoadPrompts); row.addWidget(b)
        b = qt.QPushButton("Save prompts..."); b.clicked.connect(self.onSavePrompts); row.addWidget(b)
        b = qt.QPushButton("Export STL..."); b.clicked.connect(self.onExportStl); row.addWidget(b)
        f2.addRow(row)

        # ---- 3. plan
        box3 = ctk.ctkCollapsibleButton(); box3.text = "3. Plan apicoectomy"; L.addWidget(box3)
        f3 = qt.QFormLayout(box3)
        self.planTooth = qt.QComboBox(); f3.addRow("Tooth to resect:", self.planTooth)
        self.boneSeg = qt.QComboBox(); f3.addRow("Bone segment:", self.boneSeg)
        self.boneSeg.toolTip = "optional; needed for the planned entry point and the bone-window score"
        b = qt.QPushButton("Refresh segment lists"); b.clicked.connect(self.refreshSegments); f3.addRow(b)
        self.hints = self.nodeCombo("vtkMRMLMarkupsFiducialNode", add=True, none=True)
        self.hints.toolTip = "one point near each root apex (multi-rooted teeth); empty: single root, found automatically"
        row = qt.QHBoxLayout(); row.addWidget(self.hints)
        b = qt.QPushButton("Place"); b.clicked.connect(lambda: self.placeOn(self.hints, persistent=True)); row.addWidget(b)
        f3.addRow("Apex hints:", row)
        self.side = self.nodeCombo("vtkMRMLMarkupsFiducialNode", add=True, none=True)
        self.side.toolTip = "optional point on the approach side (e.g. buccal) for the planned entry"
        row = qt.QHBoxLayout(); row.addWidget(self.side)
        b = qt.QPushButton("Place"); b.clicked.connect(lambda: self.placeOn(self.side, persistent=False)); row.addWidget(b)
        f3.addRow("Approach side:", row)
        self.resect = qt.QDoubleSpinBox(); self.resect.setRange(0.1, 20); self.resect.setValue(3.0)
        self.resect.setSuffix(" mm"); f3.addRow("Resection from apex:", self.resect)
        self.axisLen = qt.QDoubleSpinBox(); self.axisLen.setRange(0, 30); self.axisLen.setValue(0)
        self.axisLen.setSuffix(" mm"); self.axisLen.setSpecialValueText("auto")
        self.axisLen.toolTip = "length of root from the apex used to fit the root axis"
        f3.addRow("Axis fit length:", self.axisLen)
        row = qt.QHBoxLayout()
        b = qt.QPushButton("Compute plan"); b.clicked.connect(self.onPlan); row.addWidget(b)
        b = qt.QPushButton("Save plan..."); b.clicked.connect(self.onSavePlan); row.addWidget(b)
        b = qt.QPushButton("Load plan..."); b.clicked.connect(self.onLoadPlan); row.addWidget(b)
        f3.addRow(row)

        # ---- 4. score
        box4 = ctk.ctkCollapsibleButton(); box4.text = "4. Score a trial (post-cut scan)"; L.addWidget(box4)
        f4 = qt.QFormLayout(box4)
        self.postDir = ctk.ctkPathLineEdit(); self.postDir.filters = ctk.ctkPathLineEdit.Dirs
        row = qt.QHBoxLayout(); row.addWidget(self.postDir)
        b = qt.QPushButton("Load"); b.clicked.connect(self.onLoadPost); row.addWidget(b)
        f4.addRow("Post-cut folder:", row)
        self.postSel = self.nodeCombo("vtkMRMLScalarVolumeNode"); f4.addRow("Post-cut scan:", self.postSel)
        self.register = qt.QCheckBox("register to the pre-op scan (rigid)"); self.register.checked = True
        self.init = qt.QComboBox(); self.init.addItems(["Off", "useMomentsAlign", "useGeometryAlign"])
        self.init.toolTip = "Off: the model went back into the scanner in the same position"
        row = qt.QHBoxLayout(); row.addWidget(self.register); row.addWidget(qt.QLabel("start:")); row.addWidget(self.init)
        f4.addRow(row)
        self.tau = qt.QDoubleSpinBox(); self.tau.setRange(0, 1e6); self.tau.setDecimals(0); self.tau.setValue(0)
        self.tau.setSpecialValueText("auto"); f4.addRow("Removed below:", self.tau)
        self.trial = qt.QLineEdit(); f4.addRow("Trial ID:", self.trial)
        self.operator = qt.QLineEdit(); f4.addRow("Operator:", self.operator)
        self.condition = qt.QComboBox(); self.condition.editable = True; self.condition.addItems(["A", "B", "C", "D", "E"])
        f4.addRow("Condition:", self.condition)
        row = qt.QHBoxLayout()
        self.checks = qt.QSpinBox(); self.checks.setRange(0, 99)
        self.timeS = qt.QDoubleSpinBox(); self.timeS.setRange(0, 1e5); self.timeS.setSuffix(" s")
        row.addWidget(qt.QLabel("checks")); row.addWidget(self.checks)
        row.addWidget(qt.QLabel("time to root exposure")); row.addWidget(self.timeS)
        f4.addRow(row)
        self.csvPath = ctk.ctkPathLineEdit(); self.csvPath.filters = ctk.ctkPathLineEdit.Files
        self.csvPath.nameFilters = ["CSV (*.csv)"]; self.csvPath.settingKey = "InspeXioTeeth/csv"
        f4.addRow("Results CSV:", self.csvPath)
        b = qt.QPushButton("Score trial"); b.clicked.connect(self.onScore); f4.addRow(b)

        # ---- log
        self.out = qt.QPlainTextEdit(); self.out.setReadOnly(True); self.out.setMinimumHeight(160)
        L.addWidget(self.out)
        L.addStretch(1)
        self.segSel.connect("currentNodeChanged(vtkMRMLNode*)", lambda _: self.refreshSegments())

    # ---------------------------------------------------------------- widget helpers
    def nodeCombo(self, cls, add=False, none=False):
        c = slicer.qMRMLNodeComboBox()
        c.nodeTypes = [cls]
        c.addEnabled = add; c.removeEnabled = add; c.renameEnabled = add
        c.noneEnabled = none
        c.setMRMLScene(slicer.mrmlScene)
        return c

    def log(self, msg):
        print(msg)
        try:
            self.out.appendPlainText(str(msg))
            slicer.app.processEvents()
        except Exception:
            pass

    def run(self, fn, *a):
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            return fn(*a)
        except Exception as e:
            self.log("ERROR: " + str(e))
            traceback.print_exc()
        finally:
            qt.QApplication.restoreOverrideCursor()

    def need(self, combo, what):
        n = combo.currentNode()
        if n is None:
            raise ValueError(f"select {what} first")
        return n

    def currentTooth(self):
        t = self.tooth.currentText.strip()
        if not t:
            t = f"tooth_{self.tooth.count + 1:02d}"
            self.tooth.addItem(t); self.tooth.setCurrentText(t)
        return t

    def placeOn(self, combo, persistent):
        n = combo.currentNode()
        if n is None:
            n = combo.addNode()
        self.startPlace(n, persistent)

    @staticmethod
    def startPlace(node, persistent):
        sel = slicer.app.applicationLogic().GetSelectionNode()
        sel.SetReferenceActivePlaceNodeClassName(node.GetClassName())
        sel.SetActivePlaceNodeID(node.GetID())
        inter = slicer.app.applicationLogic().GetInteractionNode()
        inter.SetPlaceModePersistence(1 if persistent else 0)
        inter.SetCurrentInteractionMode(inter.Place)

    def refreshSegments(self):
        seg = self.segSel.currentNode()
        keepT, keepB = self.planTooth.currentText, self.boneSeg.currentText
        self.planTooth.clear(); self.boneSeg.clear(); self.boneSeg.addItem("(none)", "")
        if seg is None:
            return
        for sid, name in self.logic.segmentNames(seg):
            self.planTooth.addItem(name, sid); self.boneSeg.addItem(name, sid)
        for combo, keep in ((self.planTooth, keepT), (self.boneSeg, keepB)):
            i = combo.findText(keep)
            if i >= 0:
                combo.setCurrentIndex(i)
        known = {self.tooth.itemText(i) for i in range(self.tooth.count)}
        for t in self.logic.promptTeeth():
            if t not in known:
                self.tooth.addItem(t)

    def segIds(self):
        tid = self.planTooth.itemData(self.planTooth.currentIndex)
        bid = self.boneSeg.itemData(self.boneSeg.currentIndex) or None
        if not tid:
            raise ValueError("pick the tooth to resect (Refresh segment lists)")
        if tid == bid:
            raise ValueError("the tooth and the bone segment must differ")
        return tid, bid

    # ---------------------------------------------------------------- callbacks
    def onLoadScan(self):
        def go():
            node = self.logic.loadInspeXio(self.scanDir.currentPath)
            self.scanSel.setCurrentNode(node)
            self.k1.setValue(node.GetImageData().GetDimensions()[2])
        self.run(go)

    def onMakeWorking(self):
        def go():
            node = self.logic.makeWorkingVolume(self.need(self.scanSel, "a scan volume"), self.k0.value,
                                                self.k1.value or None, self.ds.value)
            self.workSel.setCurrentNode(node)
            slicer.util.setSliceViewerLayers(background=node, fit=True)
        self.run(go)

    def onPlace(self, role):
        def go():
            tooth = self.currentTooth()
            n = self.logic.promptNodes(tooth, create=True)[role]
            if role == "box":
                n.RemoveAllControlPoints()
            self.startPlace(n, persistent=(role != "box"))
            self.log(f"{tooth}: place the {role} " + ("(click two corners on a slice through the tooth, "
                     "crown tip to just past the root apex)" if role == "box" else "points, right-click to stop"))
        self.run(go)

    def onClearPoints(self):
        n = self.logic.promptNodes(self.currentTooth())
        for role in ("include", "exclude"):
            if n[role] is not None:
                n[role].RemoveAllControlPoints()

    def segNode(self):
        seg = self.segSel.currentNode()
        if seg is None:
            seg = self.segSel.addNode()
            seg.CreateDefaultDisplayNodes()
        return seg

    def onSegmentTooth(self):
        def go():
            work = self.need(self.workSel, "the working volume")
            self.logic.segmentTooth(work, self.segNode(), self.currentTooth(), self.plane.currentText)
            self.refreshSegments()
        self.run(go)

    def onSegmentAll(self):
        def go():
            work = self.need(self.workSel, "the working volume")
            seg = self.segNode()
            for t in self.logic.promptTeeth():
                self.logic.segmentTooth(work, seg, t, self.plane.currentText)
            self.refreshSegments()
        self.run(go)

    def onLoadPrompts(self):
        path = qt.QFileDialog.getOpenFileName(None, "Load prompts", "", "JSON (*.json)")
        if path:
            self.run(lambda: (self.logic.loadPrompts(path, self.need(self.workSel, "the working volume")),
                              self.refreshSegments()))

    def onSavePrompts(self):
        path = qt.QFileDialog.getSaveFileName(None, "Save prompts", "prompts.json", "JSON (*.json)")
        if path:
            self.run(self.logic.savePrompts, path, self.plane.currentText)

    def onExportStl(self):
        folder = qt.QFileDialog.getExistingDirectory(None, "Folder for STL files")
        if folder:
            self.run(self.logic.exportStl, self.need(self.segSel, "a segmentation"), folder)

    def onPlan(self):
        def go():
            tid, bid = self.segIds()
            self.logic.computePlan(self.need(self.workSel, "the working volume"), self.need(self.segSel, "a segmentation"),
                                   tid, self.hints.currentNode(), self.resect.value, self.axisLen.value, bid,
                                   self.side.currentNode())
        self.run(go)

    def onSavePlan(self):
        tooth = self.planTooth.currentText
        if tooth not in self.logic.plans:
            self.log("compute or load a plan first"); return
        path = qt.QFileDialog.getSaveFileName(None, "Save plan", f"plan_{tooth}.json", "JSON (*.json)")
        if path:
            self.run(self.logic.savePlan, tooth, path)

    def onLoadPlan(self):
        path = qt.QFileDialog.getOpenFileName(None, "Load plan", "", "JSON (*.json)")
        if path:
            self.run(self.logic.loadPlan, path)

    def onLoadPost(self):
        self.run(lambda: self.postSel.setCurrentNode(self.logic.loadInspeXio(self.postDir.currentPath)))

    def onScore(self):
        def go():
            work = self.need(self.workSel, "the working volume")
            post = self.need(self.postSel, "the post-cut scan")
            tid, bid = self.segIds()
            onGrid, _ = self.logic.registerAndResample(post, work, self.register.checked, self.init.currentText,
                                                       self.segSel.currentNode())
            trial = dict(trial=self.trial.text, operator=self.operator.text, condition=self.condition.currentText,
                         checks=self.checks.value, time_s=self.timeS.value)
            r, rows = self.logic.scoreTrial(work, self.need(self.segSel, "a segmentation"), tid, onGrid, bid,
                                            self.tau.value, trial, self.csvPath.currentPath or None)
            for row in rows:
                self.log(f"root {row['root']}: level {row['level_mm']} mm (error {row['level_error_mm']}), "
                         f"angle {row['angle_deg']} deg, apex removed {row['apex_removed']}, wrong tooth "
                         f"'{row['wrong_tooth']}', entry error {row['entry_error_mm']} mm, pass {row['pass_2mm_2checks']}")
            slicer.util.setSliceViewerLayers(background=work, foreground=onGrid, foregroundOpacity=0.5)
        self.run(go)


# =============================================================================================== test
class InspeXioTeethTest(ScriptedLoadableModuleTest):
    def runTest(self):
        import importlib, sys
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, "Testing"))
        import test_geometry
        importlib.reload(test_geometry)
        for name in dir(test_geometry):
            if name.startswith("test_"):
                getattr(test_geometry, name)()
                self.delayDisplay(f"{name} passed")
        self.delayDisplay("InspeXio Teeth geometry tests passed")
