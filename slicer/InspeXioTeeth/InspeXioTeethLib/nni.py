"""nni.py - thin wrapper around an nnInteractive inference session for tooth segmentation."""
import numpy as np

from .geometry import largest_component


class ToothSegmenter:
    """Holds one nnInteractive session on one image ([k, j, i] float array)."""

    def __init__(self, image, device=None, log=print):
        import torch
        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
        from nnInteractive import model_management as mm
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model_dir = mm.ensure_model_available(mm.get_default_model_id())
        self.session = nnInteractiveInferenceSession(device=dev, use_torch_compile=False, verbose=False,
                                                     do_autozoom=True)
        self.session.initialize_from_trained_model_folder(str(model_dir))
        image = np.asarray(image, np.float32)
        self.shape = image.shape
        self.session.set_image(image[None])
        self.target = np.zeros(image.shape, np.uint8)
        self.session.set_target_buffer(self.target)
        log(f"nnInteractive ready on {dev} ({model_dir})")

    def clip_box(self, box):
        out = []
        for (a, b), n in zip(box, self.shape):
            a, b = int(max(0, min(a, n - 1))), int(max(1, min(b, n)))
            out.append([a, max(b, a + 1)])
        return out

    def clip_point(self, p):
        return [int(max(0, min(round(v), n - 1))) for v, n in zip(p, self.shape)]

    def segment(self, box, include=(), exclude=()):
        """box [[k0,k1],[j0,j1],[i0,i1]] (half-open; one axis of size 1); points [k, j, i]. Returns a bool mask."""
        self.session.reset_interactions()
        self.target[:] = 0
        self.session.add_bbox_interaction(self.clip_box(box), include_interaction=True)
        for p in exclude:
            self.session.add_point_interaction(self.clip_point(p), include_interaction=False)
        for p in include:
            self.session.add_point_interaction(self.clip_point(p), include_interaction=True)
        if not self.target.any():
            return self.target.astype(bool)
        return largest_component(self.target > 0)
