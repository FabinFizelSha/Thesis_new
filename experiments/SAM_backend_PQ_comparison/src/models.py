"""Common single-point prompt interface for NanoSAM (TensorRT) and SAM ViT-B (PyTorch).

    m.set_image(rgb_uint8_HxWx3)           # runs the image encoder
    logits, score = m.predict_point(x, y)  # x, y in pixels of the image given to set_image
        logits : float32 CUDA tensor (H, W) at the size of that image (un-thresholded mask logits, output token 0)
        score  : float, the model's predicted IoU for that mask
Timers (ms, GPU-synchronised): m.enc_ms (last set_image), m.dec_ms (list, one entry per predict_point since reset_timers()).
Thresholding, NMS and painting are done by the common code in proposals.py, not here.
"""
import os, time
import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False          # strict FP32 for SAM ViT-B
torch.backends.cuda.matmul.allow_tf32 = False

MODELS = os.path.expanduser("~/rsg_models")
ENGINES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "nanosam")


def _now():
    torch.cuda.synchronize()
    return time.perf_counter()


class _Timed:
    def __init__(self):
        self.enc_ms = 0.0
        self.dec_ms = []

    def reset_timers(self):
        self.enc_ms = 0.0
        self.dec_ms = []


class NanoSAM(_Timed):
    name = "nanosam"

    def __init__(self, enc=f"{ENGINES}/resnet18_image_encoder.engine", dec=f"{ENGINES}/mobile_sam_mask_decoder.engine"):
        super().__init__()
        from nanosam.utils.predictor import Predictor
        self.p = Predictor(enc, dec)

    def set_image(self, rgb):
        import PIL.Image
        t = _now()
        self.p.set_image(PIL.Image.fromarray(rgb))
        self.enc_ms = (_now() - t) * 1e3
        self._hw = rgb.shape[:2]

    @torch.no_grad()
    def predict_point(self, x, y):
        t = _now()
        masks, iou, _ = self.p.predict(np.array([[float(x), float(y)]]), np.array([1]))
        out = masks[0, 0].float()                  # index 0 = single-mask output token
        sc = float(iou[0, 0])
        self.dec_ms.append((_now() - t) * 1e3)
        return out, sc


class SamViTB(_Timed):
    name = "sam_vitb"

    def __init__(self, ckpt=f"{MODELS}/sam/sam_vit_b_01ec64.pth"):
        super().__init__()
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_b"](checkpoint=ckpt).cuda().eval()   # FP32
        self.p = SamPredictor(sam)

    @torch.no_grad()
    def set_image(self, rgb):
        t = _now()
        self.p.set_image(rgb, image_format="RGB")
        self.enc_ms = (_now() - t) * 1e3
        self._hw = rgb.shape[:2]

    @torch.no_grad()
    def predict_point(self, x, y):
        t = _now()
        pts = self.p.transform.apply_coords_torch(torch.tensor([[[float(x), float(y)]]], device="cuda"), self._hw)
        lab = torch.ones((1, 1), dtype=torch.int, device="cuda")
        masks, iou, _ = self.p.predict_torch(pts, lab, multimask_output=False, return_logits=True)
        out = masks[0, 0].float()
        sc = float(iou[0, 0])
        self.dec_ms.append((_now() - t) * 1e3)
        return out, sc


def make_model(name):
    return {"nanosam": NanoSAM, "sam_vitb": SamViTB}[name]()
