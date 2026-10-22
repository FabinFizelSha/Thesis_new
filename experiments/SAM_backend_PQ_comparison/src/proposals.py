"""Prompt grid -> thresholded masks -> NMS -> label image. Identical for every model; only the wrapper in models.py differs."""
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pq_eval import masks_to_label_image

BASE = dict(input_scale=1.0, points_per_side=7, point_margin_fraction=0.167, skip_black_points=True, min_prompt_luma=4,
            mask_threshold=0.80, nms_iou=0.10, min_mask_pixels=3500, max_masks=49)


def _sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def prompt_grid(img, cfg):
    h, w = img.shape[:2]
    n = cfg["points_per_side"]
    mx, my = round(cfg["point_margin_fraction"] * w), round(cfg["point_margin_fraction"] * h)
    xs = np.linspace(mx, w - 1 - mx, n); ys = np.linspace(my, h - 1 - my, n)
    pts = [(float(x), float(y)) for y in ys for x in xs]
    if cfg["skip_black_points"]:
        thr = 3 * cfg["min_prompt_luma"]
        pts = [(x, y) for x, y in pts if int(img[int(round(y)), int(round(x))].sum()) > thr]
    return pts


@torch.no_grad()
def run_frame(model, rgb, cfg):
    """rgb: full-resolution uint8 (H, W, 3). Returns (label int32 (H, W), scores float32 (n,), timing dict). Timing covers
    resize, encoder, all prompts, thresholding, NMS and painting; not disk I/O or saving."""
    H, W = rgb.shape[:2]
    model.reset_timers()
    t0 = _sync()
    s = cfg["input_scale"]
    img = rgb if s == 1.0 else cv2.resize(rgb, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    model.set_image(img)
    masks, scores = [], []
    for x, y in prompt_grid(img, cfg):
        logits, sc = model.predict_point(x, y)
        m = logits > cfg["mask_threshold"]
        if s != 1.0:   # back to full resolution, nearest neighbour; all later filters use full-resolution pixels
            m = F.interpolate(m[None, None].float(), size=(H, W), mode="nearest")[0, 0] > 0.5
        if int(m.sum()) >= cfg["min_mask_pixels"]:
            masks.append(m); scores.append(sc)
    keep = []
    if masks:
        M = torch.stack(masks); flat = M.reshape(len(masks), -1).float()
        area = flat.sum(1); inter = flat @ flat.T
        iou = (inter / (area[:, None] + area[None, :] - inter)).cpu().numpy(); area = area.cpu().numpy()
        sc_arr = np.array(scores, np.float64)
        order = np.lexsort((-area, -sc_arr))          # sort by (score, area) descending
        for i in order:
            if all(iou[i, j] <= cfg["nms_iou"] for j in keep): keep.append(int(i))
            if len(keep) >= cfg["max_masks"]: break
        kept = M[keep].cpu().numpy() if keep else np.zeros((0, H, W), bool)
    else:
        kept = np.zeros((0, H, W), bool)
    sc_keep = np.array([scores[i] for i in keep], np.float32)
    label = masks_to_label_image(kept, sc_keep, (H, W)) if len(keep) else np.zeros((H, W), np.int32)
    t1 = _sync()
    return label, sc_keep, dict(total_ms=(t1 - t0) * 1e3, enc_ms=model.enc_ms, dec_ms=float(sum(model.dec_ms)), n_masks=len(keep))
