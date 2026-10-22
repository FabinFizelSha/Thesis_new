#!/usr/bin/env python3
"""Class-agnostic Panoptic Quality (Kirillov et al., CVPR 2019) for SAM-type mask proposals.

Inputs per frame
    pred  : int label image (0 = no mask), non-overlapping by construction
    gt    : int label image of ground-truth instances (0 = no instance)
    void  : bool image of pixels that are ignored (unlabelled pixels and
            ground-truth regions below the minimum instance size)

Matching rule (PQ): a prediction and a ground-truth instance match iff IoU > 0.5.
With non-overlapping segments, each instance can have at most one such
prediction and vice versa, so the result does not depend on any ordering.
"""
import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------- ground truth
def gt_instances(semantic_rgb, min_pixels, ignore_colours=()):
    """Split the simulator's RGB class image into instances.

    Each exact RGB colour is one class (no channel averaging); each 8-connected
    region of a class is one instance. Regions below min_pixels become void.
    Returns (gt label image, void mask)."""
    h, w, _ = semantic_rgb.shape
    codes = (semantic_rgb[..., 0].astype(np.int64) << 16) | (semantic_rgb[..., 1].astype(np.int64) << 8) | semantic_rgb[..., 2]
    ignore = {(r << 16) | (g << 8) | b for r, g, b in ignore_colours}
    gt = np.zeros((h, w), np.int32)
    void = np.zeros((h, w), bool)
    next_id = 1
    for code in np.unique(codes):
        cls = codes == code
        if int(code) in ignore:
            void |= cls
            continue
        lab, n = ndimage.label(cls, structure=np.ones((3, 3), np.uint8))
        for k in range(1, n + 1):
            region = lab == k
            if region.sum() < min_pixels:
                void |= region
            else:
                gt[region] = next_id
                next_id += 1
    return gt, void


# --------------------------------------------------------------------------- predictions
def masks_to_label_image(masks, scores, shape):
    """Paint boolean masks into one label image; where masks overlap, the
    higher-scoring mask wins (painted last). Deterministic for fixed scores."""
    out = np.zeros(shape, np.int32)
    for new_id, idx in enumerate(np.argsort(scores, kind="stable"), start=1):
        out[masks[idx]] = new_id
    return out


# --------------------------------------------------------------------------- PQ per frame
def pq_frame(pred, gt, void, thr=0.5):
    """Return (list of IoUs of matched pairs, #FP, #FN) for one frame."""
    g, p, v = gt.ravel(), pred.ravel(), void.ravel()
    gt_ids = np.unique(g[g > 0])
    pr_ids = np.unique(p[p > 0])
    area_g = dict(zip(*np.unique(g[g > 0], return_counts=True)))
    area_p = dict(zip(*np.unique(p[p > 0], return_counts=True)))
    void_p = dict(zip(*np.unique(p[(p > 0) & v], return_counts=True)))

    both = (g > 0) & (p > 0)
    pairs, inter = np.unique(np.stack([g[both], p[both]]), axis=1, return_counts=True)

    ious, matched_g, matched_p = [], set(), set()
    for (gi, pj), n in zip(pairs.T, inter):
        # void pixels of the prediction are removed before computing IoU
        union = area_g[gi] + area_p[pj] - void_p.get(pj, 0) - n
        iou = n / union
        if iou > thr:
            ious.append(float(iou))
            matched_g.add(int(gi))
            matched_p.add(int(pj))

    fn = len(gt_ids) - len(matched_g)
    # unmatched predictions lying mostly (> 50 %) on void are ignored, not FP
    fp = sum(1 for j in pr_ids
             if int(j) not in matched_p and void_p.get(j, 0) / area_p[j] <= 0.5)
    return ious, fp, fn


def pq_summary(frames):
    """Dataset-level PQ/SQ/RQ from per-frame results [(ious, fp, fn), ...]."""
    ious = [x for f in frames for x in f[0]]
    tp = len(ious)
    fp = sum(f[1] for f in frames)
    fn = sum(f[2] for f in frames)
    denom = tp + 0.5 * fp + 0.5 * fn
    sq = float(np.mean(ious)) if tp else 0.0
    rq = tp / denom if denom else 0.0
    return {"PQ": sq * rq, "SQ": sq, "RQ": rq, "TP": tp, "FP": fp, "FN": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0}


def paired_bootstrap(frames_a, frames_b, n=10000, seed=0):
    """95 % CI of PQ(A) - PQ(B), resampling frames (same frames for both models)."""
    rng = np.random.default_rng(seed)
    k = len(frames_a)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, k, k)
        diffs[i] = pq_summary([frames_a[j] for j in idx])["PQ"] - pq_summary([frames_b[j] for j in idx])["PQ"]
    return np.percentile(diffs, [2.5, 97.5])
