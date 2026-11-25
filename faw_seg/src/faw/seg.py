"""Shared full-resolution inference and metrics (same protocol as Phase 1 Exp 06 evaluation)."""
import numpy as np
import cv2

H, W = 480, 640


def positions(patch=320, stride=160):
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    return [(y, x) for y in ys for x in xs]


def predict_full(model, images_u8, patch=320, stride=160, chunk=8, thermals=None):
    """images_u8: list of HxWx3 uint8; thermals: optional list of HxWx2 thermal inputs (fusion models).
    Returns list of HxW float32 probabilities (overlap-averaged)."""
    pos = positions(patch, stride)
    out = []
    for c in range(0, len(images_u8), chunk):
        imgs = images_u8[c:c + chunk]
        batch = np.stack([im[y:y + patch, x:x + patch] for im in imgs for (y, x) in pos]).astype(np.float32) / 255.0
        if thermals is not None:
            tb = np.stack([t[y:y + patch, x:x + patch] for t in thermals[c:c + chunk] for (y, x) in pos]).astype(np.float32)
            p = model.predict([batch, tb], batch_size=8, verbose=0)[..., 0]
        else:
            p = model.predict(batch, batch_size=8, verbose=0)[..., 0]
        for k in range(len(imgs)):
            acc = np.zeros((H, W), np.float32)
            cnt = np.zeros((H, W), np.float32)
            for j, (y, x) in enumerate(pos):
                acc[y:y + patch, x:x + patch] += p[k * len(pos) + j]
                cnt[y:y + patch, x:x + patch] += 1
            out.append(acc / cnt)
    return out


def confusion(pred, gt):
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    return tp, fp, fn


def dice_iou(pred, gt):
    tp, fp, fn = confusion(pred, gt)
    d = 2 * tp + fp + fn
    return (2 * tp / d if d else 1.0), (tp / (tp + fp + fn) if (tp + fp + fn) else 1.0)


def image_metrics(pred, gt):
    tp, fp, fn = confusion(pred, gt)
    d, i = dice_iou(pred, gt)
    return {"dice": d, "iou": i,
            "precision": tp / (tp + fp) if tp + fp else (1.0 if not gt.any() else 0.0),
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "gt_pct": 100 * gt.mean(), "pred_pct": 100 * pred.mean(), "tp": tp, "fp": fp, "fn": fn}


def boundary_tolerant_dice(pred, gt, tol=3):
    """Dice where a predicted pixel within `tol` px of GT counts as correct and vice versa."""
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    gt_d = cv2.dilate(gt.astype(np.uint8), k) > 0
    pr_d = cv2.dilate(pred.astype(np.uint8), k) > 0
    p_ok = (pred & gt_d).sum(); g_ok = (gt & pr_d).sum()
    prec = p_ok / pred.sum() if pred.sum() else (1.0 if not gt.any() else 0.0)
    rec = g_ok / gt.sum() if gt.sum() else 1.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def region_metrics(pred, gt, min_overlap_px=1):
    """Region-level detection: a GT region is detected if any predicted pixel overlaps it;
    a predicted region is correct if it overlaps any GT pixel."""
    ng, lg = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    npd, lp = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    det = sum(1 for k in range(1, ng) if (pred[lg == k]).sum() >= min_overlap_px)
    cor = sum(1 for k in range(1, npd) if (gt[lp == k]).sum() >= min_overlap_px)
    return {"gt_regions": ng - 1, "gt_regions_detected": det, "pred_regions": npd - 1, "pred_regions_correct": cor}


def remove_small(pred, min_area):
    """Remove predicted connected regions smaller than min_area pixels (8-connectivity)."""
    if not min_area:
        return pred
    n, lab, st, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]
