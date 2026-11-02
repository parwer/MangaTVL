from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..masking.textmask import TextMasker, refine_mask
from ..schemas.interface import DetectionResult
from .lama_inpainter import LamaInpainter


@dataclass
class EraseResult:
    clean: np.ndarray                         # BGR page with the regions' text removed
    masks: list[np.ndarray]                   # region glyph masks (grown by the repair pass)
    reverted: list[tuple[int, int]] = field(default_factory=list)  # (region index, residue px)
    repair_px: int = 0


def _poly(seg, shape, grow=0):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.asarray(seg, np.int32).reshape(-1, 1, 2)], 255)
    return cv2.dilate(m, np.ones((grow, grow), np.uint8)) if grow else m


class TextEraser:
    def __init__(self, masker: TextMasker, inpainter: LamaInpainter, repair_rounds: int = 2,
                 guard: bool = True, guard_near: int = 15):
        self.masker = masker
        self.inpainter = inpainter
        self.repair_rounds = repair_rounds
        self.guard = guard
        self.guard_near = guard_near

    def _left_text(self, clean, zone):
        return refine_mask(clean, self.masker.predict(clean), thresh=0.4) & zone

    def erase(self, bgr: np.ndarray, masks: list[np.ndarray], dets: list[DetectionResult],
              keep: list[int] | None = None, fast: bool = False) -> EraseResult:
        """Erase the regions in ``keep``, repair, and revert regions whose text survives (``fast``: no LaMa)."""
        keep = list(range(len(masks))) if keep is None else list(keep)
        masks = [m.copy() for m in masks]
        H, W = bgr.shape[:2]
        pre = bgr.copy()
        hole = np.zeros((H, W), np.uint8)
        for i in keep:
            rm, seg = masks[i], dets[i].segmentation
            # balloon-aware flat fill: flatness judged only on balloon pixels around the text (a
            # page-level ring also hits the outline / art when text sits near the edge)
            if seg and rm.any():
                inner = cv2.erode(_poly(seg, (H, W)), np.ones((7, 7), np.uint8))
                ring = cv2.dilate(rm, np.ones((9, 9), np.uint8)) & ~rm & inner
                px = bgr[ring > 0].astype(np.float32)
                if len(px) > 30 and px.std(0).max() < 10 and (rm & ~inner).sum() < 0.02 * rm.sum():
                    pre[rm > 0] = np.median(px, 0).astype(np.uint8)
                    self.inpainter.stats["flat"] += 1
                    continue
            hole |= rm
        clean = self.inpainter.erase(pre, hole, fast=fast)

        # repair / guard zone per region: a 41 px halo, clipped to the balloon (+7 px) if any
        zone = np.zeros((H, W), np.uint8)
        lims, inners = {}, {}
        for i in keep:
            rm = masks[i]
            if not rm.any():
                continue
            z = cv2.dilate(rm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
            if dets[i].segmentation:
                lims[i] = _poly(dets[i].segmentation, (H, W), grow=7)
                inners[i] = cv2.erode(lims[i], np.ones((15, 15), np.uint8))
                z &= lims[i] | rm
            zone |= z
        res = EraseResult(clean, masks)
        for _ in range(self.repair_rounds):
            left = self._left_text(res.clean, zone)
            n = int(np.count_nonzero(left))
            if n < 30:
                break
            res.repair_px += n
            for i in keep:  # the renderer must also avoid the newly found ink
                add = left & cv2.dilate(masks[i], np.ones((41, 41), np.uint8))
                masks[i] |= add & lims[i] if i in lims else add
            res.clean = self.inpainter.erase(res.clean, cv2.dilate(left, np.ones((5, 5), np.uint8)), fast=fast)
        if not self.guard:
            return res

        # CTD also "sees" text in a freshly blanked balloon from its shape alone: keep only residue
        # that is real ink (contrasts with its surroundings), slightly grown to join stroke pieces
        left = self._left_text(res.clean, zone)
        g = cv2.cvtColor(res.clean, cv2.COLOR_BGR2GRAY).astype(np.int16)
        ink = ((np.abs(g - cv2.medianBlur(g.astype(np.uint8), 31).astype(np.int16)) > 40) * 255).astype(np.uint8)
        left &= cv2.dilate(ink, np.ones((3, 3), np.uint8))
        if not left.any():
            return res
        k = self.guard_near
        for i in list(keep):
            rm = masks[i]
            # leftover letters sit on the region's own footprint (a wide halo catches panel borders)
            near = cv2.dilate(rm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            if i in inners:  # balloon: leftover letters are inside it, art touching the outline is not
                near &= inners[i]
            resid = int(np.count_nonzero(left & near))
            if resid > max(60, 0.04 * np.count_nonzero(rm)):
                others = np.zeros((H, W), np.uint8)
                for j in keep:
                    if j != i:
                        others |= masks[j]
                back = cv2.dilate(rm, np.ones((9, 9), np.uint8)) & ~others
                res.clean[back > 0] = bgr[back > 0]
                keep.remove(i)
                res.reverted.append((i, resid))
        return res
