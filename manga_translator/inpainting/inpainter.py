from abc import ABC, abstractmethod
import numpy as np
import cv2
from PIL import Image

from ..utils.common import cv2pil, pil2cv

from ..schemas.interface import OCRResult, TranslationResult

class InpainterBase(ABC):
    def __init__(self, device="cpu", show_log=False):
        self.device = device
        self.show_log = show_log

    def inpaint(self, image, inputs: list[TranslationResult] | list[OCRResult]):
        if inputs is None or len(inputs) == 0:
            print("No inputs provided for inpainting. Returning original image.")
            return image

        inputs = self.parse_inputs(inputs)
        image, demoted = self._safe_fill_polygons(image, inputs)
        if self.show_log:
            n_safe = len(inputs) - len(demoted)
            print(f"[inpaint] polygon safe-fill: {n_safe}/{len(inputs)}, "
                  f"box-fallback: {len(demoted)}")
        if not demoted:
            return image
        mask = self._masking(image, demoted)
        inpainted_image = self._inpaint(image, mask, inputs=demoted)
        return inpainted_image

    def _safe_fill_polygons(self, image, inputs: list[OCRResult]):
        """Paint each bubble's safe interior white directly. Items whose polygon
        is missing or fails the border-uniformity safety check are returned in
        ``demoted`` so the caller can fall back to box-based cv2.inpaint."""
        cv_image = pil2cv(image).copy()
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        demoted: list[OCRResult] = []
        for item in inputs:
            poly = getattr(item.detection_result, "segmentation", None)
            if not poly:
                demoted.append(item)
                continue
            safe = self._pick_safe_polygon_mask(gray, poly)
            if safe is None:
                demoted.append(item)
                continue
            cv_image[safe > 0] = (255, 255, 255)
        return cv2pil(cv_image), demoted

    def _pick_safe_polygon_mask(
        self,
        image_gray: np.ndarray,
        polygon,
        max_std: float = 25.0,
        n_steps: int = 8,
        max_erode_frac: float = 0.12,
        min_area_frac: float = 0.3,
    ):
        """Adapted from PanelCleaner's pick_best_mask. Generate candidate masks
        by eroding the polygon inward by increasing amounts, score each by the
        standard deviation of source-image pixels at the mask boundary, return
        the lowest-deviation candidate. A low std means the boundary lands on
        uniform colour (bubble interior white) rather than the dark border line.
        Returns None if even the best candidate exceeds ``max_std`` — caller
        should treat that as 'too risky, skip'.

        Erode depth is adaptive: scaled to the polygon's diagonal so large
        bubbles (whose YOLO-seg polygon can overshoot by 30+ px) erode deep
        enough, while small bubbles don't over-erode past their safe interior.
        """
        h, w = image_gray.shape
        pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
        pw = int(pts[:, 0].max() - pts[:, 0].min())
        ph = int(pts[:, 1].max() - pts[:, 1].min())
        diag = float(np.hypot(pw, ph))
        max_erode = max(6, int(diag * max_erode_frac))
        erode_steps = sorted(set(
            int(round(max_erode * i / n_steps)) for i in range(n_steps + 1)
        ))

        base = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(base, [pts.reshape(-1, 1, 2)], 255)
        base_area = int(np.count_nonzero(base))
        if base_area == 0:
            return None

        gradient_kernel = np.ones((3, 3), np.uint8)
        candidates = []
        for erode_px in erode_steps:
            mask = base
            if erode_px > 0:
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (erode_px * 2 + 1, erode_px * 2 + 1),
                )
                mask = cv2.erode(base, k)
            # Skip candidates eroded past usefulness — guards against picking a
            # tiny over-eroded mask whose border happens to land on uniform pixels.
            if np.count_nonzero(mask) < base_area * min_area_frac:
                continue
            ring = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, gradient_kernel)
            border_pixels = image_gray[ring > 0]
            if len(border_pixels) == 0:
                continue
            candidates.append((float(border_pixels.std()), mask))

        if not candidates:
            return None
        best_std, best_mask = min(candidates, key=lambda x: x[0])
        return best_mask if best_std <= max_std else None
        
    
    def parse_inputs(self, inputs: list[TranslationResult] | list[OCRResult]) -> list[OCRResult]:
        if isinstance(inputs, list) and all(isinstance(item, TranslationResult) for item in inputs):
            tmp_inputs = [item.ocr_result for item in inputs]
            inputs = tmp_inputs
            if self.show_log:
                print("Converted TranslationResult to OCRResult for inpainting.")
        return inputs

    def _masking(self, image, inputs: list[OCRResult], expand_margin=2) -> np.ndarray:
        image = pil2cv(image)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for item in inputs:
            for box in item.boxes:
                box = self.expand_box(box, image.shape, expand_margin=expand_margin)
                x1, y1, x2, y2 = box
                cv2.rectangle(mask, (x1, y1), (x2, y2), color=255, thickness=-1)
        
        return mask

    def show_masks(self, image, inputs: list[OCRResult], expand_margin=2, masked_image=None, figsize=(8, 8)):
        import matplotlib.pyplot as plt  # debug-only; lazy import keeps it off the runtime path

        image = pil2cv(image)
        inputs = self.parse_inputs(inputs)
        mask = self._masking(image, inputs, expand_margin=expand_margin)
        if masked_image is None:
            masked_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
        masked_image = cv2pil(masked_image)
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].imshow(cv2pil(image))
        ax[0].set_title("Original Image")
        ax[0].axis('off')
        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title("Mask")
        ax[1].axis('off')
        plt.show()
    
    def get_masks(self, image, inputs: list[OCRResult], expand_margin=2) -> Image.Image:
        image = pil2cv(image)
        inputs = self.parse_inputs(inputs)
        mask = self._masking(image, inputs, expand_margin=expand_margin)
        return cv2pil(mask)

    def expand_box(self, box, image_shape, expand_margin=2):
        x1, y1, x2, y2 = box
        h, w = image_shape[:2]
        x1 = max(0, x1 - expand_margin)
        y1 = max(0, y1 - expand_margin)
        x2 = min(w, x2 + expand_margin)
        y2 = min(h, y2 + expand_margin)
        return (x1, y1, x2, y2)


    @abstractmethod
    def _inpaint(self, image, masks, **kwargs):
        raise NotImplementedError

