from abc import ABC, abstractmethod
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from ..utils.common import cv2pil, pil2cv

from ..schemas.interface import OCRResult, TranslationResult

class InpainterBase(ABC):
    def __init__(self, device="cpu"):
        self.device = device

    def inpaint(self, image, inputs: list[TranslationResult] | list[OCRResult]):
        inputs = self.parse_inputs(inputs)
        mask = self._masking(image, inputs)
        inpainted_image = self._inpaint(image, mask, inputs=inputs)
        return inpainted_image
        
    
    def parse_inputs(self, inputs: list[TranslationResult] | list[OCRResult]) -> list[OCRResult]:
        if isinstance(inputs, list) and all(isinstance(item, TranslationResult) for item in inputs):
            tmp_inputs = [item.ocr_result for item in inputs]
            inputs = tmp_inputs
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

