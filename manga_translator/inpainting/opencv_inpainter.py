from .inpainter import InpainterBase
import cv2
import numpy as np
from PIL import Image
from ..utils.common import pil2cv, cv2pil
import matplotlib.pyplot as plt
from ..schemas.interface import OCRResult, TranslationResult

class OpenCVInpainter(InpainterBase):
    def __init__(self, device="cpu", method="telea"):
        super().__init__(device)
        if method == "telea":
            self.method = cv2.INPAINT_TELEA
        elif method == "ns":
            self.method = cv2.INPAINT_NS
        else:
            raise ValueError("Invalid inpainting method. Choose 'telea' or 'ns'.")

    def _inpaint(self, image, mask, **kwargs):
        image = pil2cv(image)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=self.method)
        inpainted_image = cv2pil(inpainted_image)
        return inpainted_image