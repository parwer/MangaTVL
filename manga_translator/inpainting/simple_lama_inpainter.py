from simple_lama_inpainting import SimpleLama
from .inpainter import InpainterBase

class SimpleLamaInpainter(InpainterBase):
    def __init__(self, device="cpu"):
        super().__init__(device=device)
        self.model = SimpleLama(device=device)

    def _inpaint(self, image, masks, **kwargs):
        inpainted_image = self.model(image, masks)
        return inpainted_image