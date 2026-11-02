from .inpainter import InpainterBase
from .opencv_inpainter import OpenCVInpainter
from .lama_inpainter import LamaInpainter
from .text_eraser import EraseResult, TextEraser


def __getattr__(name):  # simple-lama (torch) only when it is used
    if name == "SimpleLamaInpainter":
        from .simple_lama_inpainter import SimpleLamaInpainter
        return SimpleLamaInpainter
    raise AttributeError(name)
