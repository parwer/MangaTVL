import textwrap
from PIL import Image, ImageDraw, ImageFont

from ..utils.common import cv2pil

from ..schemas.interface import TranslationResult
from .extract_text_box import extract_text_box
from pythainlp import word_tokenize
import re

class TextRenderer:
    def __init__(self,
                 font_path=None,
                 tokenizer=word_tokenize,
                 max_font_size=100,
                 min_font_size=18,
                 display_mode="horizontal" # "vertical" or "horizontal" or "mixed"
                 ):
        self.font_path = font_path
        self.tokenizer = tokenizer
        self.max_font_size = max_font_size
        self.min_font_size = min_font_size
        self.display_mode = display_mode

    def render(self, image, inputs: list[TranslationResult]):
        if inputs is None or len(inputs) == 0:
            print("No inputs provided for rendering. Returning original image.")
            return image

        image_copy = cv2pil(image.copy())
        draw = ImageDraw.Draw(image_copy)
        for item in inputs:
            text = item.translated_text if item.translated_text else item.ocr_result.text
            box = extract_text_box(image_copy, item)
            self._render_single(draw, text, box)
        
        return image_copy

    def _render_single(self, draw, text, box):
        text = text.replace(" ", "-")  # Replace spaces with hyphens to preserve them during tokenization

        wrap_text, font_size = self.wrap_extraction(draw, text, box)
        print(wrap_text)
        print("---"*10)

        wrap_text = wrap_text.replace(" ", "")
        wrap_text = wrap_text.replace("-", " ") # Revert hyphens back to spaces for rendering
        font = ImageFont.truetype(self.font_path, font_size) if self.font_path else ImageFont.load_default()

        try:
            text_bbox = draw.multiline_textbbox((0, 0), wrap_text, font=font, align='center')
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except Exception:
            text_width, text_height = draw.multiline_textsize(wrap_text, font=font)

        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1

        x = x1 + (box_width - text_width) / 2
        y = y1 + (box_height - text_height) / 2

        print(f"Rendering text with font size: {font_size}")

        draw.multiline_text((x, y), wrap_text, font=font, fill=(0, 0, 0), align='center', stroke_width=5, stroke_fill=(255, 255, 255))
        return
        

    def wrap_extraction(self, draw, text, box):
        target_width = box[2] - box[0]
        target_height = box[3] - box[1]
        tokenized_text = self._tokenize_text(text)

        low = self.min_font_size
        high = self.max_font_size
        best_font_size = low
        best_wrap_text = None

        while low <= high:
            mid = (low + high) // 2
            fits, wrap_text = self._fits_in_box(draw, mid, tokenized_text, target_width, target_height)
            if fits:
                best_font_size = mid
                best_wrap_text = wrap_text
                low = mid + 1
            else:
                high = mid - 1
        
        if best_wrap_text is not None:
            return best_wrap_text, best_font_size
        else:
            wrap_text = ""
            i = 0
            token = tokenized_text.split(' ')
            while i < len(token):
                wrap_text += token[i]
                if i % 4 == 0 and i != 0:
                    wrap_text += "\n"
                i+=1
            
            return wrap_text, self.min_font_size
                

    def _tokenize_text(self, text):
        tokens = self.tokenizer(text, engine="newmm")
        tokenized_text = " ".join(tokens)
        return tokenized_text

    def _fits_in_box(self, draw, font_size, tokenized_text, target_width, target_height):
        try:
            font = ImageFont.truetype(self.font_path, font_size) if self.font_path else ImageFont.load_default()
        except Exception:
            try:
                font = ImageFont.truetype(self.font_path, font_size)
            except Exception:
                font = ImageFont.load_default()

        wrap_width = len(tokenized_text)

        while wrap_width > 0:
            wrap_lines = textwrap.wrap(tokenized_text, width=wrap_width, break_long_words=False)
            wrap_text = "\n".join(wrap_lines)

            try:
                text_bbox = draw.multiline_textbbox((0, 0), wrap_text, font=font, align='center')
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except Exception:
                text_width, text_height = draw.multiline_textsize(wrap_text, font=font)
            
            if text_width <= target_width and text_height <= target_height:
                return True, wrap_text
            
            wrap_width -= 1

        return False, None