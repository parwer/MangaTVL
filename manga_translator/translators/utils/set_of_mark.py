from PIL import Image, ImageDraw, ImageFont

TAG_COLOR = (230, 30, 30)


def _font(size: int):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def mark_image(image: Image.Image, boxes: list, max_side: int = 1280) -> Image.Image:
    """Return a downscaled RGB copy of ``image`` with box i outlined and tagged "i"."""
    im = image.convert("RGB")
    s = min(1.0, max_side / max(im.size))
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS) if s < 1 else im.copy()
    d = ImageDraw.Draw(im)
    font = _font(max(14, int(22 * max(im.size) / 1280)))
    for n, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(v * s) for v in box)
        d.rectangle((x1, y1, x2, y2), outline=TAG_COLOR, width=2)
        tb = d.textbbox((0, 0), str(n), font=font)
        tw, th = tb[2] - tb[0] + 8, tb[3] - tb[1] + 6
        # tag INSIDE the box's top-left corner: outside tags collide between touching
        # balloons and the model then reads the neighbour's text.
        d.rectangle((x1 + 1, y1 + 1, x1 + 1 + tw, y1 + 1 + th), fill=TAG_COLOR)
        d.text((x1 + 5, y1 + 2), str(n), fill="white", font=font)
    return im
