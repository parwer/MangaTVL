import base64
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import ast

def xyxy2xywh(box):
    x1, y1, x2, y2 = box
    x = (x1 + x2) / 2  # center x
    y = (y1 + y2) / 2  # center y
    w = x2 - x1
    h = y2 - y1
    return (x, y, w, h)


def xywh2xyxy(box):
    x, y, w, h = box
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return (x1, y1, x2, y2)


def combine_bbox(bboxes):
    if len(bboxes) == 0:
        return np.array([0, 0, 0, 0])

    bboxes = np.array(bboxes)

    x_min = np.min(bboxes[:, 0])
    y_min = np.min(bboxes[:, 1])
    x_max = np.max(bboxes[:, 2])
    y_max = np.max(bboxes[:, 3])

    return np.array([x_min, y_min, x_max, y_max])


def inset_bbox(bbox, pad):
    """Shrink a bbox inward by `pad` pixels on each side.
    Returns (x1, y1, x2, y2) or None if the result would be degenerate."""
    x1, y1, x2, y2 = bbox
    x1, y1 = int(x1) + pad, int(y1) + pad
    x2, y2 = int(x2) - pad, int(y2) - pad
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def inscribed_rect(polygon, image_shape, padding=0):
    """Largest axis-aligned rectangle that fits entirely inside `polygon`.

    polygon: list/ndarray of [x, y] points (one polygon)
    image_shape: (h, w) of the source image — used to clip the polygon bbox
    padding: pixels to erode inward (e.g. for stroke + safety margin)

    Returns (x1, y1, x2, y2) in source-image coords, or None if no valid rect remains.
    """
    if polygon is None or len(polygon) < 3:
        return None

    pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
    img_h, img_w = image_shape[:2]

    px1 = max(0, int(pts[:, 0].min()))
    py1 = max(0, int(pts[:, 1].min()))
    px2 = min(img_w, int(pts[:, 0].max()) + 1)
    py2 = min(img_h, int(pts[:, 1].max()) + 1)
    bw, bh = px2 - px1, py2 - py1
    if bw <= 0 or bh <= 0:
        return None

    # Downsample large bubbles to bound work — accuracy loss is absorbed by `padding`.
    max_side = 200
    scale = max(1.0, max(bw, bh) / max_side)
    rw = max(1, int(round(bw / scale)))
    rh = max(1, int(round(bh / scale)))

    pts_local = pts.copy()
    pts_local[:, 0] = ((pts_local[:, 0] - px1) / scale).round().astype(np.int32)
    pts_local[:, 1] = ((pts_local[:, 1] - py1) / scale).round().astype(np.int32)

    mask = np.zeros((rh, rw), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_local.reshape(-1, 1, 2)], 255)

    if padding > 0:
        k = max(1, int(round(padding / scale)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 2 * k + 1))
        mask = cv2.erode(mask, kernel)

    if not mask.any():
        return None

    binary = (mask > 0).astype(np.int32)
    heights = np.zeros(rw, dtype=np.int32)
    best_area = 0
    best = (0, 0, 0, 0)

    for y in range(rh):
        heights = np.where(binary[y] > 0, heights + 1, 0)
        stack = []
        for x in range(rw + 1):
            h = int(heights[x]) if x < rw else 0
            start = x
            while stack and stack[-1][1] > h:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best_area:
                    best_area = area
                    best = (sx, y - sh + 1, x - 1, y)
                start = sx
            stack.append((start, h))

    if best_area == 0:
        return None

    rx1, ry1, rx2, ry2 = best
    x1 = int(round(rx1 * scale)) + px1
    y1 = int(round(ry1 * scale)) + py1
    x2 = int(round((rx2 + 1) * scale)) + px1
    y2 = int(round((ry2 + 1) * scale)) + py1
    return (x1, y1, x2, y2)


def refine_unit_value_type(value) -> list[int]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError(f"Invalid value format: {value}")
    if not isinstance(value, list):
        raise ValueError(f"Expected list[int], got {type(value)}")
    for item in value:
        if not isinstance(item, int):
            for i in item:
                if not isinstance(i, int):
                    raise ValueError(f"Expected list[int], got list[{type(i)}]")
    return value


def pil2cv(img) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img.astype(np.uint8)

    cv_image = np.array(img, dtype=np.uint8)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    return cv_image


def cv2pil(img):
    if isinstance(img, Image.Image):
        return img

    rgb_image = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    return pil_image


def convert_img_to_base64(image, quality=60):
    pattern = img_pattern(image)

    if pattern == 'base64':
        return image

    elif pattern == 'pil':
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=quality)
        b64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_string}"

    elif pattern == 'cv2':
        pil_image = cv2pil(image)
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=quality)
        b64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_string}"

    else:
        raise ValueError("Invalid image type")


def convert_base64_to_img(b64_string: str) -> Image.Image:
    if not isinstance(b64_string, str):
        if isinstance(b64_string, np.ndarray):
            return cv2pil(b64_string)
        elif isinstance(b64_string, Image.Image):
            return b64_string

    if isinstance(b64_string, str) and b64_string.startswith("data:image"):
        header, b64_data = b64_string.split(",", 1)
    else:
        b64_data = b64_string

    try:
        image_data = base64.b64decode(b64_data)
        image = Image.open(BytesIO(image_data))
    except Exception as e:
        raise ValueError("Invalid base64 image data") from e
    return image


def img_pattern(image) -> str:
    """Detect image type"""
    if isinstance(image, str) and image.startswith("http"):
        return 'url'
    elif isinstance(image, str) and image.startswith("data:image"):
        return 'base64'
    elif isinstance(image, Image.Image):
        return 'pil'
    elif isinstance(image, np.ndarray):
        return 'cv2'
    else:
        print(type(image))
        return 'unknown'


def load_image(image_path) -> Image.Image:
    # if image is a url
    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        img = Image.open(image_path).convert('RGB')
    return img


def show_image_with_boxes(image, boxes, cls_text: str=None, fig_size=(10, 10), color=None):
    """
        image pattern: PIL, cv2
        bboxes pattern: [[x1, y1, x2, y2], ...]
    """
    cv_image = pil2cv(image)
    cv_image = cv_image.copy()
    if cls_text is None:
        cls_text = ["" for _ in range(len(boxes))]

    for i, (box, cls) in enumerate(zip(boxes, cls_text)):
        x1, y1, x2, y2 = box
        color = (255, 0, 0) if color is None else color
        
        cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if cls:
            cv2.putText(cv_image, cls, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    pil_image = cv2pil(cv_image)
    plt.figure(figsize=fig_size)
    plt.imshow(pil_image)
    plt.axis('off')
    plt.show()


def show_image_with_polygons(image, polygons, cls_text=None, fig_size=(10, 10), color=None):
    """
        image pattern: PIL, cv2
        polygons pattern: [[[x1, y1], [x2, y2], ...], ...]
        Each polygon is a list of [x, y] points. None / empty polygons are skipped
        (handy when some DetectionResult.segmentation is None from a detect-only model).
    """
    cv_image = pil2cv(image).copy()
    if cls_text is None:
        cls_text = ["" for _ in range(len(polygons))]

    c = (255, 0, 0) if color is None else color
    for poly, cls in zip(polygons, cls_text):
        if poly is None or len(poly) == 0:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(cv_image, [pts], isClosed=True, color=c, thickness=2)
        if cls:
            x, y = int(poly[0][0]), int(poly[0][1])
            cv2.putText(cv_image, cls, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)

    pil_image = cv2pil(cv_image)
    plt.figure(figsize=fig_size)
    plt.imshow(pil_image)
    plt.axis('off')
    plt.show()


def show_images(images, titles=None, figsize=(15, 5), vertical=False):
    n = len(images)
    plt.figure(figsize=figsize)
    for i in range(n):
        if vertical:
            plt.subplot(n, 1, i + 1)
        else:
            plt.subplot(1, n, i + 1)
        plt.imshow(images[i])
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.show()


def resize_image(image, max_size=512):
    width, height = image.size
    if max(width, height) <= max_size:
        return image
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    return image.resize((new_width, new_height))

