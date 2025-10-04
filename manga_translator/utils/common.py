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

def show_image_with_boxes(image, boxes, cls_text, fig_size=(10, 10), colors=None):
    # image: PIL image
    cv_image = pil2cv(image)
    cv_image = cv_image.copy()
    if cls_text is None:
        cls_text = ["" for _ in range(len(boxes))]

    for i, (box, cls) in enumerate(zip(boxes, cls_text)):
        x1, y1, x2, y2 = box
        color = (255, 0, 0)
        cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if cls:
            cv2.putText(cv_image, cls, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
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

def combine_bbox(bboxes):
    bboxes = np.array(bboxes)

    x_min = np.min(bboxes[:, 0])
    y_min = np.min(bboxes[:, 1])
    x_max = np.max(bboxes[:, 2])
    y_max = np.max(bboxes[:, 3])

    return np.array([x_min, y_min, x_max, y_max])


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