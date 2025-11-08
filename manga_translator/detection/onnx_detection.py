import onnxruntime as ort
import numpy as np
from PIL import Image   
import cv2

from ..utils.common import pil2cv, xywh2xyxy
from ..schemas.interface import DetectionResult

class ONNXDetection:
    def __init__(self, model_path: str = "/home/parwer/MangaTVL/manga_translator/checkpoints/best5000.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = (self.input_shape[2], self.input_shape[3])  # (height, width)

        self.cls_names = {0: "bubble", 1: "free_text"}

    
        print(f"ONNX model loaded from {model_path}")
        print(f"Input shape: {self.input_size}")
    
    def detect(self, image: Image.Image | np.ndarray):
        boxes, confidences = self.run_inference(image)
        results = []
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2, class_id = box
            form_type = self.cls_names.get(int(class_id), "unknown")
            detection = DetectionResult(bbox=[int(x1), int(y1), int(x2), int(y2)], form_type=form_type)
            results.append(detection)
        return results
    
    def run_inference(self, image: Image.Image | np.ndarray):
        self.original_width, self.original_height = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
        input_tensor = self.preprocess(image)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        boxes, confidences = self.postprocess(outputs)
        return boxes, confidences

    def preprocess(self, image: Image.Image | np.ndarray):
        image = pil2cv(image)
        image = cv2.resize(image, self.input_size)
        image = image.transpose(2, 0, 1)  # HWC to CHW
        image = image.reshape(1, 3, self.input_size[0], self.input_size[1])
        image = image.astype('float32') / 255.0  # Normalize to [0, 1]
        return image

    def postprocess(self, outputs, conf_threshold=0.6):
        detections = outputs[0]
        # transpose as before (model output shape dependent)
        detections = detections.transpose()

        # filter detections
        detections_filtered = self.filter_detections(detections, conf_threshold)

        # ensure detections_filtered is a numpy array
        detections_filtered = np.array(detections_filtered)

        # If there are no detections (empty array), return empty results
        if detections_filtered.size == 0:
            return np.empty((0, 5)), np.empty((0,))

        # If a single detection was returned as 1D array, convert to 2D
        if detections_filtered.ndim == 1:
            detections_filtered = detections_filtered.reshape(1, -1)

        rescaled_boxes, confidences = self.rescale_back(detections_filtered, self.original_width, self.original_height)
        return rescaled_boxes, confidences

    def filter_detections(self, detections, conf_threshold=0.6):
        # detections expected shape: (N, D) where D >= 5
        # If detections is empty-like, return empty np.array
        if detections is None:
            return np.array([])

        detections = np.array(detections)
        if detections.size == 0:
            return np.array([])

        # handle two common layouts:
        #  - each detection has 5 values (cx, cy, w, h, conf_or_class)
        #  - or more than 5 (cx, cy, w, h, class_scores...)
        # Use robust handling and return 2D array of detections with appended class and conf if needed.
        try:
            # If detections is 2D and second dim equals 5
            if detections.ndim == 2 and detections.shape[1] == 5:
                considerable = [det for det in detections if det[4] >= conf_threshold]
                return np.array(considerable)
            else:
                # More than 5 dims: infer class id and score from tail
                A = []
                for det in detections:
                    # ensure det is array-like
                    det = np.array(det)
                    if det.size < 5:
                        continue
                    class_id = int(det[4:].argmax())
                    conf_score = float(det[4:].max())
                    new_det = np.append(det[:4], [class_id, conf_score])
                    A.append(new_det)
                A = np.array(A)
                if A.size == 0:
                    return np.array([])
                considerable = [det for det in A if det[-1] >= conf_threshold]
                return np.array(considerable)
        except Exception:
            # Fallback: return empty if something unexpected
            return np.array([])

    
    def rescale_back(self, detections_filtered, original_width, original_height):
        # detections_filtered assumed to be (N, 6) or (N,5) (we expect at least 5 columns)
        detections_filtered = np.array(detections_filtered)
        if detections_filtered.size == 0:
            return np.empty((0, 5)), np.empty((0,))

        if detections_filtered.ndim == 1:
            detections_filtered = detections_filtered.reshape(1, -1)

        # If only 5 columns (cx, cy, w, h, class_id/conf) we may need to handle arrangement
        # Ensure there are at least 5 columns; if 5, interpret as cx,cy,w,h,class_id
        if detections_filtered.shape[1] < 5:
            # cannot process, return empty
            return np.empty((0, 5)), np.empty((0,))

        # Support cases where last column is conf_score or class+conf (6 cols)
        # We'll try to extract class_id from column 4 and conf_score from last column when available
        if detections_filtered.shape[1] >= 6:
            cx = detections_filtered[:, 0]
            cy = detections_filtered[:, 1]
            w = detections_filtered[:, 2]
            h = detections_filtered[:, 3]
            class_id = detections_filtered[:, 4]
            conf_score = detections_filtered[:, -1]
        else:
            # shape == 5: assume [cx, cy, w, h, class_id_or_conf]
            cx = detections_filtered[:, 0]
            cy = detections_filtered[:, 1]
            w = detections_filtered[:, 2]
            h = detections_filtered[:, 3]
            # If the fifth column is a float (confidence), we set class_id to 0 and conf_score to that value
            last_col = detections_filtered[:, 4]
            # Heuristic: if values in last_col are >1 then probably class id; if <=1 then confidence
            if np.all(last_col <= 1.0):
                class_id = np.zeros_like(last_col)
                conf_score = last_col
            else:
                class_id = last_col
                conf_score = np.ones_like(last_col)

        # Rescale from model input normalized coords to original image size
        cx = cx / self.input_size[1] * original_width
        cy = cy / self.input_size[0] * original_height
        w = w / self.input_size[1] * original_width
        h = h / self.input_size[0] * original_height

        x1, y1, x2, y2 = xywh2xyxy((cx, cy, w, h))
        # x1,y1,x2,y2 may be arrays already - stack accordingly
        rescaled_boxes = np.column_stack((x1, y1, x2, y2, class_id))
        keep, keep_conf = self.MNS(rescaled_boxes, conf_score)
        return keep, keep_conf
    
    def MNS(self, boxes, conf_scores, iou_threshold=0.5):

        # boxes [[x1, y1, x2, y2], ...]
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = conf_scores.argsort()

        keep = []
        keep_conf = []

        while len(order) > 0:
            idx = order[-1]
            A = boxes[idx]
            conf = conf_scores[idx]

            order = order[:-1]

            xx1 = np.take(x1, indices=order)
            yy1 = np.take(y1, indices=order)
            xx2 = np.take(x2, indices=order)
            yy2 = np.take(y2, indices=order)

            keep.append(A)
            keep_conf.append(conf)

            # iou
            xx1 = np.maximum(x1[idx], xx1)
            yy1 = np.maximum(y1[idx], yy1)
            xx2 = np.minimum(x2[idx], xx2)
            yy2 = np.minimum(y2[idx], yy2)

            w = np.maximum(xx2-xx1, 0)
            h = np.maximum(yy2-yy1, 0)

            intersection = w*h

            # union area
            other_areas = np.take(areas, indices=order)
            union = areas[idx] + other_areas - intersection

            iou = intersection / union

            boleans = iou < iou_threshold
            order = order[boleans]
        
        return np.array(keep), np.array(keep_conf)