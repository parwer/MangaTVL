import numpy as np
from typing import List, Tuple, Optional
from ..schemas.interface import OCRResult, DetectionResult

"""
No Longer Used - This file is a placeholder for potential future bubble-OCR matching implementation.
"""

def iou_and_overlap(boxA, boxB):
    # boxes: [xmin,ymin,xmax,ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union = areaA + areaB - inter_area
    iou = inter_area / (union + 1e-9) if union > 0 else 0.0
    overlap_pct_of_b = inter_area / (areaB + 1e-9) if areaB > 0 else 0.0
    return float(iou), float(overlap_pct_of_b), float(inter_area)


def center_in_box(box, other):
    # check if center of other is inside box
    cx = (other[0] + other[2]) / 2.0
    cy = (other[1] + other[3]) / 2.0
    return (box[0] <= cx <= box[2]) and (box[1] <= cy <= box[3])


def match_ocr_to_bubbles(bubble_detections: List[DetectionResult],
                         ocr_results: List[Tuple[str, List[float]]],
                         iou_thresh: float = 0.1,
                         overlap_thresh: float = 0.2,
                         allow_multi_assign: bool = False) -> List[Tuple[DetectionResult, List[str], List[OCRResult]]]:
    """
    Match OCR results (as tuples (text, bbox)) to detected bubbles (DetectionResult).

    For matched OCRs per bubble we aggregate into a single OCRResult:
      - text: concatenation of matched texts (joined by space)
      - boxes: appended list of bboxes from matched OCR tuples

    Returns only bubbles that have at least one matched OCR.

    Args:
        bubble_detections: list of DetectionResult objects (must have .bbox)
        ocr_results: list of tuples (text, [xmin,ymin,xmax,ymax])
        iou_thresh: IoU threshold
        overlap_thresh: overlap percent threshold (w.r.t OCR bbox)
        allow_multi_assign: if True, an OCR tuple can be assigned to multiple bubbles

    Returns:
        List of tuples (DetectionResult, [texts], [OCRResult])
          - texts: list of individual OCR text strings matched to that bubble
          - OCRResult: aggregated OCRResult objects (typically one per bubble)
    """

    # Prepare
    bubble_boxes = [det.bbox for det in bubble_detections]
    matched_texts: List[List[str]] = [[] for _ in bubble_boxes]
    matched_indices: List[List[int]] = [[] for _ in bubble_boxes]  # store indices of matched ocr_results

    if not ocr_results or not bubble_boxes:
        return []

    N_b = len(bubble_boxes)
    N_o = len(ocr_results)
    scores = np.zeros((N_b, N_o), dtype=float)

    # ensure ocr bboxes are arrays
    ocr_bboxes: List[Optional[List[float]]] = []
    for text, bbox in ocr_results:
        if bbox is None:
            ocr_bboxes.append(None)
        else:
            ocr_bboxes.append([float(b) for b in bbox])

    for i, b in enumerate(bubble_boxes):
        for j, ob in enumerate(ocr_bboxes):
            if ob is None:
                continue
            iou, overlap_pct, _ = iou_and_overlap(b, ob)
            center_flag = 1.0 if center_in_box(b, ob) else 0.0
            scores[i, j] = max(iou, overlap_pct, center_flag)

    if allow_multi_assign:
        for i in range(N_b):
            for j in range(N_o):
                ob = ocr_bboxes[j]
                if ob is None:
                    continue
                iou, overlap_pct, _ = iou_and_overlap(bubble_boxes[i], ob)
                if iou >= iou_thresh or overlap_pct >= overlap_thresh or center_in_box(bubble_boxes[i], ob):
                    matched_texts[i].append(ocr_results[j][0])
                    matched_indices[i].append(j)
    else:
        for j in range(N_o):
            best_i = int(np.argmax(scores[:, j]))
            best_score = float(scores[best_i, j])
            ob = ocr_bboxes[j]
            if ob is None:
                continue
            iou, overlap_pct, _ = iou_and_overlap(bubble_boxes[best_i], ob)
            if best_score >= max(iou_thresh, overlap_thresh) or center_in_box(bubble_boxes[best_i], ob):
                matched_texts[best_i].append(ocr_results[j][0])
                matched_indices[best_i].append(j)

    results = []

    for i, det in enumerate(bubble_detections):
        if len(matched_indices[i]) == 0:
            continue
        # aggregate texts and boxes
        texts = matched_texts[i]
        boxes: List[List[int]] = []
        for idx in matched_indices[i]:
            _, bbox = ocr_results[idx]
            if bbox is None:
                continue
            # ensure ints
            boxes.append([int(round(float(v))) for v in bbox])

        # concatenate texts into one string
        combined_text = " ".join(t for t in texts if t)

        results.append(OCRResult(
            text=combined_text,
            boxes=boxes,
            detection_result=det
        ))

    return results
