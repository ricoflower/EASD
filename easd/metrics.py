"""VOC all-point AP50 and optional official COCO evaluation on detection arrays."""

import contextlib
import io

import numpy as np


def normalized_xywh_to_xyxy(boxes, shape):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    scale = np.array((shape[1], shape[0]), dtype=np.float32)
    return np.concatenate(
        (
            (boxes[:, :2] - boxes[:, 2:4] / 2) * scale,
            (boxes[:, :2] + boxes[:, 2:4] / 2) * scale,
            boxes[:, 4:5],
        ),
        axis=1,
    )


def _voc_iou(box, targets):
    # Preserve the inclusive-pixel convention of the original VOC evaluator.
    intersection = np.maximum(
        np.minimum(box[2:], targets[:, 2:]) - np.maximum(box[:2], targets[:, :2]) + 1, 0
    ).prod(axis=1)
    area = np.maximum(box[2:] - box[:2] + 1, 0).prod()
    target_areas = np.maximum(targets[:, 2:] - targets[:, :2] + 1, 0).prod(axis=1)
    return intersection / np.maximum(area + target_areas - intersection, 1e-12)


def voc_ap50(predictions, ground_truth, class_names, iou_threshold=0.5):
    """Inputs are per-image Nx6 predictions and Mx5 pixel-xyxy,class ground truth."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must contain the same images")
    per_class = {}
    for category, name in enumerate(class_names):
        targets = [boxes[boxes[:, 4] == category, :4] for boxes in ground_truth]
        positives = sum(len(boxes) for boxes in targets)
        if not positives:
            continue
        matched = [np.zeros(len(boxes), dtype=bool) for boxes in targets]
        candidates = [
            (image, row)
            for image, boxes in enumerate(predictions)
            for row in boxes
            if int(row[5]) == category
        ]
        candidates.sort(key=lambda item: float(item[1][4]), reverse=True)
        true_positive = np.zeros(len(candidates))
        false_positive = np.zeros(len(candidates))
        for index, (image, row) in enumerate(candidates):
            if not len(targets[image]):
                false_positive[index] = 1
                continue
            overlaps = _voc_iou(row[:4], targets[image])
            best = int(overlaps.argmax())
            if overlaps[best] >= iou_threshold and not matched[image][best]:
                true_positive[index] = 1
                matched[image][best] = True
            else:
                false_positive[index] = 1
        tp, fp = true_positive.cumsum(), false_positive.cumsum()
        recall = tp / positives
        precision = tp / np.maximum(tp + fp, 1e-12)
        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([0.0], precision, [0.0]))
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        changes = np.where(recall[1:] != recall[:-1])[0]
        per_class[name] = float(
            np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1])
        )
    return {
        "metric": "VOC AP50",
        "value": float(np.mean(list(per_class.values()))) if per_class else 0.0,
        "per_class": per_class,
        "samples": len(ground_truth),
    }


def coco_ap(predictions, ground_truth, class_names, image_shapes):
    """Use pycocotools' COCO AP50:95 without temporary text/JSON conversion files."""
    if not len(predictions) == len(ground_truth) == len(image_shapes):
        raise ValueError("Predictions, ground truth, and image shapes must have equal lengths")
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise ImportError(
            "COCO evaluation requires: pip install -r requirements-eval.txt"
        ) from error
    dataset = {
        "info": {},
        "images": [],
        "annotations": [],
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(class_names)],
    }
    detections = []
    for image_id, (predicted, actual, shape) in enumerate(
        zip(predictions, ground_truth, image_shapes)
    ):
        dataset["images"].append({"id": image_id, "height": shape[0], "width": shape[1]})
        for box in actual:
            width, height = float(box[2] - box[0]), float(box[3] - box[1])
            dataset["annotations"].append(
                dict(
                    id=len(dataset["annotations"]) + 1,
                    image_id=image_id,
                    category_id=int(box[4]) + 1,
                    bbox=[float(box[0]), float(box[1]), width, height],
                    area=width * height,
                    iscrowd=0,
                )
            )
        for box in predicted:
            detections.append(
                dict(
                    image_id=image_id,
                    category_id=int(box[5]) + 1,
                    score=float(box[4]),
                    bbox=[
                        float(box[0]),
                        float(box[1]),
                        float(box[2] - box[0]),
                        float(box[3] - box[1]),
                    ],
                )
            )
    if not detections or not dataset["annotations"]:
        return {"metric": "COCO AP50:95", "value": 0.0, "ap50": 0.0, "samples": len(ground_truth)}
    with contextlib.redirect_stdout(io.StringIO()):
        actual = COCO()
        actual.dataset = dataset
        actual.createIndex()
        predicted = actual.loadRes(detections)
        evaluator = COCOeval(actual, predicted, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "metric": "COCO AP50:95",
        "value": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "samples": len(ground_truth),
    }
