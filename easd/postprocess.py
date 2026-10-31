"""Anchor-head decoding and class-wise NMS in raw sensor coordinates."""

import numpy as np
import torch
from torchvision.ops import nms

from easd.config import DEFAULT_ANCHOR_MASKS
from easd.spikes import padded_shape, padding_for


class DetectionPostProcessor:
    def __init__(
        self, anchors, num_classes, anchor_masks=DEFAULT_ANCHOR_MASKS, confidence=0.5, nms_iou=0.5
    ):
        if not 0 <= confidence <= 1 or not 0 <= nms_iou <= 1:
            raise ValueError("Confidence and NMS IoU thresholds must lie in [0,1]")
        self.anchors = np.asarray(anchors)
        self.num_classes = num_classes
        self.anchor_masks = tuple(tuple(mask) for mask in anchor_masks)
        self.confidence = confidence
        self.nms_iou = nms_iou

    def decode(self, head_logits, canvas_shape):
        """Return normalized xywh, object confidence, and class confidence."""
        outputs = []
        for mask, logits in zip(self.anchor_masks, head_logits):
            batch, _, height, width = logits.shape
            prediction = logits.reshape(batch, len(mask), self.num_classes + 5, height, width)
            prediction = prediction.permute(0, 1, 3, 4, 2).contiguous().sigmoid()
            yy, xx = torch.meshgrid(
                torch.arange(height, device=logits.device, dtype=logits.dtype),
                torch.arange(width, device=logits.device, dtype=logits.dtype),
                indexing="ij",
            )
            stride_y, stride_x = canvas_shape[0] / height, canvas_shape[1] / width
            anchors = logits.new_tensor(self.anchors[list(mask)])
            boxes = torch.empty_like(prediction[..., :4])
            boxes[..., 0] = prediction[..., 0] * 2.0 - 0.5 + xx
            boxes[..., 1] = prediction[..., 1] * 2.0 - 0.5 + yy
            boxes[..., 2] = (prediction[..., 2] * 2) ** 2 * (anchors[:, 0] / stride_x)[
                None, :, None, None
            ]
            boxes[..., 3] = (prediction[..., 3] * 2) ** 2 * (anchors[:, 1] / stride_y)[
                None, :, None, None
            ]
            scale = logits.new_tensor((width, height, width, height))
            outputs.append(
                torch.cat(
                    (
                        boxes.reshape(batch, -1, 4) / scale,
                        prediction[..., 4:].reshape(batch, -1, self.num_classes + 1),
                    ),
                    -1,
                )
            )
        if len(outputs) != 3 or len(head_logits) != 3:
            raise ValueError("Expected exactly three coarse-to-fine detection heads")
        return torch.cat(outputs, dim=1)

    def select(self, decoded, sensor_shape):
        """Filter normalized predictions, apply NMS, then remove canvas padding."""
        canvas = padded_shape(sensor_shape)
        left, _, top, _ = padding_for(sensor_shape, canvas)
        outputs = []
        for prediction in decoded:
            class_scores, categories = prediction[:, 5:].max(dim=1)
            scores = prediction[:, 4] * class_scores
            selected = scores >= self.confidence
            prediction, scores, categories = (
                prediction[selected],
                scores[selected],
                categories[selected],
            )
            if not len(prediction):
                outputs.append(np.empty((0, 6), dtype=np.float32))
                continue
            corners = torch.cat(
                (
                    prediction[:, :2] - prediction[:, 2:4] / 2,
                    prediction[:, :2] + prediction[:, 2:4] / 2,
                ),
                dim=1,
            )
            keep = []
            for category in categories.unique(sorted=True):
                indices = torch.where(categories == category)[0]
                keep.append(indices[nms(corners[indices], scores[indices], self.nms_iou)])
            keep = torch.cat(keep)
            # Preserve the original normalized->center/size->pixel arithmetic.
            box_xy = (corners[keep, :2] + corners[keep, 2:]) / 2
            box_wh = corners[keep, 2:] - corners[keep, :2]
            boxes = torch.cat((box_xy - box_wh / 2, box_xy + box_wh / 2), dim=1)
            boxes = boxes * boxes.new_tensor((canvas[1], canvas[0], canvas[1], canvas[0]))
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left).clamp(0, sensor_shape[1])
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top).clamp(0, sensor_shape[0])
            detections = torch.cat((boxes, scores[keep, None], categories[keep, None]), dim=1)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            outputs.append(detections[valid].float().cpu().numpy())
        return outputs

    @torch.no_grad()
    def __call__(self, head_logits, sensor_shape):
        return self.select(self.decode(head_logits, padded_shape(sensor_shape)), sensor_shape)
