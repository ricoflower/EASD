"""Spike datasets and YOLO target assignment. Labels are normalized xywh,class."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.spikes import (boxes_to_padded, clip_boxes, load_spikes, pad_spikes,
                          padded_shape, parse_annotation, raw_to_spike)


class YoloDataset(Dataset):
    def __init__(self, input_type, annotation_lines, input_shape, num_classes,
                 anchors, anchors_mask, epoch_length=1, mosaic=False, mixup=False,
                 mosaic_prob=0.0, mixup_prob=0.0, train=True, special_aug_ratio=0.7,
                 data_root=None):
        super().__init__()
        if mosaic or mixup:
            raise ValueError("EASD uses spike tensors without RGB Mosaic/MixUp augmentation")
        self.annotation_lines = annotation_lines
        self.input_shape = padded_shape(input_shape)
        self.num_classes = num_classes
        self.anchors = np.asarray(anchors, dtype=np.float32)
        self.anchors_mask = anchors_mask
        self.bbox_attrs = 5 + num_classes
        self.threshold = 4
        self.input_type = input_type
        self.length = len(annotation_lines)
        self.epoch_now = -1
        self.data_root = Path(data_root) if data_root else None

    def __len__(self):
        return self.length

    def RawToSpike(self, video_seq, h, w, flipud=True):
        return raw_to_spike(video_seq, h, w, flipud)

    def __getitem__(self, index):
        path, boxes = parse_annotation(self.annotation_lines[index])
        if self.data_root is not None and not path.is_absolute():
            path = self.data_root / path
        spikes = load_spikes(path, self.input_type)
        boxes = clip_boxes(boxes, self.num_classes)
        boxes = boxes_to_padded(boxes, spikes.shape[-2:], self.input_shape)
        spikes = pad_spikes(spikes, self.input_shape)
        return spikes, boxes, self.get_target(boxes)

    def get_near_points(self, x, y, i, j):
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]

    def get_target(self, targets):
        num_layers = len(self.anchors_mask)
        input_shape = np.array(self.input_shape, dtype='int32')
        grid_shapes = [input_shape // (16, 8, 4)[l] for l in range(num_layers)]
        y_true = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs), dtype='float32') for l in range(num_layers)]
        box_best_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)]
        if len(targets) == 0:
            return y_true
        for l in range(num_layers):
            in_h, in_w = grid_shapes[l]
            anchors = np.array(self.anchors) / (16, 8, 4)[l]
            batch_target = np.zeros_like(targets)
            batch_target[:, [0, 2]] = targets[:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[:, [1, 3]] * in_h
            batch_target[:, 4] = targets[:, 4]
            ratios_of_gt_anchors = np.expand_dims(batch_target[:, 2:4], 1) / np.expand_dims(anchors, 0)
            ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)
            ratios = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis=-1)
            max_ratios = np.max(ratios, axis=-1)
            for t, ratio in enumerate(max_ratios):
                over_threshold = ratio < self.threshold
                over_threshold[np.argmin(ratio)] = True
                for k, mask in enumerate(self.anchors_mask[l]):
                    if not over_threshold[mask]:
                        continue
                    i = int(np.floor(batch_target[t, 0]))
                    j = int(np.floor(batch_target[t, 1]))
                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]
                        if local_i >= in_w or local_i < 0 or local_j >= in_h or (local_j < 0):
                            continue
                        if box_best_ratio[l][k, local_j, local_i] != 0:
                            if box_best_ratio[l][k, local_j, local_i] > ratio[mask]:
                                y_true[l][k, local_j, local_i, :] = 0
                            else:
                                continue
                        c = int(batch_target[t, 4])
                        y_true[l][k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[l][k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[l][k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[l][k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[l][k, local_j, local_i, 4] = 1
                        y_true[l][k, local_j, local_i, c + 5] = 1
                        box_best_ratio[l][k, local_j, local_i] = ratio[mask]
        return y_true


def yolo_dataset_collate(batch):
    images, boxes, targets = zip(*batch)
    return (torch.stack(images),
            [torch.as_tensor(box, dtype=torch.float32) for box in boxes],
            [torch.from_numpy(np.stack(level)) for level in zip(*targets)])
