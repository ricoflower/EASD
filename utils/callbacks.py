"""Optional loss logging and VOC mAP evaluation on the padded spike canvas."""

from pathlib import Path
from importlib import import_module
import shutil

import numpy as np
import torch

from utils.dataloader import YoloDataset
from utils.spikes import padded_shape
from utils.utils_bbox import DecodeBox


class LossHistory:
    def __init__(self, log_dir, model=None, input_shape=(250, 400)):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.losses, self.val_loss = [], []
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(str(self.log_dir))

    def append_loss(self, epoch, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        for name, value in (("loss", loss), ("val_loss", val_loss)):
            self.writer.add_scalar(name, value, epoch)
            with (self.log_dir / f"epoch_{name}.txt").open("a") as stream:
                stream.write(f"{value}\n")


class EvalCallback:
    def __init__(self, input_type, net, input_shape, anchors, anchors_mask,
                 class_names, num_classes, val_lines, log_dir, cuda,
                 map_out_path=".temp_map_out", max_boxes=100, confidence=0.05,
                 nms_iou=0.5, letterbox_image=False, MINOVERLAP=0.5,
                 eval_flag=True, period=1, metric="voc"):
        self.net = net
        self.input_shape = padded_shape(input_shape)
        self.class_names = class_names
        self.num_classes = num_classes
        self.log_dir = Path(log_dir)
        self.map_out_path = self.log_dir / map_out_path
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.MINOVERLAP = MINOVERLAP
        self.eval_flag = eval_flag
        self.period = period
        if metric not in ("voc", "coco"):
            raise ValueError("metric must be 'voc' or 'coco'")
        if metric == "coco":
            import_module("pycocotools.cocoeval")  # Fail early if dependency is missing.
        self.metric = metric
        self.maps, self.epoches = [], []
        self.dataset = YoloDataset(input_type, val_lines, self.input_shape,
                                   num_classes, anchors, anchors_mask)
        self.bbox_util = DecodeBox(anchors, num_classes, self.input_shape, anchors_mask)

    @torch.no_grad()
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        device = next(self.net.parameters()).device
        outputs = self.net(image.unsqueeze(0).to(device))
        predictions = torch.cat(self.bbox_util.decode_box(outputs[:3]), dim=1)
        results = self.bbox_util.non_max_suppression(
            predictions, self.num_classes, self.input_shape, self.input_shape, False,
            conf_thres=self.confidence, nms_thres=self.nms_iou)[0]
        path = Path(map_out_path) / "detection-results" / f"{image_id}.txt"
        with path.open("w") as stream:
            if results is not None:
                scores = results[:, 4] * results[:, 5]
                for index in np.argsort(scores)[::-1][:self.max_boxes]:
                    top, left, bottom, right = results[index, :4]
                    name = class_names[int(results[index, 6])]
                    stream.write(f"{name} {scores[index]} {left} {top} {right} {bottom}\n")
        return outputs[3]

    def on_epoch_end(self, epoch, model_eval):
        if not self.eval_flag or epoch % self.period:
            return
        from utils.utils_map import get_map
        self.net = model_eval
        was_training = self.net.training
        self.net.eval()
        for folder in ("ground-truth", "detection-results"):
            (self.map_out_path / folder).mkdir(parents=True, exist_ok=True)
        try:
            height, width = self.input_shape
            for index in range(len(self.dataset)):
                image, boxes, _ = self.dataset[index]
                image_id = f"{index:08d}"
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)
                path = self.map_out_path / "ground-truth" / f"{image_id}.txt"
                with path.open("w") as stream:
                    for x, y, w, h, category in boxes:
                        left, right = (x - w / 2) * width, (x + w / 2) * width
                        top, bottom = (y - h / 2) * height, (y + h / 2) * height
                        stream.write(f"{self.class_names[int(category)]} {left} {top} {right} {bottom}\n")
            if self.metric == "coco":
                from utils.utils_map import get_coco_map
                score = get_coco_map(self.class_names, str(self.map_out_path))[0]
            else:
                score = get_map(self.MINOVERLAP, False, path=str(self.map_out_path))
            self.maps.append(score)
            self.epoches.append(epoch)
            with (self.log_dir / "epoch_map.txt").open("a") as stream:
                stream.write(f"{epoch},{score}\n")
        finally:
            self.net.train(was_training)
            shutil.rmtree(self.map_out_path)
