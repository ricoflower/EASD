"""Regression tests for the runtime failures repaired during the EASD cleanup."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from easd.spatial import EntropyWindowMixer, EntropyMask
from easd import EASD, EASDDetector
from easd.loss import DetectionLoss
from easd.checkpoints import load_easd_weights, load_decoder_weights, migrate_legacy_state_dict
from easd.data import SpikeDetectionDataset, collate_spikes
from easd.spikes import boxes_to_padded, clip_boxes, pad_spikes, raw_to_spike
from easd.config import load_anchors
from easd.postprocess import DetectionPostProcessor

MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]


class FixedMask(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x):
        return self.mask[: x.shape[0]]


class EASDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(11)
        cls.anchors = load_anchors()

    def test_symmetric_padding_and_label_coordinates(self):
        raw = torch.arange(41 * 250 * 400).reshape(41, 250, 400)
        padded = pad_spikes(raw)
        np.testing.assert_array_equal(
            padded.numpy(), np.pad(raw.numpy(), ((0, 0), (3, 3), (0, 0)), "symmetric")
        )
        torch.testing.assert_close(pad_spikes(padded), padded)
        boxes = boxes_to_padded([[0.5, 0.4, 0.2, 0.2, 0]], (250, 400), (256, 400))
        self.assertAlmostEqual(boxes[0, 1] * 256 - 3, 100, places=4)
        self.assertAlmostEqual(boxes[0, 3] * 256, 50, places=4)

    def test_visible_box_clipping(self):
        boxes = clip_boxes([[0.98, 0.5, 0.2, 0.4, 2], [1.1, 0.5, 0.05, 0.1, 0]], 3)
        self.assertEqual(boxes.shape, (1, 5))
        self.assertAlmostEqual(boxes[0, 0], 0.94, places=5)
        self.assertAlmostEqual(boxes[0, 2], 0.12, places=5)

    def test_bit_packed_dat_decode(self):
        spikes = np.random.default_rng(5).integers(0, 2, (2, 250, 400), dtype=np.uint8)
        encoded = np.packbits(spikes[:, ::-1].reshape(-1), bitorder="little").tobytes()
        np.testing.assert_array_equal(raw_to_spike(encoded), spikes)
        with self.assertRaises(ValueError):
            raw_to_spike(encoded[:-1])

    def test_dataset_empty_labels_and_effective_strides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spike sample.npy"
            np.save(path, np.zeros((41, 250, 400), dtype=np.uint8))
            dataset = SpikeDetectionDataset(
                "dsec", [f"{path} 0.5,0.4,0.1,0.2,0", str(path)], (250, 400), 3, self.anchors, MASK
            )
            images, boxes, targets = collate_spikes([dataset[0], dataset[1]])
            self.assertEqual(tuple(images.shape), (2, 41, 256, 400))
            self.assertEqual(boxes[1].shape, (0, 5))
            self.assertEqual(
                [tuple(t.shape[2:4]) for t in targets], [(16, 25), (32, 50), (64, 100)]
            )
            self.assertGreater(sum(t[0, ..., 4].sum().item() for t in targets), 0)
            self.assertEqual(sum(t[1, ..., 4].sum().item() for t in targets), 0)

    def test_entropy_is_independent_per_sample(self):
        mask = EntropyMask()
        x = torch.randn(2, 1, 32, 48) * torch.tensor([0.1, 8.0])[:, None, None, None]
        torch.testing.assert_close(mask(x), torch.cat([mask(sample[None]) for sample in x]))
        self.assertFalse(mask(torch.zeros(1, 1, 32, 48)).any())

    def test_routing_empty_full_and_different_batch_counts(self):
        for foreground in (
            torch.zeros(2, 24, dtype=torch.bool),
            torch.ones(2, 24, dtype=torch.bool),
            torch.stack((torch.arange(24) < 3, torch.arange(24) < 17)),
        ):
            mixer = EntropyWindowMixer(12).eval()
            mixer.predictor.entropy_mask = FixedMask(foreground)
            x = torch.randn(2, 12, 32, 48)
            batch = mixer(x)
            separate = []
            for index in range(2):
                mixer.predictor.entropy_mask = FixedMask(foreground[index : index + 1])
                separate.append(mixer(x[index : index + 1]))
            torch.testing.assert_close(batch, torch.cat(separate), rtol=2e-5, atol=2e-5)
            batch.square().mean().backward()
            self.assertTrue(torch.isfinite(mixer.project_v.weight.grad).all())
            if foreground.any():
                self.assertGreater(mixer.project_q.weight.grad.abs().sum().item(), 0)

    def test_train_mode_and_gradients_reach_both_branches(self):
        model = EASD(num_classes=3).train()
        dataset = SpikeDetectionDataset("dsec", [], (32, 48), 3, self.anchors, MASK)
        boxes = np.array([[0.5, 0.5, 0.2, 0.3, 0]], dtype=np.float32)
        targets = [torch.from_numpy(t[None]) for t in dataset.build_targets(boxes)]
        outputs = model(torch.randint(0, 2, (1, 41, 32, 48)).float())
        criterion = DetectionLoss(self.anchors, 3, (32, 48), MASK)
        loss = sum(criterion(i, output, targets[i]) for i, output in enumerate(outputs.head_logits))
        loss.backward()
        self.assertTrue(all(module.training for module in model.modules()))
        for parameter in (
            model.temporal_encoder.branches[0].input_conv.weight,
            model.spatial_encoder.intensity_embedding.weight,
            model.decoder.stem.conv.conv.weight,
            model.head_fine.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_full_resolution_and_checkpoint_round_trip(self):
        model = EASD(num_classes=3).eval()
        self.assertEqual(len(model.state_dict()), 283)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            torch.save(model.state_dict(), path)
            restored = EASD(num_classes=3).eval()
            self.assertFalse(load_easd_weights(restored, path).migrated)
            with torch.no_grad():
                outputs = restored(torch.zeros(41, 250, 400))
        self.assertEqual(
            [tuple(t.shape) for t in (*outputs.head_logits, outputs.representation)],
            [(1, 24, 16, 25), (1, 24, 32, 50), (1, 24, 64, 100), (1, 3, 256, 400)],
        )

    def test_prefixed_pretrained_backbone_loading(self):
        source = EASD(num_classes=3)
        target = EASD(num_classes=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yolo.pth"
            torch.save(
                {"backbone." + key: value for key, value in source.decoder.state_dict().items()},
                path,
            )
            load_decoder_weights(target.decoder, path)
        for key, value in source.decoder.state_dict().items():
            torch.testing.assert_close(target.decoder.state_dict()[key], value)

    def test_predictions_remove_padding_in_sensor_coordinates(self):
        processor = DetectionPostProcessor(self.anchors, 3)
        # Two overlapping candidates: same-class NMS retains the higher score.
        decoded = torch.tensor(
            [
                [
                    [40 / 400, 33 / 256, 40 / 400, 40 / 256, 0.8, 0.1, 0.9, 0.1],
                    [40 / 400, 33 / 256, 40 / 400, 40 / 256, 0.7, 0.1, 0.9, 0.1],
                ]
            ]
        )
        prediction = processor.select(decoded, (250, 400))[0]
        self.assertEqual(prediction.shape, (1, 6))
        np.testing.assert_allclose(prediction[0], [20, 10, 60, 50, 0.72, 1], atol=1e-5)

    def test_checkpoint_migration_and_unknown_keys(self):
        fixture = {
            "extractor.conv0_0.weight": torch.ones(1),
            "sp_attn.group.body.0.mixer.route.conv.0.weight": torch.ones(1),
            "yolo_head_P3.bias": torch.ones(1),
            "alpha": torch.ones(1),
            "sp_attn.group.body.0.mixer.route.out_mask.0.bias": torch.ones(1),
        }
        migrated, legacy, removed = migrate_legacy_state_dict(fixture)
        self.assertTrue(legacy)
        self.assertEqual(
            set(migrated),
            {
                "temporal_encoder.branches.0.input_conv.weight",
                "spatial_encoder.attention_group.blocks.0.mixer.predictor.conv.0.weight",
                "head_fine.bias",
            },
        )
        self.assertEqual(len(removed), 2)
        with tempfile.TemporaryDirectory() as directory:
            model = EASD()
            path = Path(directory) / "invalid.pth"
            state = model.state_dict()
            state["unexpected.weight"] = torch.ones(1)
            torch.save(state, path)
            with self.assertRaises(RuntimeError):
                load_easd_weights(model, path)

    def test_detector_native_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            model = EASD().eval()
            torch.save(model.state_dict(), path)
            detector = EASDDetector(weights=path, device="cpu", confidence=0.99)
            prediction = detector.predict(torch.zeros(41, 32, 48))
            self.assertEqual(len(prediction), 1)
            self.assertEqual(prediction[0].shape, (0, 6))


if __name__ == "__main__":
    unittest.main()
