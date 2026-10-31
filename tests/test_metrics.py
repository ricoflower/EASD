"""Evaluation fixtures independent of any model or private benchmark data."""

import importlib.util
import unittest

import numpy as np

from easd.metrics import coco_ap, voc_ap50


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.actual = [np.array([[10, 20, 40, 60, 0]], dtype=np.float32)]
        self.perfect = [np.array([[10, 20, 40, 60, 0.9, 0]], dtype=np.float32)]

    def test_voc_perfect_and_missed_detection(self):
        self.assertEqual(voc_ap50(self.perfect, self.actual, ["object"])["value"], 1.0)
        empty = [np.empty((0, 6), dtype=np.float32)]
        self.assertEqual(voc_ap50(empty, self.actual, ["object"])["value"], 0.0)

    def test_voc_false_positive_before_true_positive(self):
        predicted = [
            np.array([[80, 80, 100, 100, 0.99, 0], [10, 20, 40, 60, 0.9, 0]], dtype=np.float32)
        ]
        self.assertAlmostEqual(voc_ap50(predicted, self.actual, ["object"])["value"], 0.5)

    def test_voc_duplicate_does_not_match_another_object(self):
        actual = [np.concatenate((self.actual[0], [[70, 80, 90, 100, 0]]))]
        predicted = [np.concatenate((self.perfect[0], self.perfect[0]))]
        self.assertAlmostEqual(voc_ap50(predicted, actual, ["object"])["value"], 0.5)

    def test_voc_wrong_class(self):
        predicted = [self.perfect[0].copy()]
        predicted[0][:, 5] = 1
        self.assertEqual(voc_ap50(predicted, self.actual, ["object", "other"])["value"], 0.0)

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "optional COCO dependency")
    def test_coco_perfect_and_no_detections(self):
        result = coco_ap(self.perfect, self.actual, ["object"], [(250, 400)])
        self.assertAlmostEqual(result["value"], 1.0)
        result = coco_ap([np.empty((0, 6))], self.actual, ["object"], [(250, 400)])
        self.assertEqual(result["value"], 0.0)


if __name__ == "__main__":
    unittest.main()
