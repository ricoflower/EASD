"""EASD network and complete spike-camera detection API."""

from easd.inference import EASDDetector
from easd.model import EASD, EASDOutput

__all__ = ["EASD", "EASDOutput", "EASDDetector"]
