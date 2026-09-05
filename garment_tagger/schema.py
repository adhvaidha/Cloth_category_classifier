"""
schema.py

Plain-dict schema helpers for the Phase 1 garment tagger output.

We deliberately keep this as small dataclasses -> dict converters
(rather than pydantic/etc.) so Phase 1 has zero new dependencies
beyond what the existing Cloth_category_classifier repo already uses.

Shape of the JSON produced by `deepfashion_infer.run_inference()`:

{
  "image": "<absolute path to the image that was processed>",
  "checkpoint": "<absolute path to the .pth checkpoint used>",
  "resnet_type": "resnet18",
  "category": [
    {"name": "Blouse", "probability": 0.42},
    ...
  ],
  "attributes": {
    "texture": [{"name": "floral", "probability": 0.81}, ...],
    "fabric":  [...],
    "shape":   [...],
    "part":    [...],
    "style":   [...]
  },
  "bounding_box": {
    "normalized": [x1, y1, x2, y2],   # raw model output, each in [0, 1]
    "pixels_224": [x1, y1, x2, y2],   # same box scaled to the 224x224
                                       # center-crop, i.e. what
                                       # test_on_image.py draws
    "crop_size": 224
  }
}

All probabilities are the actual sigmoid (attributes) / softmax
(category) outputs of the DeepFashion model for the given image -
nothing here is hand-entered or copied from a PNG.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class NamedProbability:
    """A single (label, probability) prediction."""
    name: str
    probability: float

    def to_dict(self) -> dict:
        return {"name": self.name, "probability": self.probability}


@dataclass
class BoundingBox:
    """Predicted bounding box, in both normalized and pixel form."""
    normalized: List[float]   # [x1, y1, x2, y2] in [0, 1]
    pixels_224: List[float]   # same box scaled to the 224x224 crop
    crop_size: int = 224

    def to_dict(self) -> dict:
        return {
            "normalized": self.normalized,
            "pixels_224": self.pixels_224,
            "crop_size": self.crop_size,
        }


@dataclass
class GarmentTagResult:
    """Full structured output for one image."""
    image: str
    checkpoint: str
    resnet_type: str
    category: List[NamedProbability]
    attributes: Dict[str, List[NamedProbability]]
    bounding_box: BoundingBox

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "checkpoint": self.checkpoint,
            "resnet_type": self.resnet_type,
            "category": [c.to_dict() for c in self.category],
            "attributes": {
                attr_type: [p.to_dict() for p in preds]
                for attr_type, preds in self.attributes.items()
            },
            "bounding_box": self.bounding_box.to_dict(),
        }
