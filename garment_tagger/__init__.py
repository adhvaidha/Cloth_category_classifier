"""
garment_tagger

Phase 1 of the Smart Wardrobe / AI Outfit Planner pipeline:

    Wardrobe image -> DeepFashion garment tagging -> structured JSON

This package wraps the existing, verified DeepFashion classifier
(models.model.FashionResnet + results/pretrained_model/epoch_340.pth)
so it can be called programmatically and produce structured data,
instead of only the matplotlib PNG that test_on_image.py produces.

Later phases (occasion/weather filtering, candidate outfit generation,
compatibility scoring, ranking, API, etc.) are NOT part of this
package yet.
"""

from garment_tagger.deepfashion_infer import (
    run_inference,
    build_model,
    load_labels,
    load_image,
    save_json,
    DEFAULT_CHECKPOINT,
    DEFAULT_RESNET_TYPE,
)

__all__ = [
    "run_inference",
    "build_model",
    "load_labels",
    "load_image",
    "save_json",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_RESNET_TYPE",
]
