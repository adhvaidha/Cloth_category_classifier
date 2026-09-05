"""
filters

Phase 2 of the Smart Wardrobe / AI Outfit Planner pipeline:

    Phase 1 garment JSON -> occasion + weather rules -> filtered garments

This package answers "is this garment appropriate for this
situation?" using explicit, documented application-level rules. It
does NOT claim DeepFashion itself understands occasion or weather,
does NOT build outfits, and does NOT use any embedding/compatibility
model. Those belong to later phases.
"""

from filters.filter_engine import (
    filter_wardrobe,
    evaluate_garment,
    is_category_confident,
    run_all_test_scenarios,
    CONFIDENCE_THRESHOLD,
    MARGIN_THRESHOLD,
)
from filters.occasion import SUPPORTED_OCCASIONS
from filters.weather import SUPPORTED_TEMP_BUCKETS, temp_to_bucket

__all__ = [
    "filter_wardrobe",
    "evaluate_garment",
    "is_category_confident",
    "run_all_test_scenarios",
    "CONFIDENCE_THRESHOLD",
    "MARGIN_THRESHOLD",
    "SUPPORTED_OCCASIONS",
    "SUPPORTED_TEMP_BUCKETS",
    "temp_to_bucket",
]
