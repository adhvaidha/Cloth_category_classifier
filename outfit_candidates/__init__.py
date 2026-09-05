"""
outfit_candidates

Phase 3 of the Smart Wardrobe / AI Outfit Planner pipeline:

    Phase 2 filtered garments -> structurally valid candidate outfits

This package only enumerates which garment combinations are
structurally possible to wear together (using DeepFashion's own
upper/lower/full-body category_type). It does NOT judge whether they
look good together (no embedding model, no compatibility model, no
ranking) - that's later phases.
"""

from outfit_candidates.generator import (
    generate_candidates,
    run_on_directory,
    load_category_slot_map,
    get_survivors,
)

__all__ = [
    "generate_candidates",
    "run_on_directory",
    "load_category_slot_map",
    "get_survivors",
]
