"""
schema.py (outfit_candidates)

Small, dependency-free dataclasses for Phase 3's output shape.

{
  "request": {"occasion": "office", "temperature_c": 30, "rain": false},
  "garments_entering_phase3": [
    {"image": "top1.jfif", "slot": "upper", "phase2_decision": "preferred"}
  ],
  "excluded_no_structural_slot": [
    {"image": "shoe1.jfif", "reason": "..."}
  ],
  "candidate_outfits": [
    {"items": ["top1.jfif", "bottom1.jfif"], "structure": "upper_lower"}
  ],
  "rejected_structurally_invalid": [
    {"items": ["top1.jfif", "top2.jfif"], "reason": "..."}
  ]
}
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SurvivingGarment:
    image: str
    slot: Optional[str]          # 'upper' | 'lower' | 'full_body' | None
    phase2_decision: str         # 'suitable' | 'preferred' | 'unclassified'
    category_name: Optional[str]

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "slot": self.slot,
            "phase2_decision": self.phase2_decision,
            "category_name": self.category_name,
        }


@dataclass
class ExcludedGarment:
    image: str
    reason: str

    def to_dict(self) -> dict:
        return {"image": self.image, "reason": self.reason}


@dataclass
class CandidateOutfit:
    items: List[str]
    structure: str  # 'upper_lower' | 'full_body' | 'full_body_layered'

    def to_dict(self) -> dict:
        return {"items": self.items, "structure": self.structure}


@dataclass
class RejectedPair:
    items: List[str]
    reason: str

    def to_dict(self) -> dict:
        return {"items": self.items, "reason": self.reason}


@dataclass
class Phase3Result:
    occasion: str
    temperature_c: float
    rain: bool
    garments_entering_phase3: List[SurvivingGarment] = field(default_factory=list)
    excluded_no_structural_slot: List[ExcludedGarment] = field(default_factory=list)
    candidate_outfits: List[CandidateOutfit] = field(default_factory=list)
    rejected_structurally_invalid: List[RejectedPair] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "request": {
                "occasion": self.occasion,
                "temperature_c": self.temperature_c,
                "rain": self.rain,
            },
            "garments_entering_phase3": [g.to_dict() for g in self.garments_entering_phase3],
            "excluded_no_structural_slot": [g.to_dict() for g in self.excluded_no_structural_slot],
            "candidate_outfits": [c.to_dict() for c in self.candidate_outfits],
            "rejected_structurally_invalid": [r.to_dict() for r in self.rejected_structurally_invalid],
        }
