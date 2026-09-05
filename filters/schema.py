"""
schema.py (filters)

Small, dependency-free dataclasses describing Phase 2's output shape.

{
  "request": {"occasion": "office", "temperature_c": 30, "rain": false},
  "garments": [
    {
      "image": "wardrobe/top1.jfif",
      "decision": "suitable" | "preferred" | "filtered" | "unclassified",
      "reason": "<human-readable summary>",
      "details": {
        "category": {"name": "...", "probability": 0.0, "confident": true},
        "occasion_verdicts": [{"verdict": "...", "reason": "..."}],
        "weather_verdicts": [{"verdict": "...", "reason": "..."}]
      }
    },
    ...
  ]
}
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Verdict:
    """One rule's opinion about a garment.

    verdict is one of: 'hard_reject', 'soft_penalty', 'preferred', 'neutral'
    """
    verdict: str
    reason: str

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "reason": self.reason}


@dataclass
class GarmentDecision:
    image: str
    decision: str  # 'suitable' | 'preferred' | 'filtered' | 'unclassified'
    reason: str
    category_name: Optional[str]
    category_probability: Optional[float]
    category_confident: bool
    occasion_verdicts: List[Verdict] = field(default_factory=list)
    weather_verdicts: List[Verdict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image": self.image,
            "decision": self.decision,
            "reason": self.reason,
            "details": {
                "category": {
                    "name": self.category_name,
                    "probability": self.category_probability,
                    "confident": self.category_confident,
                },
                "occasion_verdicts": [v.to_dict() for v in self.occasion_verdicts],
                "weather_verdicts": [v.to_dict() for v in self.weather_verdicts],
            },
        }


@dataclass
class FilterResult:
    occasion: str
    temperature_c: float
    rain: bool
    garments: List[GarmentDecision]

    def to_dict(self) -> dict:
        return {
            "request": {
                "occasion": self.occasion,
                "temperature_c": self.temperature_c,
                "rain": self.rain,
            },
            "garments": [g.to_dict() for g in self.garments],
        }
