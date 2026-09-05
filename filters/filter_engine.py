# ---------------------------------------------------------------
# filter_engine.py
#
# PHASE 2:
# WARDROBE -> OCCASION + WEATHER -> FILTERED GARMENTS
#
# Attribute-first version.
#
# Phase 1 provides:
#   - category
#   - category predictions
#   - category probabilities
#   - CLIP attributes
#   - category-specific attributes
#
# Phase 2 uses those attributes to determine suitability.
#
# NO filename-based classification.
# NO DeepFashion.
# NO outfit generation.
# NO outfit ranking.
# ---------------------------------------------------------------

import argparse
import json
import os
import sys
from pathlib import Path


# --------------------------------------------------------------------
# Project imports
# --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from filters import occasion as occasion_rules
from filters import weather as weather_rules
from filters.schema import Verdict, GarmentDecision, FilterResult


# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

# These are now ONLY used for diagnostics / reliability information.
#
# They are NOT hard filters.
#
# A garment with an uncertain category can still be evaluated from
# its visual attributes.
# --------------------------------------------------------------------

CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "CLIP_CATEGORY_MIN_CONFIDENCE",
        "0.50",
    )
)

MARGIN_THRESHOLD = float(
    os.getenv(
        "CLIP_CATEGORY_MIN_MARGIN",
        "0.15",
    )
)


ATTRIBUTE_CONFIDENCE = float(
    os.getenv(
        "CLIP_ATTRIBUTE_CONFIDENCE",
        "0.40",
    )
)


FABRIC_CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "CLIP_FABRIC_CONFIDENCE",
        "0.40",
    )
)


# --------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------

def load_garment(json_path):
    """
    Load one Phase 1 garment JSON file.
    """

    with open(
        json_path,
        encoding="utf-8",
    ) as f:
        return json.load(f)


# --------------------------------------------------------------------
# Category helpers
# --------------------------------------------------------------------

def _normalize_category_predictions(category_list):
    """
    Normalize CLIP category predictions.

    Accepted:

        [
            {
                "name": "Dress",
                "probability": 0.8
            }
        ]

    or:

        [
            {
                "category": "Dress",
                "probability": 0.8
            }
        ]

    or:

        {
            "category": "Dress",
            "probability": 0.8
        }

    or:

        "Dress"

    No filename is inspected.
    """

    if category_list is None:
        return []

    if isinstance(
        category_list,
        dict,
    ):
        category_list = [
            category_list
        ]

    if isinstance(
        category_list,
        str,
    ):
        return [
            {
                "name": category_list,
                "probability": 1.0,
            }
        ]

    if not isinstance(
        category_list,
        list,
    ):
        return []

    normalized = []

    for prediction in category_list:

        if isinstance(
            prediction,
            str,
        ):
            normalized.append(
                {
                    "name": prediction,
                    "probability": 0.0,
                }
            )
            continue

        if not isinstance(
            prediction,
            dict,
        ):
            continue

        name = (
            prediction.get("name")
            or prediction.get("category_name")
            or prediction.get("category")
        )

        if not name:
            continue

        try:
            probability = float(
                prediction.get(
                    "probability",
                    prediction.get(
                        "confidence",
                        0.0,
                    ),
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            probability = 0.0

        item = dict(prediction)

        item["name"] = str(name)
        item["probability"] = probability

        normalized.append(item)

    normalized.sort(
        key=lambda item: float(
            item.get(
                "probability",
                0.0,
            )
        ),
        reverse=True,
    )

    return normalized


def category_confidence_info(
    category_list,
):
    """
    Return category confidence information.

    IMPORTANT:
    This is diagnostic information only.

    It does NOT filter garments.
    """

    predictions = _normalize_category_predictions(
        category_list
    )

    if not predictions:
        return {
            "confident": False,
            "top1": 0.0,
            "top2": 0.0,
            "margin": 0.0,
        }

    try:
        top1 = float(
            predictions[0].get(
                "probability",
                0.0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        top1 = 0.0

    if len(predictions) > 1:

        try:
            top2 = float(
                predictions[1].get(
                    "probability",
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            top2 = 0.0

    else:
        top2 = 0.0

    margin = top1 - top2

    return {
        "confident": (
            top1 >= CONFIDENCE_THRESHOLD
            and margin >= MARGIN_THRESHOLD
        ),
        "top1": top1,
        "top2": top2,
        "margin": margin,
    }
def is_category_confident(category_list):
    """
    Backwards-compatible helper.

    Returns True when the top CLIP category satisfies the
    configured confidence and top-1/top-2 margin thresholds.

    IMPORTANT:
    This is only a diagnostic/helper function now.
    It is NOT used as a hard filter by evaluate_garment().
    """

    info = category_confidence_info(category_list)

    return bool(
        info.get("confident", False)
    )


# --------------------------------------------------------------------
# Attribute helpers
# --------------------------------------------------------------------

def _top_attr_name(
    attributes,
    attr_type,
    min_probability=0.0,
):
    """
    Return the highest-probability attribute name.
    """

    if not isinstance(
        attributes,
        dict,
    ):
        return None

    predictions = (
        attributes.get(attr_type)
        or []
    )

    if not isinstance(
        predictions,
        list,
    ):
        return None

    best_name = None
    best_probability = 0.0

    for prediction in predictions:

        if not isinstance(
            prediction,
            dict,
        ):
            continue

        name = (
            prediction.get("name")
            or prediction.get("attribute")
            or prediction.get("label")
        )

        if not name:
            continue

        try:
            probability = float(
                prediction.get(
                    "probability",
                    prediction.get(
                        "confidence",
                        0.0,
                    ),
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if probability > best_probability:
            best_name = str(name)
            best_probability = probability

    if (
        best_name is None
        or best_probability < min_probability
    ):
        return None

    return best_name


def _top_attribute(
    attributes,
    attr_type,
):
    """
    Return:

        (name, probability)

    for the strongest prediction.
    """

    if not isinstance(
        attributes,
        dict,
    ):
        return None, 0.0

    predictions = (
        attributes.get(attr_type)
        or []
    )

    if not isinstance(
        predictions,
        list,
    ):
        return None, 0.0

    best_name = None
    best_probability = 0.0

    for prediction in predictions:

        if not isinstance(
            prediction,
            dict,
        ):
            continue

        name = (
            prediction.get("name")
            or prediction.get("attribute")
            or prediction.get("label")
        )

        if not name:
            continue

        try:
            probability = float(
                prediction.get(
                    "probability",
                    prediction.get(
                        "confidence",
                        0.0,
                    ),
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if probability > best_probability:
            best_name = str(name)
            best_probability = probability

    return (
        best_name,
        best_probability,
    )


def _attribute_probability(
    attributes,
    attr_type,
    names,
):
    """
    Find the strongest probability for any requested
    attribute name.
    """

    if not isinstance(
        attributes,
        dict,
    ):
        return 0.0

    predictions = (
        attributes.get(attr_type)
        or []
    )

    if not isinstance(
        predictions,
        list,
    ):
        return 0.0

    wanted = {
        str(name).strip().lower()
        for name in names
    }

    best = 0.0

    for prediction in predictions:

        if not isinstance(
            prediction,
            dict,
        ):
            continue

        name = (
            prediction.get("name")
            or prediction.get("attribute")
            or prediction.get("label")
        )

        if not name:
            continue

        if (
            str(name).strip().lower()
            not in wanted
        ):
            continue

        try:
            probability = float(
                prediction.get(
                    "probability",
                    prediction.get(
                        "confidence",
                        0.0,
                    ),
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        best = max(
            best,
            probability,
        )

    return best


# --------------------------------------------------------------------
# Category-specific attribute lookup
# --------------------------------------------------------------------

def _get_specific_attribute(
    attributes,
    category,
):
    """
    Find the relevant category-specific attribute group.
    """

    if not isinstance(
        attributes,
        dict,
    ):
        return (
            None,
            None,
            0.0,
        )

    category = (
        str(category or "")
        .strip()
        .lower()
    )

    groups = []

    if category in {
        "dress",
    }:
        groups = [
            "dress_type",
        ]

    elif category in {
        "shirt",
        "blouse",
        "tee",
        "tank",
        "hoodie",
        "sweater",
        "cardigan",
        "shrug",
    }:
        groups = [
            "top_style",
        ]

    elif category in {
        "jeans",
        "pants",
        "shorts",
        "skirt",
    }:
        groups = [
            "bottom_style",
        ]

    elif category in {
        "jacket",
        "blazer",
        "coat",
    }:
        groups = [
            "outerwear_style",
        ]

    elif category in {
        "shoes",
        "shoe",
        "footwear",
    }:
        groups = [
            "shoe_style",
        ]

    else:
        groups = [
            "dress_type",
            "top_style",
            "bottom_style",
            "outerwear_style",
            "shoe_style",
        ]

    for group in groups:

        name, probability = _top_attribute(
            attributes,
            group,
        )

        if name:
            return (
                group,
                name,
                probability,
            )

    return (
        None,
        None,
        0.0,
    )


# --------------------------------------------------------------------
# Occasion attribute evaluation
# --------------------------------------------------------------------

def _occasion_attribute_verdicts(
    occasion,
    category_name,
    attributes,
):
    """
    Determine occasion suitability from CLIP visual attributes.

    Attribute-first:
        category
        + formality
        + style
        + pattern
        + category-specific attributes

    Filename is NEVER used.

    Important design:
    - Explicit positive evidence can make an item preferred.
    - Explicit negative evidence can hard-reject an item.
    - For restrictive occasions such as OFFICE, ambiguity is NOT
      automatically treated as suitable.
    """

    verdicts = []

    category = (
        str(category_name or "")
        .strip()
        .lower()
    )

    if not isinstance(attributes, dict):
        attributes = {}

    # ---------------------------------------------------------------
    # General attributes
    # ---------------------------------------------------------------

    formality_name, formality_prob = _top_attribute(
        attributes,
        "formality",
    )

    style_name, style_prob = _top_attribute(
        attributes,
        "style",
    )

    pattern_name, pattern_prob = _top_attribute(
        attributes,
        "pattern",
    )

    season_name, season_prob = _top_attribute(
        attributes,
        "season",
    )

    formality = (
        formality_name or ""
    ).strip().lower()

    style = (
        style_name or ""
    ).strip().lower()

    pattern = (
        pattern_name or ""
    ).strip().lower()

    season = (
        season_name or ""
    ).strip().lower()

    # ---------------------------------------------------------------
    # Category-specific attribute
    # ---------------------------------------------------------------

    specific_group, specific_name, specific_prob = (
        _get_specific_attribute(
            attributes,
            category,
        )
    )

    specific = (
        specific_name or ""
    ).strip().lower()

    # ---------------------------------------------------------------
    # GENERAL FORMALITY
    # ---------------------------------------------------------------

    formal_strength = max(
        _attribute_probability(
            attributes,
            "formality",
            {
                "formal",
                "very formal",
                "business casual",
                "smart casual",
            },
        ),
        _attribute_probability(
            attributes,
            "style",
            {
                "professional",
                "elegant",
                "classic",
                "business",
                "minimal",
            },
        ),
    )

    casual_strength = max(
        _attribute_probability(
            attributes,
            "formality",
            {
                "casual",
                "very casual",
            },
        ),
        _attribute_probability(
            attributes,
            "style",
            {
                "casual",
                "bohemian",
                "boho",
                "sporty",
                "streetwear",
                "relaxed",
            },
        ),
    )

    # ---------------------------------------------------------------
    # OFFICE-SPECIFIC POSITIVE EVIDENCE
    # ---------------------------------------------------------------

    office_specific = max(
        _attribute_probability(
            attributes,
            "dress_type",
            {
                "office dress",
                "business dress",
                "work dress",
                "formal dress",
            },
        ),
        _attribute_probability(
            attributes,
            "top_style",
            {
                "office blouse",
                "office shirt",
                "formal shirt",
                "business shirt",
            },
        ),
        _attribute_probability(
            attributes,
            "outerwear_style",
            {
                "office blazer",
                "formal blazer",
                "business jacket",
            },
        ),
        _attribute_probability(
            attributes,
            "bottom_style",
            {
                "formal trousers",
                "formal pants",
                "formal skirt",
                "pencil skirt",
            },
        ),
    )

    # ---------------------------------------------------------------
    # PARTY / EVENING
    # ---------------------------------------------------------------

    party_strength = max(
        _attribute_probability(
            attributes,
            "dress_type",
            {
                "party dress",
                "cocktail dress",
                "evening dress",
                "club dress",
                "party wear",
            },
        ),
        _attribute_probability(
            attributes,
            "style",
            {
                "festive",
                "glamorous",
                "party",
            },
        ),
    )

    # ---------------------------------------------------------------
    # CASUAL FULL-BODY CLOTHING
    # ---------------------------------------------------------------

    casual_dress_strength = max(
        _attribute_probability(
            attributes,
            "dress_type",
            {
                "sundress",
                "summer dress",
                "casual dress",
                "casual day dress",
                "beach dress",
                "bohemian dress",
            },
        ),
        _attribute_probability(
            attributes,
            "style",
            {
                "casual",
                "bohemian",
                "boho",
                "beach",
                "relaxed",
            },
        ),
    )

    # ---------------------------------------------------------------
    # OFFICE
    # ---------------------------------------------------------------

    if occasion == "office":

        # ===========================================================
        # ATTRIBUTE-DRIVEN OFFICE SUITABILITY
        # ===========================================================
        #
        # The category name is NOT used as a hardcoded rejection rule.
        #
        # Instead, office suitability is derived from the visual
        # attributes predicted by Phase 1:
        #
        #   formality
        #   style
        #   category-specific style
        #
        # Strongly casual visual evidence is a genuine incompatibility
        # with formal/office wear. Merely lacking positive office
        # evidence is NOT an incompatibility.
        #
        # This allows the same logic to generalize to garments that
        # were never explicitly enumerated in this file.
        # ===========================================================

        # -----------------------------------------------------------
        # Attribute groups representing strong casual evidence.
        # -----------------------------------------------------------

        strong_casual_strength = max(
            casual_strength,

            _attribute_probability(
                attributes,
                "style",
                {
                    "athletic",
                    "athleisure",
                    "sporty",
                    "streetwear",
                    "gym",
                    "workout",
                    "beach",
                    "beachwear",
                    "lounge",
                    "sleepwear",
                },
            ),

            _attribute_probability(
                attributes,
                "top_style",
                {
                    "graphic tee",
                    "athletic top",
                    "gym top",
                    "sporty top",
                    "crop top",
                    "casual tee",
                },
            ),

            _attribute_probability(
                attributes,
                "bottom_style",
                {
                    "athletic shorts",
                    "sport shorts",
                    "gym shorts",
                    "beach shorts",
                    "casual shorts",
                    "sweatpants",
                    "track pants",
                    "athletic pants",
                    "joggers",
                    "lounge pants",
                },
            ),

            _attribute_probability(
                attributes,
                "dress_type",
                {
                    "sundress",
                    "summer dress",
                    "casual dress",
                    "casual day dress",
                    "beach dress",
                    "bohemian dress",
                },
            ),
        )

        # -----------------------------------------------------------
        # Strong professional evidence.
        # -----------------------------------------------------------

        strong_professional_strength = max(
            formal_strength,
            office_specific,

            _attribute_probability(
                attributes,
                "style",
                {
                    "professional",
                    "business",
                    "business casual",
                    "elegant",
                    "tailored",
                    "polished",
                    "sophisticated",
                    "classic",
                    "minimal",
                },
            ),

            _attribute_probability(
                attributes,
                "formality",
                {
                    "formal",
                    "very formal",
                    "business casual",
                    "smart casual",
                },
            ),

            _attribute_probability(
                attributes,
                "top_style",
                {
                    "office blouse",
                    "office shirt",
                    "formal shirt",
                    "business shirt",
                    "dress shirt",
                    "tailored shirt",
                },
            ),

            _attribute_probability(
                attributes,
                "bottom_style",
                {
                    "formal trousers",
                    "formal pants",
                    "dress pants",
                    "tailored trousers",
                    "tailored pants",
                    "formal skirt",
                    "pencil skirt",
                },
            ),

            _attribute_probability(
                attributes,
                "outerwear_style",
                {
                    "office blazer",
                    "formal blazer",
                    "business jacket",
                    "tailored blazer",
                },
            ),
        )

        # ===========================================================
        # 1. Explicitly strong party/evening evidence
        # ===========================================================

        if party_strength >= ATTRIBUTE_CONFIDENCE:

            verdicts.append(
                (
                    "hard_reject",
                    (
                        "Visual attributes strongly identify this "
                        "garment as party/evening wear, which is "
                        "incompatible with office wear."
                    ),
                )
            )

            return verdicts

        # ===========================================================
        # 2. Strong casual evidence
        # ===========================================================
        #
        # This is the important distinction:
        #
        # We do NOT say:
        #
        #     category == Shorts -> reject
        #
        # We say:
        #
        #     strong casual visual evidence -> reject for office
        #
        # Therefore the rule generalizes to shorts, gymwear,
        # beachwear, strongly athletic garments, etc., according to
        # what the attribute model actually sees.
        # ===========================================================

        if (
            strong_casual_strength >= 0.80
            and strong_professional_strength < 0.60
        ):

            verdicts.append(
                (
                    "hard_reject",
                    (
                        "Visual attributes indicate a strongly "
                        "casual/athletic style that is incompatible "
                        "with formal office wear."
                    ),
                )
            )

            return verdicts

        # ===========================================================
        # 3. Strong professional / office evidence
        # ===========================================================

        if strong_professional_strength >= ATTRIBUTE_CONFIDENCE:

            evidence = []

            if office_specific >= ATTRIBUTE_CONFIDENCE:
                evidence.append(
                    f"{specific_group}={specific_name} "
                    f"({specific_prob:.0%})"
                )

            if formal_strength >= ATTRIBUTE_CONFIDENCE:
                evidence.append(
                    f"formality={formality_name} "
                    f"({formality_prob:.0%})"
                )

            if style_name:
                evidence.append(
                    f"style={style_name} "
                    f"({style_prob:.0%})"
                )

            verdicts.append(
                (
                    "preferred",
                    (
                        "Visual attributes indicate a "
                        "professional/office-compatible style"
                        + (
                            ": " + ", ".join(evidence)
                            if evidence
                            else "."
                        )
                    ),
                )
            )

            return verdicts

        # ===========================================================
        # 4. Moderate casual evidence -> soft penalty
        # ===========================================================
        #
        # A garment can be somewhat casual without being categorically
        # incompatible. Keep it available so Phase 3/4 can use it when
        # necessary, but rank it below stronger office garments.
        # ===========================================================

        if strong_casual_strength >= 0.60:

            verdicts.append(
                (
                    "soft_penalty",
                    (
                        "Visual attributes indicate a moderately "
                        "casual style for office wear; the garment "
                        "remains eligible but should rank below "
                        "more professional alternatives."
                    ),
                )
            )

            return verdicts

        # ===========================================================
        # 5. Full-body garments require enough evidence to justify
        #    office selection.
        # ===========================================================

        if category in {
            "dress",
            "jumpsuit",
            "romper",
        }:

            verdicts.append(
                (
                    "hard_reject",
                    (
                        "This full-body garment does not have "
                        "sufficient professional or office-specific "
                        "visual evidence for formal office wear."
                    ),
                )
            )

            return verdicts

        # ===========================================================
        # 6. Ambiguous / neutral garments remain eligible.
        # ===========================================================

        if not attributes:

            verdicts.append(
                (
                    "neutral",
                    (
                        "No visual attributes are available to "
                        "establish office suitability; the garment "
                        "remains eligible for downstream ranking."
                    ),
                )
            )

            return verdicts

        verdicts.append(
            (
                "neutral",
                (
                    "No strong office incompatibility was detected "
                    "from the available visual attributes; the "
                    "garment remains eligible for outfit generation."
                ),
            )
        )

        return verdicts

    # ---------------------------------------------------------------
    # WEDDING
    # ---------------------------------------------------------------

    elif occasion == "wedding":

        if party_strength >= ATTRIBUTE_CONFIDENCE:

            verdicts.append(
                (
                    "preferred",
                    (
                        "Visual attributes indicate "
                        "party/evening suitability."
                    ),
                )
            )

        elif formal_strength >= ATTRIBUTE_CONFIDENCE:

            verdicts.append(
                (
                    "preferred",
                    (
                        "Visual attributes indicate "
                        "formal/event suitability."
                    ),
                )
            )

        elif casual_strength >= 0.75:

            verdicts.append(
                (
                    "soft_penalty",
                    (
                        "Visual attributes indicate a "
                        "strongly casual style for a wedding."
                    ),
                )
            )

        else:

            verdicts.append(
                (
                    "neutral",
                    (
                        "No strong wedding incompatibility "
                        "was detected."
                    ),
                )
            )

    # ---------------------------------------------------------------
    # DATE
    # ---------------------------------------------------------------

    elif occasion == "date":

        if party_strength >= ATTRIBUTE_CONFIDENCE:

            verdicts.append(
                (
                    "preferred",
                    (
                        "Visual attributes indicate "
                        "evening/occasion suitability."
                    ),
                )
            )

        elif (
            formal_strength >= ATTRIBUTE_CONFIDENCE
            or (
                style in {
                    "elegant",
                    "romantic",
                }
                and style_prob >= ATTRIBUTE_CONFIDENCE
            )
        ):

            verdicts.append(
                (
                    "preferred",
                    (
                        "Visual attributes indicate "
                        "an elevated date-appropriate style."
                    ),
                )
            )

        elif casual_strength >= 0.75:

            verdicts.append(
                (
                    "soft_penalty",
                    (
                        "Visual attributes indicate "
                        "a strongly casual style."
                    ),
                )
            )

        else:

            verdicts.append(
                (
                    "neutral",
                    (
                        "No strong date incompatibility "
                        "was detected."
                    ),
                )
            )

    # ---------------------------------------------------------------
    # COLLEGE / CASUAL
    # ---------------------------------------------------------------

    elif occasion in {
        "college",
        "casual",
    }:

        if casual_strength >= ATTRIBUTE_CONFIDENCE:

            verdicts.append(
                (
                    "preferred",
                    (
                        "Visual attributes indicate "
                        "a casual/college-compatible style."
                    ),
                )
            )

        elif formal_strength >= 0.75:

            verdicts.append(
                (
                    "soft_penalty",
                    (
                        "Visual attributes indicate a more "
                        "formal style than typical "
                        "college/casual wear."
                    ),
                )
            )

        else:

            verdicts.append(
                (
                    "neutral",
                    (
                        "No strong college/casual "
                        "incompatibility was detected."
                    ),
                )
            )

    # ---------------------------------------------------------------
    # GENERIC FALLBACK
    # ---------------------------------------------------------------

    else:

        verdicts.append(
            (
                "neutral",
                (
                    "No specific attribute-level rule exists "
                    f"for occasion '{occasion}'."
                ),
            )
        )

    return verdicts

# --------------------------------------------------------------------
# Main garment evaluation
# --------------------------------------------------------------------

def evaluate_garment(
    garment_json,
    occasion,
    temperature_c,
    rain,
    force_unclassified_reason=None,
):
    """
    Evaluate one Phase 1 garment.

    Attribute-first design:

        category
            +
        visual attributes
            +
        weather attributes
            ↓
        Phase 2 decision

    No filename information is used.
    """

    # ---------------------------------------------------------------
    # Category information
    # ---------------------------------------------------------------

    category_list = _normalize_category_predictions(
        garment_json.get(
            "category"
        )
    )

    category_name = (
        category_list[0].get(
            "name"
        )
        if category_list
        else None
    )

    category_prob = (
        category_list[0].get(
            "probability"
        )
        if category_list
        else None
    )

    # ---------------------------------------------------------------
    # COMPLETE CLIP ATTRIBUTES
    # ---------------------------------------------------------------

    attributes = (
        garment_json.get(
            "attributes"
        )
        or {}
    )

    if not isinstance(
        attributes,
        dict,
    ):
        attributes = {}

    # ---------------------------------------------------------------
    # Category confidence
    #
    # IMPORTANT:
    # This is diagnostic only.
    # It is NOT a hard filter.
    # ---------------------------------------------------------------

    confidence_info = (
        category_confidence_info(
            category_list
        )
    )

    category_confident = (
        confidence_info["confident"]
    )

    # ---------------------------------------------------------------
    # Fabric
    # ---------------------------------------------------------------

    fabric_name = _top_attr_name(
        attributes,
        "fabric",
        min_probability=FABRIC_CONFIDENCE_THRESHOLD,
    )

    # ---------------------------------------------------------------
    # Weather
    # ---------------------------------------------------------------

    bucket = weather_rules.temp_to_bucket(
        temperature_c
    )

    # ---------------------------------------------------------------
    # OCCASION
    # ---------------------------------------------------------------

    raw_occasion_verdicts = (
        _occasion_attribute_verdicts(
            occasion=occasion,
            category_name=category_name,
            attributes=attributes,
        )
    )

    # ---------------------------------------------------------------
    # TEMPERATURE
    # ---------------------------------------------------------------

    raw_temp_verdicts = (
        weather_rules.evaluate_temperature(
            bucket=bucket,
            category_name=category_name,
            category_confident=category_confident,
            fabric_name=fabric_name,
            category_predictions=category_list,
            attributes=attributes,
        )
    )

    # ---------------------------------------------------------------
    # RAIN
    # ---------------------------------------------------------------

    raw_rain_verdicts = (
        weather_rules.evaluate_rain(
            rain=rain,
            category_name=category_name,
            category_confident=category_confident,
            fabric_name=fabric_name,
            category_predictions=category_list,
            attributes=attributes,
        )
    )

    # ---------------------------------------------------------------
    # Convert verdicts to schema objects
    # ---------------------------------------------------------------

    occasion_verdicts = [
        Verdict(
            verdict,
            reason,
        )
        for verdict, reason
        in raw_occasion_verdicts
    ]

    weather_verdicts = [
        Verdict(
            verdict,
            reason,
        )
        for verdict, reason
        in (
            raw_temp_verdicts
            + raw_rain_verdicts
        )
    ]

    # ---------------------------------------------------------------
    # Combine evidence
    # ---------------------------------------------------------------

    all_verdicts = (
        raw_occasion_verdicts
        + raw_temp_verdicts
        + raw_rain_verdicts
    )

    hard_rejects = [
        (
            verdict,
            reason,
        )
        for verdict, reason
        in all_verdicts
        if verdict == "hard_reject"
    ]

    soft_penalties = [
        (
            verdict,
            reason,
        )
        for verdict, reason
        in all_verdicts
        if verdict == "soft_penalty"
    ]

    preferences = [
        (
            verdict,
            reason,
        )
        for verdict, reason
        in all_verdicts
        if verdict == "preferred"
    ]

    # ---------------------------------------------------------------
    # DEBUG OUTPUT
    # ---------------------------------------------------------------

    (
        formality_name,
        formality_prob,
    ) = _top_attribute(
        attributes,
        "formality",
    )

    (
        style_name,
        style_prob,
    ) = _top_attribute(
        attributes,
        "style",
    )

    (
        pattern_name,
        pattern_prob,
    ) = _top_attribute(
        attributes,
        "pattern",
    )

    (
        specific_group,
        specific_name,
        specific_prob,
    ) = _get_specific_attribute(
        attributes,
        category_name,
    )

    print(
        "\n[ATTRIBUTE PROFILE]"
        f" {garment_json.get('image', '<unknown>')}"
        f" | category={category_name}"
        f" | category_conf={category_prob}"
        f" | formality={formality_name}"
        f" ({formality_prob:.2%})"
        f" | style={style_name}"
        f" ({style_prob:.2%})"
        f" | pattern={pattern_name}"
        f" ({pattern_prob:.2%})"
        f" | specific={specific_group}:{specific_name}"
        f" ({specific_prob:.2%})",
        file=sys.stderr,
    )

    # ---------------------------------------------------------------
    # Occasion debug
    # ---------------------------------------------------------------

    if raw_occasion_verdicts:

        print(
            "[ATTRIBUTE OCCASION]"
            f" {category_name or '<unknown>'}"
            f" | occasion={occasion}"
            " | "
            + " | ".join(
                f"{verdict}: {reason}"
                for verdict, reason
                in raw_occasion_verdicts
            ),
            file=sys.stderr,
        )

    # ---------------------------------------------------------------
    # WEATHER debug
    # ---------------------------------------------------------------

    if (
        raw_temp_verdicts
        or raw_rain_verdicts
    ):

        print(
            "[WEATHER]"
            f" {category_name or '<unknown>'}"
            f" | temperature={temperature_c}C"
            f" | rain={rain}"
            " | "
            + " | ".join(
                f"{verdict}: {reason}"
                for verdict, reason
                in (
                    raw_temp_verdicts
                    + raw_rain_verdicts
                )
            ),
            file=sys.stderr,
        )

    # ---------------------------------------------------------------
    # FINAL DECISION
    # ---------------------------------------------------------------
    #
    # IMPORTANT:
    #
    # There is NO:
    #
    #     if not category_confident:
    #         filtered
    #
    # Category confidence is NOT a hard gate anymore.
    #
    # The garment is filtered only when an actual occasion/weather
    # rule says hard_reject.
    # ---------------------------------------------------------------

    if hard_rejects:

        decision = "filtered"

        reason = " ".join(
            reason
            for _, reason
            in hard_rejects
        )

    elif preferences and not soft_penalties:

        decision = "preferred"

        reason = " ".join(
            reason
            for _, reason
            in preferences
        )

    elif soft_penalties:

        decision = "suitable"

        reason = (
            "Acceptable, with soft penalties: "
            + " ".join(
                reason
                for _, reason
                in soft_penalties
            )
        )

    else:

        decision = "suitable"

        reason = (
            "No hard occasion/weather "
            "incompatibility was found."
        )

    # ---------------------------------------------------------------
    # Return Phase 2 schema
    # ---------------------------------------------------------------

    return GarmentDecision(
        image=garment_json.get(
            "image",
            "<unknown>",
        ),

        decision=decision,

        reason=reason.strip(),

        category_name=category_name,

        category_probability=category_prob,

        category_confident=category_confident,

        occasion_verdicts=occasion_verdicts,

        weather_verdicts=weather_verdicts,
    )


# --------------------------------------------------------------------
# Filter entire wardrobe
# --------------------------------------------------------------------

def filter_wardrobe(
    predictions_dir,
    occasion,
    temperature_c,
    rain,
    category_overrides=None,
):
    """
    Run Phase 2 against every Phase 1 JSON file.

    IMPORTANT DATA CONTRACT:

        Phase 1 JSON filename
                ↓
        stable garment identifier
                ↓
        Phase 2
                ↓
        same identifier
                ↓
        Phase 3
                ↓
        same identifier
                ↓
        Phase 4 embedding lookup

    The filename is used ONLY as the stable garment identifier.

    It is NEVER used to determine:
        - category
        - occasion suitability
        - weather suitability
        - style
        - formality

    Classification remains entirely based on Phase 1 category
    predictions and visual attributes.
    """

    predictions_dir = Path(predictions_dir)

    json_files = sorted(
        predictions_dir.glob("*.json")
    )

    garments = []

    for json_file in json_files:

        # -------------------------------------------------------------
        # Load Phase 1 prediction
        # -------------------------------------------------------------

        garment_json = load_garment(
            json_file
        )

        if not isinstance(
            garment_json,
            dict,
        ):
            print(
                f"WARNING: Skipping invalid Phase 1 JSON: "
                f"{json_file}",
                file=sys.stderr,
            )
            continue

        # -------------------------------------------------------------
        # PRESERVE GARMENT IDENTIFIER
        # -------------------------------------------------------------
        #
        # Phase 1 may already contain:
        #
        #     garment_id
        #     image
        #     id
        #
        # If none exists, use the JSON filename stem as the stable
        # identifier.
        #
        # THIS IS NOT FILENAME-BASED CLASSIFICATION.
        #
        # We are NOT doing:
        #
        #     "bottom1" -> Bottom
        #
        # We are only doing:
        #
        #     bottom1.json -> garment ID "bottom1"
        #
        # Category classification continues to come from the CLIP
        # predictions already stored inside garment_json.
        # -------------------------------------------------------------

        existing_id = (
            garment_json.get("garment_id")
            or garment_json.get("image")
            or garment_json.get("id")
        )

        if existing_id is not None:

            existing_id = str(
                existing_id
            ).strip()

        if (
            not existing_id
            or existing_id.lower() == "<unknown>"
        ):

            garment_id = json_file.stem

        else:

            garment_id = existing_id

        # -------------------------------------------------------------
        # Put the stable identifier into the object passed through
        # Phase 2.
        # -------------------------------------------------------------

        garment_json["image"] = garment_id

        # -------------------------------------------------------------
        # DEBUG
        # -------------------------------------------------------------

        print(
            f"[PHASE 2 INPUT] "
            f"id={garment_id} "
            f"| source={json_file.name}",
            file=sys.stderr,
        )

        # -------------------------------------------------------------
        # Evaluate garment.
        #
        # IMPORTANT:
        # evaluate_garment() still determines category and suitability
        # exclusively from the Phase 1 prediction/attribute data.
        # -------------------------------------------------------------

        decision = evaluate_garment(
            garment_json=garment_json,
            occasion=occasion,
            temperature_c=temperature_c,
            rain=rain,
            force_unclassified_reason=None,
        )

        garments.append(
            decision
        )

    # -------------------------------------------------------------
    # Return complete Phase 2 result
    # -------------------------------------------------------------

    return FilterResult(
        occasion=occasion,
        temperature_c=temperature_c,
        rain=rain,
        garments=garments,
    )


# --------------------------------------------------------------------
# Filename overrides
# --------------------------------------------------------------------
#
# Completely disabled.
#
# This exists only so old code that imports DEFAULT_KNOWN_OVERRIDES
# does not break.
# --------------------------------------------------------------------

DEFAULT_KNOWN_OVERRIDES = {}


# --------------------------------------------------------------------
# Test scenarios
# --------------------------------------------------------------------

TEST_SCENARIOS = [

    {
        "name": "casual_college",
        "occasion": "college",
        "temperature_c": 25,
        "rain": False,
    },

    {
        "name": "office",
        "occasion": "office",
        "temperature_c": 30,
        "rain": False,
    },

    {
        "name": "wedding",
        "occasion": "wedding",
        "temperature_c": 28,
        "rain": False,
    },

    {
        "name": "date",
        "occasion": "date",
        "temperature_c": 24,
        "rain": False,
    },

    {
        "name": "hot_weather",
        "occasion": "casual",
        "temperature_c": 35,
        "rain": False,
    },

    {
        "name": "rainy_weather",
        "occasion": "casual",
        "temperature_c": 24,
        "rain": True,
    },
]


# --------------------------------------------------------------------
# Run all scenarios
# --------------------------------------------------------------------

def run_all_test_scenarios(
    predictions_dir,
    output_dir=None,
    use_known_overrides=False,
):
    """
    Run all Phase 2 test scenarios.

    Filename overrides are intentionally disabled.
    """

    results = {}

    for scenario in TEST_SCENARIOS:

        result = filter_wardrobe(
            predictions_dir=predictions_dir,
            occasion=scenario["occasion"],
            temperature_c=scenario["temperature_c"],
            rain=scenario["rain"],
            category_overrides={},
        )

        results[
            scenario["name"]
        ] = result.to_dict()

        if output_dir is not None:

            out_path = (
                Path(output_dir)
                / f"{scenario['name']}.json"
            )

            out_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                out_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    result.to_dict(),
                    f,
                    indent=2,
                )

    return results


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 2: filter Phase 1 garment "
            "JSON by occasion + weather."
        )
    )

    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="predictions",
        help=(
            "Directory containing Phase 1 "
            "*.json files."
        ),
    )

    parser.add_argument(
        "--occasion",
        type=str,
        choices=sorted(
            occasion_rules.SUPPORTED_OCCASIONS
        ),
    )

    parser.add_argument(
        "--temperature_c",
        type=float,
    )

    parser.add_argument(
        "--rain",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--run_test_scenarios",
        action="store_true",
    )

    parser.add_argument(
        "--test_output_dir",
        type=str,
        default="predictions_filtered",
    )

    parser.add_argument(
        "--no_known_overrides",
        action="store_true",
        help=(
            "Deprecated. Filename overrides "
            "are disabled permanently."
        ),
    )

    return parser.parse_args()


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():

    args = parse_args()

    print(
        "=============================================================="
    )
    print(
        "PHASE 2 — ATTRIBUTE-FIRST FILTER"
    )
    print(
        "=============================================================="
    )

    print(
        f"CLIP category confidence: "
        f"{CONFIDENCE_THRESHOLD:.0%}"
        f" / margin {MARGIN_THRESHOLD:.0%}",
        file=sys.stderr,
    )

    print(
        "Filename-based category detection: DISABLED",
        file=sys.stderr,
    )

    print(
        "Category confidence is NOT a hard filter.",
        file=sys.stderr,
    )

    # ---------------------------------------------------------------
    # Test scenario mode
    # ---------------------------------------------------------------

    if args.run_test_scenarios:

        results = run_all_test_scenarios(
            predictions_dir=args.predictions_dir,
            output_dir=args.test_output_dir,
            use_known_overrides=False,
        )

        print(
            json.dumps(
                results,
                indent=2,
            )
        )

        return

    # ---------------------------------------------------------------
    # Single-request mode
    # ---------------------------------------------------------------

    if (
        args.occasion is None
        or args.temperature_c is None
    ):

        print(
            "--occasion and --temperature_c "
            "are required unless "
            "--run_test_scenarios is set.",
            file=sys.stderr,
        )

        sys.exit(1)

    result = filter_wardrobe(
        predictions_dir=args.predictions_dir,
        occasion=args.occasion,
        temperature_c=args.temperature_c,
        rain=args.rain,
        category_overrides={},
    )

    result_dict = result.to_dict()

    print(
        json.dumps(
            result_dict,
            indent=2,
        )
    )

    if args.output:

        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                result_dict,
                f,
                indent=2,
            )

        print(
            f"\nSaved to {output_path}",
            file=sys.stderr,
        )


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

if __name__ == "__main__":
    main()