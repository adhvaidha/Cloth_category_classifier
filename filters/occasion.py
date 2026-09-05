"""
occasion.py

Phase 2 occasion suitability rules.

DeepFashion provides evidence about what a garment looks like, but it
does not directly predict whether something is appropriate for an
office, wedding, date, etc.

This module therefore combines:

    - category probabilities
    - DeepFashion attributes
    - explicit application-level occasion rules

IMPORTANT:
Uncertain category predictions are NOT automatically rejected.

Hard rejection is reserved for strong evidence of an incompatible
garment. Otherwise the garment receives a soft penalty, preference,
or remains neutral.
"""

# --------------------------------------------------------------------
# FORMALITY TIERS
# --------------------------------------------------------------------

TIER_CASUAL = 1
TIER_SMART = 2
TIER_FORMAL = 3


CATEGORY_TIER = {

    # ---------------------------------------------------------------
    # Casual
    # ---------------------------------------------------------------

    "Tee": TIER_CASUAL,
    "Tank": TIER_CASUAL,
    "Hoodie": TIER_CASUAL,
    "Flannel": TIER_CASUAL,
    "Anorak": TIER_CASUAL,
    "Bomber": TIER_CASUAL,
    "Parka": TIER_CASUAL,
    "Poncho": TIER_CASUAL,
    "Jersey": TIER_CASUAL,
    "Jeans": TIER_CASUAL,
    "Jeggings": TIER_CASUAL,
    "Joggers": TIER_CASUAL,
    "Leggings": TIER_CASUAL,
    "Sweatpants": TIER_CASUAL,
    "Sweatshorts": TIER_CASUAL,
    "Shorts": TIER_CASUAL,
    "Cutoffs": TIER_CASUAL,
    "Gauchos": TIER_CASUAL,
    "Jodhpurs": TIER_CASUAL,
    "Sarong": TIER_CASUAL,
    "Trunks": TIER_CASUAL,
    "Onesie": TIER_CASUAL,
    "Robe": TIER_CASUAL,
    "Sweater": TIER_CASUAL,

    # ---------------------------------------------------------------
    # Smart casual
    # ---------------------------------------------------------------

    "Blouse": TIER_SMART,
    "Button-Down": TIER_SMART,
    "Cardigan": TIER_SMART,
    "Turtleneck": TIER_SMART,
    "Top": TIER_SMART,
    "Skirt": TIER_SMART,
    "Chinos": TIER_SMART,
    "Culottes": TIER_SMART,
    "Capris": TIER_SMART,
    "Henley": TIER_SMART,
    "Halter": TIER_SMART,
    "Coverup": TIER_SMART,
    "Romper": TIER_SMART,
    "Kimono": TIER_SMART,
    "Nightdress": TIER_SMART,

    # ---------------------------------------------------------------
    # Formal / dressy
    # ---------------------------------------------------------------

    "Blazer": TIER_FORMAL,
    "Coat": TIER_FORMAL,
    "Dress": TIER_FORMAL,
    "Jumpsuit": TIER_FORMAL,
    "Peacoat": TIER_FORMAL,
    "Shirtdress": TIER_FORMAL,
    "Sundress": TIER_FORMAL,
    "Cape": TIER_FORMAL,
    "Caftan": TIER_FORMAL,
    "Kaftan": TIER_FORMAL,

    # Jacket is generally casual/outerwear in this application.
    "Jacket": TIER_CASUAL,
}


# --------------------------------------------------------------------
# SPECIAL GROUPS
# --------------------------------------------------------------------

WORKOUT_CATEGORIES = {
    "Tank",
    "Joggers",
    "Sweatpants",
    "Leggings",
    "Jersey",
    "Hoodie",
    "Trunks",
    "Sweatshorts",
}


LOUNGE_CATEGORIES = {
    "Sweatpants",
    "Sweatshorts",
    "Joggers",
    "Robe",
    "Onesie",
}


CASUAL_CATEGORIES = {
    name
    for name, tier in CATEGORY_TIER.items()
    if tier == TIER_CASUAL
}


SMART_CATEGORIES = {
    name
    for name, tier in CATEGORY_TIER.items()
    if tier == TIER_SMART
}


FORMAL_CATEGORIES = {
    name
    for name, tier in CATEGORY_TIER.items()
    if tier == TIER_FORMAL
}


# --------------------------------------------------------------------
# OCCASION RULES
# --------------------------------------------------------------------

OCCASION_RULES = {

    "casual": {
        "hard_reject": set(),
        "soft_penalize": set(),
        "preferred": CASUAL_CATEGORIES | SMART_CATEGORIES,
    },

    "college": {
        "hard_reject": set(),
        "soft_penalize": FORMAL_CATEGORIES,
        "preferred": CASUAL_CATEGORIES | SMART_CATEGORIES,
    },

    "travel": {
        "hard_reject": set(),
        "soft_penalize": FORMAL_CATEGORIES,
        "preferred": CASUAL_CATEGORIES | SMART_CATEGORIES,
    },

    "office": {

        # Only clearly unsuitable items are hard rejected.
        # Casual clothing is NOT automatically removed anymore.
        "hard_reject": {
            "Trunks",
            "Sweatshorts",
            "Onesie",
            "Robe",
        },

        "soft_penalize": {
            "Tee",
            "Tank",
            "Hoodie",
            "Sweatpants",
            "Joggers",
            "Shorts",
            "Cutoffs",
            "Jersey",
            "Jeans",
        },

        "preferred": SMART_CATEGORIES | FORMAL_CATEGORIES,
    },

    "formal": {

        "hard_reject": WORKOUT_CATEGORIES | {
            "Trunks",
            "Sweatshorts",
            "Onesie",
            "Robe",
        },

        "soft_penalize": CASUAL_CATEGORIES,

        "preferred": FORMAL_CATEGORIES,
    },

    "wedding": {

        "hard_reject": WORKOUT_CATEGORIES | {
            "Trunks",
            "Sweatshorts",
            "Onesie",
            "Robe",
        },

        "soft_penalize": CASUAL_CATEGORIES,

        "preferred": FORMAL_CATEGORIES | SMART_CATEGORIES,
    },

    "party": {

        "hard_reject": LOUNGE_CATEGORIES,

        "soft_penalize": {
            "Tee",
            "Tank",
            "Jeans",
            "Shorts",
            "Cutoffs",
        },

        "preferred": SMART_CATEGORIES | FORMAL_CATEGORIES,
    },

    "date": {

        "hard_reject": LOUNGE_CATEGORIES,

        "soft_penalize": {
            "Tee",
            "Tank",
        },

        "preferred": SMART_CATEGORIES | FORMAL_CATEGORIES,
    },

    "workout": {

        "hard_reject": FORMAL_CATEGORIES | {
            "Jeans",
            "Skirt",
            "Chinos",
            "Culottes",
            "Capris",
            "Coat",
            "Cardigan",
            "Blazer",
        },

        "soft_penalize": SMART_CATEGORIES - {
            "Skirt",
            "Chinos",
            "Culottes",
            "Capris",
        },

        "preferred": WORKOUT_CATEGORIES,
    },
}


SUPPORTED_OCCASIONS = set(OCCASION_RULES.keys())


# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------


def get_tier(category_name):
    """Return the manually curated formality tier."""

    return CATEGORY_TIER.get(
        category_name,
        TIER_SMART,
    )


def _probability_map(category_predictions):
    """
    Convert Phase 1 category predictions into:

        {
            "Shorts": 0.47,
            "Jeans": 0.37,
            ...
        }
    """

    result = {}

    if not category_predictions:
        return result

    for item in category_predictions:

        if not isinstance(item, dict):
            continue

        name = item.get("name")
        probability = item.get("probability")

        if name is None or probability is None:
            continue

        try:
            result[str(name)] = float(probability)

        except (TypeError, ValueError):
            continue

    return result


def _attribute_map(attributes):
    """
    Extract the highest-probability attribute from each DeepFashion
    attribute group.

    Example:

        {
            "fabric": ("leather", 0.75),
            "style": ("summer", 0.18)
        }
    """

    result = {}

    if not isinstance(attributes, dict):
        return result

    for group, values in attributes.items():

        if not isinstance(values, list) or not values:
            continue

        best = None

        for item in values:

            if not isinstance(item, dict):
                continue

            name = item.get("name")
            probability = item.get("probability")

            if name is None or probability is None:
                continue

            try:
                probability = float(probability)

            except (TypeError, ValueError):
                continue

            if best is None or probability > best[1]:
                best = (
                    str(name),
                    probability,
                )

        if best is not None:
            result[group] = best

    return result


# --------------------------------------------------------------------
# MAIN OCCASION EVALUATION
# --------------------------------------------------------------------


def evaluate_occasion(
    occasion,
    category_name=None,
    category_confident=False,
    category_predictions=None,
    attributes=None,
):
    """
    Evaluate one garment for one occasion.

    IMPORTANT:

    The old implementation did this:

        if not category_confident:
            return []

    That meant uncertain garments received no useful occasion
    evaluation and could later become "unclassified".

    The new implementation instead examines the entire category
    probability distribution and DeepFashion attributes.

    Parameters
    ----------
    occasion:
        Supported occasion name.

    category_name:
        Top-1 category.

    category_confident:
        Legacy confidence result. Kept for compatibility.

    category_predictions:
        Complete Phase 1 category prediction list.

    attributes:
        Complete Phase 1 attribute dictionary.

    Returns
    -------
    list of (verdict, reason)

    verdict is one of:

        "hard_reject"
        "soft_penalty"
        "preferred"
    """

    if occasion not in OCCASION_RULES:

        raise ValueError(
            f"Unsupported occasion: {occasion!r}. "
            f"Supported: {sorted(SUPPORTED_OCCASIONS)}"
        )

    rules = OCCASION_RULES[occasion]

    predictions = _probability_map(
        category_predictions
    )

    attrs = _attribute_map(
        attributes
    )

    verdicts = []

    # ---------------------------------------------------------------
    # Preserve the supplied top category
    # ---------------------------------------------------------------

    if (
        category_name
        and category_name not in predictions
    ):

        predictions[category_name] = (
            1.0
            if category_confident
            else 0.0
        )

    # ---------------------------------------------------------------
    # CATEGORY EVIDENCE
    # ---------------------------------------------------------------

    for category, probability in predictions.items():

        # Strong evidence
        strong = probability >= 0.70

        # Moderate evidence
        moderate = probability >= 0.35

        # -----------------------------------------------------------
        # Hard rejection
        # -----------------------------------------------------------

        if (
            category in rules["hard_reject"]
            and strong
        ):

            verdicts.append(
                (
                    "hard_reject",
                    (
                        f"Strong category evidence "
                        f"({probability:.0%}) for "
                        f"'{category}', which is not "
                        f"suitable for a {occasion} "
                        f"occasion."
                    ),
                )
            )

        # -----------------------------------------------------------
        # Soft penalty
        # -----------------------------------------------------------

        elif (
            category in rules["soft_penalize"]
            and moderate
        ):

            verdicts.append(
                (
                    "soft_penalty",
                    (
                        f"Category evidence "
                        f"({probability:.0%}) suggests "
                        f"'{category}', which is less "
                        f"suitable for a {occasion} "
                        f"occasion."
                    ),
                )
            )

        # -----------------------------------------------------------
        # Preferred
        # -----------------------------------------------------------

        elif (
            category in rules["preferred"]
            and moderate
        ):

            verdicts.append(
                (
                    "preferred",
                    (
                        f"Category evidence "
                        f"({probability:.0%}) suggests "
                        f"'{category}', which suits a "
                        f"{occasion} occasion."
                    ),
                )
            )

    # ---------------------------------------------------------------
    # DEEPFASHION ATTRIBUTES
    # ---------------------------------------------------------------

    style_name, style_prob = attrs.get(
        "style",
        (None, 0.0),
    )

    texture_name, texture_prob = attrs.get(
        "texture",
        (None, 0.0),
    )

    shape_name, shape_prob = attrs.get(
        "shape",
        (None, 0.0),
    )

    part_name, part_prob = attrs.get(
        "part",
        (None, 0.0),
    )

    # ---------------------------------------------------------------
    # CASUAL / COLLEGE / TRAVEL
    # ---------------------------------------------------------------

    if occasion in {
        "casual",
        "college",
        "travel",
    }:

        if (
            style_name == "summer"
            and style_prob >= 0.50
        ):

            verdicts.append(
                (
                    "preferred",
                    (
                        f"DeepFashion style attribute "
                        f"'{style_name}' supports a "
                        f"{occasion} setting."
                    ),
                )
            )

    # ---------------------------------------------------------------
    # OFFICE / FORMAL / WEDDING
    # ---------------------------------------------------------------

    if occasion in {
        "office",
        "formal",
        "wedding",
    }:

        # Classic styling is a positive signal.
        if (
            style_name == "classic"
            and style_prob >= 0.50
        ):

            verdicts.append(
                (
                    "preferred",
                    (
                        f"DeepFashion style attribute "
                        f"'{style_name}' supports a "
                        f"more polished appearance."
                    ),
                )
            )

        # Floral and print are NEVER hard rejected.
        #
        # They may be less formal, but they can still be appropriate
        # depending on the complete outfit.
        if (
            texture_name in {
                "floral",
                "print",
            }
            and texture_prob >= 0.70
        ):

            verdicts.append(
                (
                    "soft_penalty",
                    (
                        f"Strong '{texture_name}' pattern "
                        f"evidence may be less formal, "
                        f"but is not automatically "
                        f"inappropriate."
                    ),
                )
            )

    # ---------------------------------------------------------------
    # LONG SLEEVES
    # ---------------------------------------------------------------

    if (
        occasion in {
            "office",
            "formal",
            "wedding",
            "date",
        }
        and part_name in {
            "long sleeve",
            "sleeve",
        }
        and part_prob >= 0.50
    ):

        verdicts.append(
            (
                "preferred",
                (
                    "DeepFashion sleeve information "
                    "supports a more covered/"
                    "polished outfit."
                ),
            )
        )

    # ---------------------------------------------------------------
    # CROP / MINI
    # ---------------------------------------------------------------
    #
    # These are soft signals only.
    #
    # They do NOT automatically reject a garment.
    # ---------------------------------------------------------------

    if (
        occasion in {
            "office",
            "formal",
            "wedding",
        }
        and shape_name in {
            "crop",
            "mini",
        }
        and shape_prob >= 0.70
    ):

        verdicts.append(
            (
                "soft_penalty",
                (
                    f"Strong '{shape_name}' shape "
                    f"evidence may be less appropriate "
                    f"for this occasion."
                ),
            )
        )

    return verdicts