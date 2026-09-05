"""
weather.py

Phase 2 weather suitability rules.

Weather decisions use DeepFashion category probabilities and attributes
as evidence rather than treating uncertain predictions as absolute truth.

Hard rejection is reserved for strong, obvious incompatibilities.
Most uncertain cases receive a soft penalty or remain neutral.
"""

# --------------------------------------------------------------------
# Category groups
# --------------------------------------------------------------------

LIGHT_CATEGORIES = {
    "Tank", "Halter", "Tee", "Shorts", "Cutoffs", "Sundress",
    "Culottes", "Gauchos", "Capris", "Coverup", "Sarong",
    "Romper", "Skirt", "Top",
}

WARM_HEAVY_CATEGORIES = {
    "Coat", "Parka", "Peacoat", "Anorak", "Bomber", "Sweater",
    "Turtleneck", "Cardigan", "Hoodie", "Jacket", "Flannel",
    "Kimono", "Robe", "Joggers", "Sweatpants",
}


# --------------------------------------------------------------------
# Fabric groups
# --------------------------------------------------------------------

LIGHT_FABRICS = {
    "linen",
    "chiffon",
    "mesh",
    "satin",
}

HEAVY_FABRICS = {
    "knit",
    "fur",
    "suede",
    "corduroy",
    "velvet",
    "velveteen",
    "wool",
}

RAIN_UNFRIENDLY_FABRICS = {
    "suede",
    "leather",
    "satin",
    "velvet",
    "velveteen",
    "chiffon",
    "lace",
}

RAIN_FRIENDLY_FABRICS = {
    "nylon",
}

RAIN_FRIENDLY_CATEGORIES = {
    "Coat",
    "Parka",
    "Anorak",
    "Jacket",
}


# --------------------------------------------------------------------
# Temperature buckets
# --------------------------------------------------------------------

TEMP_BUCKETS = [
    ("hot", 30.0, float("inf")),
    ("warm", 24.0, 30.0),
    ("mild", 18.0, 24.0),
    ("cool", 11.0, 18.0),
    ("cold", float("-inf"), 11.0),
]

SUPPORTED_TEMP_BUCKETS = {bucket[0] for bucket in TEMP_BUCKETS}


def temp_to_bucket(temperature_c):
    """Convert Celsius temperature to a weather bucket."""

    for name, low, high in TEMP_BUCKETS:
        if low <= temperature_c < high:
            return name

    return "mild"


def _probability_map(predictions):
    """
    Convert a Phase 1 prediction list into:

        {"Coat": 0.82, "Jacket": 0.11, ...}
    """

    result = {}

    if not predictions:
        return result

    for item in predictions:
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
                best = (str(name), probability)

        if best is not None:
            result[group] = best

    return result


def _fabric_verdict(bucket, fabric_name, fabric_probability):
    """
    Fabric is treated as evidence, not absolute truth.

    Fabric predictions do not normally create hard exclusions because
    DeepFashion attributes themselves can be noisy.
    """

    if fabric_name is None:
        return None

    # Require reasonably strong attribute evidence before applying a
    # weather preference/penalty.
    if fabric_probability < 0.60:
        return None

    if bucket == "hot" and fabric_name in HEAVY_FABRICS:
        return (
            "soft_penalty",
            f"Strong fabric evidence ({fabric_probability:.0%}) for "
            f"'{fabric_name}' suggests the garment may feel warm in "
            f"hot weather.",
        )

    if bucket == "cold" and fabric_name in LIGHT_FABRICS:
        return (
            "soft_penalty",
            f"Strong fabric evidence ({fabric_probability:.0%}) for "
            f"'{fabric_name}' suggests the garment may feel too light "
            f"for cold weather.",
        )

    if bucket == "hot" and fabric_name in LIGHT_FABRICS:
        return (
            "preferred",
            f"Strong fabric evidence ({fabric_probability:.0%}) for "
            f"'{fabric_name}' suits hot weather.",
        )

    if bucket == "cold" and fabric_name in HEAVY_FABRICS:
        return (
            "preferred",
            f"Strong fabric evidence ({fabric_probability:.0%}) for "
            f"'{fabric_name}' suits cold weather.",
        )

    return None


def evaluate_temperature(
    bucket,
    category_name=None,
    category_confident=False,
    fabric_name=None,
    category_predictions=None,
    attributes=None,
):
    """
    Evaluate a garment against temperature.

    Category probabilities are used instead of requiring one confident
    top-1 category.

    Strong category evidence can create a hard rejection for obvious
    temperature mismatches.

    Moderate/uncertain evidence produces soft penalties instead.
    """

    if bucket not in SUPPORTED_TEMP_BUCKETS:
        raise ValueError(
            f"Unsupported temperature bucket: {bucket!r}"
        )

    verdicts = []

    predictions = _probability_map(category_predictions)
    attrs = _attribute_map(attributes)

    if category_name and category_name not in predictions:
        predictions[category_name] = (
            1.0 if category_confident else 0.0
        )

    # ---------------------------------------------------------------
    # Category evidence
    # ---------------------------------------------------------------

    for category, probability in predictions.items():

        strong = probability >= 0.70
        moderate = probability >= 0.35

        if bucket == "hot":

            if category in WARM_HEAVY_CATEGORIES:

                if strong:
                    verdicts.append(
                        (
                            "hard_reject",
                            f"Strong category evidence ({probability:.0%}) "
                            f"for '{category}', which is likely too warm "
                            f"for {bucket} weather.",
                        )
                    )

                elif moderate:
                    verdicts.append(
                        (
                            "soft_penalty",
                            f"Category evidence ({probability:.0%}) "
                            f"suggests '{category}', which may be too "
                            f"warm for {bucket} weather.",
                        )
                    )

            elif category in LIGHT_CATEGORIES and moderate:

                verdicts.append(
                    (
                        "preferred",
                        f"Category evidence ({probability:.0%}) suggests "
                        f"'{category}', which suits hot weather.",
                    )
                )

        elif bucket == "warm":

            if category in WARM_HEAVY_CATEGORIES and moderate:
                verdicts.append(
                    (
                        "soft_penalty",
                        f"Category evidence ({probability:.0%}) suggests "
                        f"'{category}', which may run warm.",
                    )
                )

            elif category in LIGHT_CATEGORIES and moderate:
                verdicts.append(
                    (
                        "preferred",
                        f"Category evidence ({probability:.0%}) suggests "
                        f"'{category}', which suits warm weather.",
                    )
                )

        elif bucket == "cool":

            if category in LIGHT_CATEGORIES and moderate:
                verdicts.append(
                    (
                        "soft_penalty",
                        f"Category evidence ({probability:.0%}) suggests "
                        f"'{category}', which may be too light for cool "
                        f"weather.",
                    )
                )

            elif category in WARM_HEAVY_CATEGORIES and moderate:
                verdicts.append(
                    (
                        "preferred",
                        f"Category evidence ({probability:.0%}) suggests "
                        f"'{category}', which suits cool weather.",
                    )
                )

        elif bucket == "cold":

            if category in LIGHT_CATEGORIES:

                if strong:
                    verdicts.append(
                        (
                            "hard_reject",
                            f"Strong category evidence ({probability:.0%}) "
                            f"for '{category}', which is likely too light "
                            f"for cold weather.",
                        )
                    )

                elif moderate:
                    verdicts.append(
                        (
                            "soft_penalty",
                            f"Category evidence ({probability:.0%}) "
                            f"suggests '{category}', which may be too "
                            f"light for cold weather.",
                        )
                    )

            elif category in WARM_HEAVY_CATEGORIES and moderate:

                verdicts.append(
                    (
                        "preferred",
                        f"Category evidence ({probability:.0%}) suggests "
                        f"'{category}', which suits cold weather.",
                    )
                )

    # ---------------------------------------------------------------
    # Fabric evidence
    # ---------------------------------------------------------------

    if fabric_name is None:
        fabric_name, fabric_probability = attrs.get(
            "fabric",
            (None, 0.0),
        )
    else:
        # If caller supplies the fabric name but not its probability,
        # use the Phase 1 attribute map where available.
        _, detected_probability = attrs.get(
            "fabric",
            (None, 0.0),
        )
        fabric_probability = detected_probability

    fabric_verdict = _fabric_verdict(
        bucket,
        fabric_name,
        fabric_probability,
    )

    if fabric_verdict is not None:
        verdicts.append(fabric_verdict)

    # ---------------------------------------------------------------
    # Style/part evidence
    # ---------------------------------------------------------------

    style_name, style_probability = attrs.get(
        "style",
        (None, 0.0),
    )

    part_name, part_probability = attrs.get(
        "part",
        (None, 0.0),
    )

    if bucket in {"hot", "warm"}:

        if (
            style_name == "summer"
            and style_probability >= 0.60
        ):
            verdicts.append(
                (
                    "preferred",
                    f"Strong summer-style evidence "
                    f"({style_probability:.0%}) supports warm weather.",
                )
            )

        if (
            part_name in {"long sleeve"}
            and part_probability >= 0.70
        ):
            verdicts.append(
                (
                    "soft_penalty",
                    f"Strong long-sleeve evidence "
                    f"({part_probability:.0%}) may feel warm in "
                    f"{bucket} weather.",
                )
            )

    if bucket in {"cool", "cold"}:

        if (
            part_name in {"long sleeve"}
            and part_probability >= 0.60
        ):
            verdicts.append(
                (
                    "preferred",
                    f"Strong long-sleeve evidence "
                    f"({part_probability:.0%}) supports cooler weather.",
                )
            )

    return verdicts


def evaluate_rain(
    rain,
    category_name=None,
    category_confident=False,
    fabric_name=None,
    category_predictions=None,
    attributes=None,
):
    """
    Evaluate rain suitability.

    Rain-related fabric predictions are now soft evidence rather than
    automatic hard rejection.

    This prevents a noisy DeepFashion fabric prediction from deleting
    an otherwise usable garment.
    """

    if not rain:
        return []

    predictions = _probability_map(category_predictions)
    attrs = _attribute_map(attributes)

    verdicts = []

    if category_name and category_name not in predictions:
        predictions[category_name] = (
            1.0 if category_confident else 0.0
        )

    # ---------------------------------------------------------------
    # Fabric
    # ---------------------------------------------------------------

    if fabric_name is None:
        fabric_name, fabric_probability = attrs.get(
            "fabric",
            (None, 0.0),
        )
    else:
        _, fabric_probability = attrs.get(
            "fabric",
            (None, 0.0),
        )

    if (
        fabric_name in RAIN_UNFRIENDLY_FABRICS
        and fabric_probability >= 0.70
    ):
        verdicts.append(
            (
                "soft_penalty",
                f"Strong fabric evidence ({fabric_probability:.0%}) "
                f"for '{fabric_name}' may be unsuitable for rain.",
            )
        )

    elif (
        fabric_name in RAIN_FRIENDLY_FABRICS
        and fabric_probability >= 0.50
    ):
        verdicts.append(
            (
                "preferred",
                f"Fabric evidence ({fabric_probability:.0%}) for "
                f"'{fabric_name}' is rain-friendly.",
            )
        )

    # ---------------------------------------------------------------
    # Rain-protective categories
    # ---------------------------------------------------------------

    for category, probability in predictions.items():

        if (
            category in RAIN_FRIENDLY_CATEGORIES
            and probability >= 0.35
        ):
            verdicts.append(
                (
                    "preferred",
                    f"Category evidence ({probability:.0%}) suggests "
                    f"'{category}', which offers rain protection.",
                )
            )

    return verdicts