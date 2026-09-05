"""
generator.py

Phase 3: FILTERED GARMENTS -> STRUCTURALLY VALID CANDIDATE OUTFITS

Input:
    Phase 2 filter_engine output (*.json)

Output:
    Structurally plausible candidate outfits.

Phase 3 does NOT:
    - use ResNetItemEmbedder
    - use OutfitCompatibilityModel
    - rank or score outfits
    - judge color/style compatibility
    - build a frontend/API

IMPORTANT:
    Phase 3 preserves the original garment identifier supplied by
    Phase 2.

    A garment identifier must NEVER silently become:
        "<unknown>"
        None
        ""
        or another generated identifier.

    If Phase 2 has lost the image/garment identifier, the garment is
    excluded from Phase 3 and the problem is explicitly reported.
"""

import argparse
import json
import sys
from pathlib import Path


# --------------------------------------------------------------------
# PROJECT PATH
# --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------
# SCHEMA
# --------------------------------------------------------------------

from outfit_candidates.schema import (
    SurvivingGarment,
    ExcludedGarment,
    CandidateOutfit,
    RejectedPair,
    Phase3Result,
)


# --------------------------------------------------------------------
# DEEPFASHION STRUCTURAL TYPES
# --------------------------------------------------------------------

# DeepFashion:
#
#   1 = upper-body
#   2 = lower-body
#   3 = full-body
#
# We use the dataset's category type to determine structural slots.

SLOT_BY_CATEGORY_TYPE = {
    1: "upper",
    2: "lower",
    3: "full_body",
}


# --------------------------------------------------------------------
# VALID OUTER LAYERS
# --------------------------------------------------------------------

# Categories that can reasonably be worn over a full-body garment.

LAYERING_CATEGORIES = {
    "Shrug",
    "Cardigan",
    "Jacket",
    "Blazer",
    "Bomber",
    "Anorak",
    "Parka",
    "Peacoat",
    "Coat",
}


# --------------------------------------------------------------------
# CATEGORY SLOT MAP
# --------------------------------------------------------------------

def load_category_slot_map(anno_dir=PROJECT_ROOT):
    """
    Build:

        {category_name: structural_slot}

    directly from:

        Anno/list_category_cloth.txt

    DeepFashion category type:

        1 -> upper
        2 -> lower
        3 -> full_body
    """

    path = Path(anno_dir) / "Anno" / "list_category_cloth.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find DeepFashion category file: {path}"
        )

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    slot_map = {}

    # First line = count
    # Second line = header
    # Remaining lines = category name + type
    for line in lines[2:]:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        try:
            category_type = int(parts[-1])
        except ValueError:
            continue

        category_name = " ".join(parts[:-1])

        slot = SLOT_BY_CATEGORY_TYPE.get(category_type)

        if slot is not None:
            slot_map[category_name] = slot

        # Explicitly treat valid outer layers as upper-body
        # garments for structural outfit generation.
        if category_name in LAYERING_CATEGORIES:
            slot_map[category_name] = "upper"

    return slot_map


# --------------------------------------------------------------------
# PHASE 2 LOADING
# --------------------------------------------------------------------

def load_phase2_result(json_path):
    """Load one Phase 2 JSON result."""

    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------
# SURVIVORS
# --------------------------------------------------------------------

def get_survivors(phase2_result):
    """
    Return garments that Phase 2 did NOT actively filter.

    Surviving decisions:
        preferred
        suitable
        unclassified

    Only:
        filtered

    is excluded.
    """

    garments = phase2_result.get("garments", [])

    if not isinstance(garments, list):
        return []

    return [
        garment
        for garment in garments
        if isinstance(garment, dict)
        and garment.get("decision") != "filtered"
    ]


# --------------------------------------------------------------------
# GARMENT IDENTIFIER
# --------------------------------------------------------------------

def _get_garment_id(garment):
    """
    Return the stable garment identifier supplied by Phase 2.

    Supported identifier fields:

        garment_id
        id
        image

    The first valid value is used.

    IMPORTANT:

        This function NEVER creates "<unknown>".

        Missing identifiers return None so the caller can explicitly
        reject the broken garment and report the upstream problem.
    """

    if not isinstance(garment, dict):
        return None

    # Preferred explicit identifier.
    garment_id = garment.get("garment_id")

    if garment_id is not None:
        garment_id = str(garment_id).strip()

        if (
            garment_id
            and garment_id.lower() != "<unknown>"
        ):
            return garment_id

    # Alternate identifier.
    garment_id = garment.get("id")

    if garment_id is not None:
        garment_id = str(garment_id).strip()

        if (
            garment_id
            and garment_id.lower() != "<unknown>"
        ):
            return garment_id

    # Existing Phase 2 schema historically uses image.
    image = garment.get("image")

    if image is not None:
        image = str(image).strip()

        if (
            image
            and image.lower() != "<unknown>"
        ):
            return image

    return None


def _display_name(garment_id):
    """
    Return the filename portion of an identifier/path.

    This is ONLY for display/storage compatibility.

    It is NEVER used for classification.
    """

    if garment_id is None:
        return None

    return Path(str(garment_id)).name


# --------------------------------------------------------------------
# STRUCTURAL SLOT ASSIGNMENT
# --------------------------------------------------------------------

def _slot_for_garment(garment, category_slot_map):
    """
    Determine the structural slot for a surviving garment.

    Category confidence does not determine structural validity.

    If the predicted category maps to:

        upper
        lower
        full_body

    that slot is returned.

    Returns:

        (slot, reason_if_excluded)

    or:

        (None, reason)
    """

    details = garment.get("details", {})

    if not isinstance(details, dict):
        details = {}

    category = details.get("category", {})

    if not isinstance(category, dict):
        category = {}

    category_name = category.get("name")

    if not category_name:
        return (
            None,
            "No category prediction is available; cannot determine "
            "a structural outfit slot.",
        )

    category_name = str(category_name).strip()

    slot = category_slot_map.get(category_name)

    if slot is None:
        return (
            None,
            f"Category '{category_name}' has no known structural "
            f"slot mapping.",
        )

    return slot, None


# --------------------------------------------------------------------
# STRUCTURAL VALIDITY
# --------------------------------------------------------------------

def _classify_pair(
    garment_a,
    garment_b,
    slot_a,
    slot_b,
    include_layering=True,
):
    """
    Determine whether two structural slots can coexist.

    Valid:
        upper + lower
        full_body + valid outer upper layer

    Invalid:
        upper + upper
        lower + lower
        full_body + full_body
        full_body + lower
    """

    pair = {slot_a, slot_b}

    # ---------------------------------------------------------------
    # Normal outfit
    # ---------------------------------------------------------------

    if pair == {"upper", "lower"}:
        return True, "upper_lower", None

    # ---------------------------------------------------------------
    # Full-body layering
    # ---------------------------------------------------------------

    if pair == {"full_body", "upper"}:

        if not include_layering:
            return (
                False,
                None,
                "Layering disabled by configuration "
                "(full_body + upper not generated).",
            )

        if slot_a == "upper":
            upper_garment = garment_a
        else:
            upper_garment = garment_b

        details = upper_garment.get("details", {})

        if not isinstance(details, dict):
            details = {}

        category = details.get("category", {})

        if not isinstance(category, dict):
            category = {}

        upper_category = category.get("name")

        if upper_category not in LAYERING_CATEGORIES:
            return (
                False,
                None,
                f"{upper_category} is not a valid outer layer "
                "for a full-body garment.",
            )

        return True, "full_body_layered", None

    # ---------------------------------------------------------------
    # Invalid combinations
    # ---------------------------------------------------------------

    if slot_a == "upper" and slot_b == "upper":
        return (
            False,
            None,
            "Two upper-body garments cannot both fill the "
            "single top slot in this basic outfit structure.",
        )

    if slot_a == "lower" and slot_b == "lower":
        return (
            False,
            None,
            "Two lower-body garments cannot be worn together.",
        )

    if slot_a == "full_body" and slot_b == "full_body":
        return (
            False,
            None,
            "Only one full-body garment can be worn at a time.",
        )

    if pair == {"full_body", "lower"}:
        return (
            False,
            None,
            "A full-body garment already covers the lower body; "
            "pairing with another lower-body garment is structurally "
            "invalid.",
        )

    return (
        False,
        None,
        f"Unhandled/unsupported slot combination: "
        f"{slot_a} + {slot_b}.",
    )


# --------------------------------------------------------------------
# CANDIDATE GENERATION
# --------------------------------------------------------------------

def generate_candidates(
    phase2_result,
    category_slot_map,
    include_layering=True,
):
    """
    Run Phase 3 over one Phase 2 result.

    Steps:

        1. Keep garments that survived Phase 2.
        2. Validate that every garment has a stable identifier.
        3. Assign every garment a structural slot.
        4. Generate valid single-item full-body outfits.
        5. Generate structurally valid pairs.
        6. Record invalid structural pairs.

    No compatibility/style scoring happens here.

    CRITICAL DATA CONTRACT:

        Phase 2 garment ID
            ↓
        Phase 3 candidate item ID
            ↓
        Phase 4 embedding lookup

    The ID is never replaced by "<unknown>".
    """

    if not isinstance(phase2_result, dict):
        raise TypeError(
            "phase2_result must be a dictionary."
        )

    request = phase2_result.get("request", {})

    if not isinstance(request, dict):
        request = {}

    survivors = get_survivors(phase2_result)

    entering = []
    excluded = []
    usable = []
    candidates = []
    rejected = []

    # ================================================================
    # PROCESS EVERY PHASE 2 SURVIVOR
    # ================================================================

    for index, garment in enumerate(survivors):

        # ------------------------------------------------------------
        # Validate garment object
        # ------------------------------------------------------------

        if not isinstance(garment, dict):

            print(
                f"WARNING: Phase 3 skipping invalid garment "
                f"at index {index}"
            )

            excluded.append(
                ExcludedGarment(
                    image=f"<invalid_garment_{index}>",
                    reason=(
                        "Phase 2 survivor is not a valid "
                        "garment dictionary."
                    ),
                )
            )

            continue

        # ------------------------------------------------------------
        # PRESERVE GARMENT ID
        # ------------------------------------------------------------

        garment_id = _get_garment_id(garment)

        if not garment_id:

            print()
            print("=" * 70)
            print("PHASE 3 DATA CONTRACT ERROR")
            print("=" * 70)
            print(
                f"Survivor index: {index}"
            )
            print(
                "Phase 2 garment has NO valid garment identifier."
            )
            print(
                "Phase 3 will NOT create <unknown>."
            )
            print()
            print("Phase 2 garment:")
            print(
                json.dumps(
                    garment,
                    indent=2,
                    ensure_ascii=False,
                )
            )
            print("=" * 70)

            excluded.append(
                ExcludedGarment(
                    image=f"<missing_id_{index}>",
                    reason=(
                        "Phase 2 survivor is missing its stable "
                        "garment identifier. This must be fixed "
                        "upstream in Phase 2."
                    ),
                )
            )

            continue

        # ------------------------------------------------------------
        # Structural slot
        # ------------------------------------------------------------

        slot, exclude_reason = _slot_for_garment(
            garment,
            category_slot_map,
        )

        details = garment.get("details", {})

        if not isinstance(details, dict):
            details = {}

        category = details.get("category", {})

        if not isinstance(category, dict):
            category = {}

        category_name = category.get("name")

        # ------------------------------------------------------------
        # Record Phase 3 input
        # ------------------------------------------------------------

        entering.append(
            SurvivingGarment(
                image=_display_name(garment_id),
                slot=slot,
                phase2_decision=garment.get("decision"),
                category_name=category_name,
            )
        )

        # ------------------------------------------------------------
        # No structural slot
        # ------------------------------------------------------------

        if slot is None:

            excluded.append(
                ExcludedGarment(
                    image=_display_name(garment_id),
                    reason=exclude_reason,
                )
            )

            continue

        # ------------------------------------------------------------
        # Keep garment
        # ------------------------------------------------------------

        usable.append(
            (
                garment,
                garment_id,
                slot,
            )
        )

    # ================================================================
    # DEBUG: PHASE 3 INPUT
    # ================================================================

    print()
    print("-" * 70)
    print("PHASE 3 — CANDIDATE GENERATION")
    print("-" * 70)

    print(
        f"Phase 2 survivors received: {len(survivors)}"
    )

    print(
        f"Usable garments: {len(usable)}"
    )

    print(
        f"Excluded garments: {len(excluded)}"
    )

    print()
    print("PHASE 3 INPUT GARMENTS")

    for index, (garment, garment_id, slot) in enumerate(
        usable,
        start=1,
    ):

        details = garment.get("details", {})

        if not isinstance(details, dict):
            details = {}

        category = details.get("category", {})

        if not isinstance(category, dict):
            category = {}

        category_name = category.get("name")

        print(
            f"  [{index}] "
            f"id={garment_id} | "
            f"category={category_name} | "
            f"slot={slot}"
        )

    # ================================================================
    # SINGLE FULL-BODY OUTFITS
    # ================================================================

    for garment, garment_id, slot in usable:

        if slot == "full_body":

            candidates.append(
                CandidateOutfit(
                    items=[garment_id],
                    structure="full_body",
                )
            )

    # ================================================================
    # PAIRWISE CANDIDATES
    # ================================================================

    for i in range(len(usable)):

        for j in range(i + 1, len(usable)):

            garment_a, id_a, slot_a = usable[i]
            garment_b, id_b, slot_b = usable[j]

            valid, structure, reason = _classify_pair(
                garment_a,
                garment_b,
                slot_a,
                slot_b,
                include_layering,
            )

            if valid:

                candidates.append(
                    CandidateOutfit(
                        items=[
                            id_a,
                            id_b,
                        ],
                        structure=structure,
                    )
                )

            else:

                rejected.append(
                    RejectedPair(
                        items=[
                            id_a,
                            id_b,
                        ],
                        reason=reason,
                    )
                )

    # ================================================================
    # FINAL DATA-CONTRACT VALIDATION
    # ================================================================

    validated_candidates = []

    for candidate in candidates:

        candidate_dict = candidate.to_dict()

        items = candidate_dict.get("items", [])

        invalid_ids = []

        for item in items:

            if item is None:
                invalid_ids.append(item)
                continue

            item_string = str(item).strip()

            if not item_string:
                invalid_ids.append(item)
                continue

            if item_string.lower() == "<unknown>":
                invalid_ids.append(item)

        if invalid_ids:

            print(
                "WARNING: Phase 3 rejected candidate with "
                f"invalid garment IDs: {items}"
            )

            continue

        validated_candidates.append(candidate)

    candidates = validated_candidates

    # ================================================================
    # DEBUG SUMMARY
    # ================================================================

    print()
    print("-" * 70)
    print("PHASE 3 SUMMARY")
    print("-" * 70)

    print(
        f"Phase 2 survivors received: {len(survivors)}"
    )

    print(
        f"Usable garments: {len(usable)}"
    )

    print(
        f"Excluded garments: {len(excluded)}"
    )

    print(
        f"Valid candidates: {len(candidates)}"
    )

    print(
        f"Rejected structural pairs: {len(rejected)}"
    )

    print()

    for index, candidate in enumerate(
        candidates[:30],
        start=1,
    ):

        candidate_dict = candidate.to_dict()

        print(
            f"CANDIDATE #{index}: "
            f"structure={candidate_dict.get('structure')} "
            f"items={candidate_dict.get('items')}"
        )

    # ================================================================
    # ABSOLUTE SAFETY CHECK
    # ================================================================

    for candidate in candidates:

        items = candidate.to_dict().get("items", [])

        for item in items:

            if str(item).strip().lower() == "<unknown>":

                raise RuntimeError(
                    "PHASE 3 DATA CONTRACT VIOLATION: "
                    "A candidate still contains <unknown>."
                )

    # ================================================================
    # RETURN RESULT
    # ================================================================

    return Phase3Result(
        occasion=request.get("occasion"),
        temperature_c=request.get("temperature_c"),
        rain=request.get("rain"),
        garments_entering_phase3=entering,
        excluded_no_structural_slot=excluded,
        candidate_outfits=candidates,
        rejected_structurally_invalid=rejected,
    )


# --------------------------------------------------------------------
# DIRECTORY RUNNER
# --------------------------------------------------------------------

def run_on_directory(
    phase2_dir,
    output_dir=None,
    include_layering=True,
    anno_dir=PROJECT_ROOT,
):
    """
    Run Phase 3 over every *.json file in a Phase 2 output directory.
    """

    phase2_dir = Path(phase2_dir)

    if not phase2_dir.exists():
        raise FileNotFoundError(
            f"Phase 2 directory does not exist: {phase2_dir}"
        )

    category_slot_map = load_category_slot_map(anno_dir)

    results = {}

    for json_file in sorted(phase2_dir.glob("*.json")):

        phase2_result = load_phase2_result(json_file)

        result = generate_candidates(
            phase2_result,
            category_slot_map,
            include_layering,
        )

        result_dict = result.to_dict()

        results[json_file.stem] = result_dict

        if output_dir is not None:

            out_path = (
                Path(output_dir)
                / f"{json_file.stem}.json"
            )

            out_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(out_path, "w", encoding="utf-8") as f:

                json.dump(
                    result_dict,
                    f,
                    indent=2,
                )

    return results


# --------------------------------------------------------------------
# COMMAND LINE
# --------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 3: generate structurally valid "
            "candidate outfits from Phase 2 output."
        )
    )

    parser.add_argument(
        "--phase2_dir",
        type=str,
        default="predictions_filtered",
        help="Directory containing Phase 2 *.json results.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="candidate_outfits",
        help="Directory to write Phase 3 *.json results.",
    )

    parser.add_argument(
        "--anno_dir",
        type=str,
        default=str(PROJECT_ROOT),
        help="Directory containing the Anno folder.",
    )

    parser.add_argument(
        "--no_layering",
        action="store_true",
        help=(
            "Disable full_body + upper layering candidates."
        ),
    )

    return parser.parse_args()


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------

def main():

    args = parse_args()

    results = run_on_directory(
        phase2_dir=args.phase2_dir,
        output_dir=args.output_dir,
        include_layering=not args.no_layering,
        anno_dir=args.anno_dir,
    )

    print(
        json.dumps(
            results,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()