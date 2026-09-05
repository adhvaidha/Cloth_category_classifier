"""
CLIP garment classification + visual attribute extraction.

Purpose:
    Use OpenCLIP to identify:
        1. Garment category
        2. Dominant color
        3. Formality
        4. Style / aesthetic
        5. Pattern
        6. Material / fabric appearance
        7. Season suitability
        8. Category-specific garment attributes

Important:
    - No filename-based categorization.
    - No DeepFashion involvement.
    - Each attribute group is classified independently.
    - Probabilities are meaningful only within their own group.
    - Attribute predictions are visual evidence, NOT hard truth.
    - Results are saved for later integration into Phase 2.

Usage from project root:

    python tests/test_clip_classification.py

Output:

    debug/clip_classification.json
"""

from pathlib import Path
import json
import sys

import torch
from PIL import Image
import open_clip


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WARDROBE_DIR = PROJECT_ROOT / "wardrobe"
DEBUG_DIR = PROJECT_ROOT / "debug"
OUTPUT_FILE = DEBUG_DIR / "clip_classification.json"


# ============================================================================
# DEVICE
# ============================================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# ============================================================================
# MODEL
# ============================================================================

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


# ============================================================================
# CATEGORY DEFINITIONS
# ============================================================================

CATEGORIES = [
    "t-shirt",
    "shirt",
    "blouse",
    "tank top",
    "hoodie",
    "sweater",
    "cardigan",
    "shrug",
    "jacket",
    "blazer",
    "coat",
    "jeans",
    "trousers",
    "pants",
    "shorts",
    "skirt",
    "dress",
    "jumpsuit",
    "romper",
    "shoes",
]


# ============================================================================
# ATTRIBUTE DEFINITIONS
# ============================================================================

# These are intentionally independent groups.
#
# DO NOT combine all of these into one giant softmax.
#
# For example:
#
#     Floral = 85%
#
# means:
#
#     "Floral is the strongest pattern candidate."
#
# It does NOT mean:
#
#     "There is an 85% probability that the entire garment is floral."
#
# CLIP is being used as a zero-shot visual classifier.

ATTRIBUTE_GROUPS = {

    # ------------------------------------------------------------------------
    # COLOR
    # ------------------------------------------------------------------------

    "color": [
        "black",
        "white",
        "grey",
        "beige",
        "brown",
        "cream",
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "navy blue",
        "purple",
        "pink",
        "maroon",
        "teal",
        "multicolor",
    ],

    # ------------------------------------------------------------------------
    # FORMALITY
    # ------------------------------------------------------------------------

    "formality": [
        "very casual",
        "casual",
        "smart casual",
        "business casual",
        "formal",
        "very formal",
    ],

    # ------------------------------------------------------------------------
    # STYLE / AESTHETIC
    # ------------------------------------------------------------------------

    "style": [
        "casual",
        "minimalist",
        "classic",
        "elegant",
        "romantic",
        "sporty",
        "streetwear",
        "preppy",
        "bohemian",
        "vintage",
        "edgy",
        "festive",
    ],

    # ------------------------------------------------------------------------
    # PATTERN
    # ------------------------------------------------------------------------

    "pattern": [
        "solid color",
        "floral pattern",
        "striped pattern",
        "checked pattern",
        "plaid pattern",
        "polka dot pattern",
        "geometric pattern",
        "graphic print",
        "animal print",
        "abstract print",
        "textured pattern",
    ],

    # ------------------------------------------------------------------------
    # MATERIAL / FABRIC APPEARANCE
    # ------------------------------------------------------------------------

    "material": [
        "cotton",
        "linen",
        "denim",
        "wool",
        "knit",
        "leather",
        "suede",
        "silk",
        "satin",
        "polyester",
        "chiffon",
        "corduroy",
        "fleece",
    ],

    # ------------------------------------------------------------------------
    # SEASON
    # ------------------------------------------------------------------------

    "season": [
        "summer clothing",
        "spring clothing",
        "autumn clothing",
        "winter clothing",
        "all-season clothing",
    ],

    # ------------------------------------------------------------------------
    # FIT / SILHOUETTE
    # ------------------------------------------------------------------------

    "fit": [
        "slim fit",
        "fitted",
        "regular fit",
        "relaxed fit",
        "oversized",
        "loose fit",
        "flowy",
        "structured",
    ],

    # ------------------------------------------------------------------------
    # SLEEVE
    # ------------------------------------------------------------------------

    "sleeve": [
        "sleeveless",
        "short sleeves",
        "elbow length sleeves",
        "three quarter sleeves",
        "long sleeves",
    ],

    # ------------------------------------------------------------------------
    # LENGTH
    # ------------------------------------------------------------------------

    "length": [
        "cropped",
        "waist length",
        "hip length",
        "knee length",
        "midi length",
        "ankle length",
        "full length",
    ],
}


# ============================================================================
# CATEGORY-SPECIFIC ATTRIBUTE GROUPS
# ============================================================================

# These groups are only evaluated when the detected category makes them
# relevant. This prevents us from asking CLIP questions such as:
#
#     "Is this a sundress?"
#
# when the image is actually a pair of trousers.

CATEGORY_SPECIFIC_GROUPS = {

    # ------------------------------------------------------------------------
    # DRESSES
    # ------------------------------------------------------------------------

    "dress": {
        "dress_type": [
            "casual day dress",
            "sundress",
            "shirt dress",
            "office dress",
            "business dress",
            "cocktail dress",
            "party dress",
            "evening dress",
            "formal dress",
            "maxi dress",
        ],
    },

    # ------------------------------------------------------------------------
    # TOPS
    # ------------------------------------------------------------------------

    "shirt": {
        "top_style": [
            "casual shirt",
            "formal shirt",
            "button-up shirt",
            "oversized shirt",
            "office shirt",
            "linen shirt",
            "printed shirt",
        ],
    },

    "blouse": {
        "top_style": [
            "casual blouse",
            "office blouse",
            "elegant blouse",
            "romantic blouse",
            "flowy blouse",
            "printed blouse",
        ],
    },

    "t-shirt": {
        "top_style": [
            "basic t-shirt",
            "graphic t-shirt",
            "oversized t-shirt",
            "fitted t-shirt",
            "sporty t-shirt",
            "casual t-shirt",
        ],
    },

    "hoodie": {
        "top_style": [
            "basic hoodie",
            "oversized hoodie",
            "sporty hoodie",
            "streetwear hoodie",
        ],
    },

    "sweater": {
        "top_style": [
            "casual sweater",
            "knit sweater",
            "oversized sweater",
            "fitted sweater",
            "formal sweater",
        ],
    },

    # ------------------------------------------------------------------------
    # BOTTOMS
    # ------------------------------------------------------------------------

    "jeans": {
        "bottom_style": [
            "skinny jeans",
            "straight leg jeans",
            "wide leg jeans",
            "baggy jeans",
            "bootcut jeans",
            "ripped jeans",
            "dark formal-looking jeans",
        ],
    },

    "trousers": {
        "bottom_style": [
            "formal trousers",
            "office trousers",
            "wide leg trousers",
            "straight leg trousers",
            "tailored trousers",
            "relaxed trousers",
        ],
    },

    "pants": {
        "bottom_style": [
            "formal pants",
            "casual pants",
            "wide leg pants",
            "straight leg pants",
            "cargo pants",
            "relaxed pants",
            "tailored pants",
        ],
    },

    "shorts": {
        "bottom_style": [
            "casual shorts",
            "denim shorts",
            "sport shorts",
            "formal-looking shorts",
            "relaxed shorts",
        ],
    },

    "skirt": {
        "bottom_style": [
            "mini skirt",
            "midi skirt",
            "maxi skirt",
            "pencil skirt",
            "pleated skirt",
            "flowy skirt",
            "casual skirt",
            "formal skirt",
        ],
    },

    # ------------------------------------------------------------------------
    # OUTERWEAR
    # ------------------------------------------------------------------------

    "jacket": {
        "outerwear_style": [
            "casual jacket",
            "denim jacket",
            "leather jacket",
            "bomber jacket",
            "sporty jacket",
            "formal jacket",
            "lightweight jacket",
        ],
    },

    "blazer": {
        "outerwear_style": [
            "formal blazer",
            "office blazer",
            "casual blazer",
            "oversized blazer",
            "tailored blazer",
        ],
    },

    "coat": {
        "outerwear_style": [
            "formal coat",
            "casual coat",
            "winter coat",
            "trench coat",
            "long coat",
            "structured coat",
        ],
    },

    # ------------------------------------------------------------------------
    # FOOTWEAR
    # ------------------------------------------------------------------------

    "shoes": {
        "shoe_type": [
            "sneakers",
            "running shoes",
            "boots",
            "formal shoes",
            "loafers",
            "sandals",
            "heels",
            "flats",
            "casual shoes",
        ],
    },
}


# ============================================================================
# GENERAL CONFIGURATION
# ============================================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".jfif",
    ".webp",
    ".bmp",
}

TOP_K_CATEGORY = 5
TOP_K_ATTRIBUTE = 5

# CLIP's logits are scaled by this value before softmax.
# This matches the behavior of the original implementation.
LOGIT_SCALE = 100.0


# ============================================================================
# HELPERS
# ============================================================================

def find_images():
    """Return all supported wardrobe images."""

    if not WARDROBE_DIR.exists():
        raise FileNotFoundError(
            f"Wardrobe directory does not exist:\n{WARDROBE_DIR}"
        )

    images = sorted(
        path
        for path in WARDROBE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    return images


def build_category_prompts():
    """
    Build prompts for garment category classification.

    We explicitly describe the image as a clothing item so CLIP compares
    garment concepts instead of generic text concepts.
    """

    return [
        f"a photo of a {category} clothing item"
        for category in CATEGORIES
    ]


def build_attribute_prompts(group_name, values):
    """
    Build prompts for one attribute group.
    """

    prompts = []

    for value in values:
        prompts.append(
            f"a photo of a clothing item that is {value}"
        )

    return prompts


def softmax_predictions(similarity, labels, top_k):
    """
    Convert similarity scores into a probability-like distribution
    within ONE classification group.

    Returns:
        [
            {
                "name": ...,
                "probability": ...,
                "similarity": ...
            }
        ]
    """

    probabilities = torch.softmax(
        similarity * LOGIT_SCALE,
        dim=0,
    )

    k = min(top_k, len(labels))

    values, indices = torch.topk(
        probabilities,
        k=k,
    )

    results = []

    for value, index in zip(values, indices):

        idx = index.item()

        results.append(
            {
                "name": labels[idx],
                "probability": float(value.item()),
                "similarity": float(
                    similarity[idx].item()
                ),
            }
        )

    return results


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_clip():

    print("=" * 70)
    print("LOADING CLIP")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Pretrained weights: {PRETRAINED}")
    print()

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device=DEVICE,
    )

    tokenizer = open_clip.get_tokenizer(
        MODEL_NAME
    )

    model.eval()

    print("CLIP loaded successfully")
    print()

    return model, preprocess, tokenizer


# ============================================================================
# TEXT EMBEDDING CACHE
# ============================================================================

def build_text_embeddings(model, tokenizer):
    """
    Pre-compute all text embeddings once.

    This is considerably more efficient than rebuilding the text embeddings
    for every wardrobe image.
    """

    print("=" * 70)
    print("BUILDING CLIP TEXT EMBEDDINGS")
    print("=" * 70)

    text_embeddings = {}

    # ------------------------------------------------------------------------
    # CATEGORY
    # ------------------------------------------------------------------------

    category_prompts = build_category_prompts()

    category_tokens = tokenizer(
        category_prompts
    ).to(DEVICE)

    with torch.inference_mode():

        category_features = model.encode_text(
            category_tokens
        )

        category_features /= category_features.norm(
            dim=-1,
            keepdim=True,
        )

    text_embeddings["category"] = {
        "labels": CATEGORIES,
        "features": category_features,
    }

    print(
        f"Category prompts: {len(category_prompts)}"
    )

    # ------------------------------------------------------------------------
    # GENERAL ATTRIBUTES
    # ------------------------------------------------------------------------

    for group_name, values in ATTRIBUTE_GROUPS.items():

        prompts = build_attribute_prompts(
            group_name,
            values,
        )

        tokens = tokenizer(
            prompts
        ).to(DEVICE)

        with torch.inference_mode():

            features = model.encode_text(
                tokens
            )

            features /= features.norm(
                dim=-1,
                keepdim=True,
            )

        text_embeddings[group_name] = {
            "labels": values,
            "features": features,
        }

        print(
            f"{group_name:<15}: {len(values)} prompts"
        )

    # ------------------------------------------------------------------------
    # CATEGORY-SPECIFIC ATTRIBUTES
    # ------------------------------------------------------------------------

    for category, groups in CATEGORY_SPECIFIC_GROUPS.items():

        text_embeddings[category] = {}

        for group_name, values in groups.items():

            prompts = build_attribute_prompts(
                group_name,
                values,
            )

            tokens = tokenizer(
                prompts
            ).to(DEVICE)

            with torch.inference_mode():

                features = model.encode_text(
                    tokens
                )

                features /= features.norm(
                    dim=-1,
                    keepdim=True,
                )

            text_embeddings[category][group_name] = {
                "labels": values,
                "features": features,
            }

            print(
                f"{category}.{group_name:<15}: "
                f"{len(values)} prompts"
            )

    print()
    print("Text embeddings ready.")

    return text_embeddings


# ============================================================================
# CATEGORY CLASSIFICATION
# ============================================================================

def classify_category(
    image_features,
    text_embeddings,
):
    """
    Classify the garment category.
    """

    group = text_embeddings["category"]

    similarity = (
        image_features @ group["features"].T
    ).squeeze(0)

    predictions = softmax_predictions(
        similarity,
        group["labels"],
        TOP_K_CATEGORY,
    )

    return predictions


# ============================================================================
# ATTRIBUTE CLASSIFICATION
# ============================================================================

def classify_attribute_group(
    image_features,
    group_data,
):
    """
    Classify one independent attribute group.
    """

    similarity = (
        image_features @ group_data["features"].T
    ).squeeze(0)

    return softmax_predictions(
        similarity,
        group_data["labels"],
        TOP_K_ATTRIBUTE,
    )


# ============================================================================
# COMPLETE IMAGE ANALYSIS
# ============================================================================

def analyze_image(
    image_path,
    model,
    preprocess,
    text_embeddings,
):
    """
    Run complete CLIP analysis on one wardrobe image.
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = (
        preprocess(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.inference_mode():

        image_features = model.encode_image(
            image_tensor
        )

        image_features /= image_features.norm(
            dim=-1,
            keepdim=True,
        )

    # ------------------------------------------------------------------------
    # CATEGORY
    # ------------------------------------------------------------------------

    category_predictions = classify_category(
        image_features,
        text_embeddings,
    )

    top_category = category_predictions[0]["name"]

    # ------------------------------------------------------------------------
    # GENERAL ATTRIBUTES
    # ------------------------------------------------------------------------

    attributes = {}

    for group_name in ATTRIBUTE_GROUPS:

        attributes[group_name] = classify_attribute_group(
            image_features,
            text_embeddings[group_name],
        )

    # ------------------------------------------------------------------------
    # CATEGORY-SPECIFIC ATTRIBUTES
    # ------------------------------------------------------------------------

    category_specific = {}

    if top_category in CATEGORY_SPECIFIC_GROUPS:

        category_groups = (
            CATEGORY_SPECIFIC_GROUPS[
                top_category
            ]
        )

        for group_name in category_groups:

            category_specific[group_name] = (
                classify_attribute_group(
                    image_features,
                    text_embeddings[top_category][group_name],
                )
            )

    # ------------------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------------------

    return {
        "image": image_path.name,

        "category": category_predictions,

        "top_category": {
            "name": top_category,
            "probability": category_predictions[0][
                "probability"
            ],
        },

        "attributes": attributes,

        "category_specific_attributes": (
            category_specific
        ),
    }


# ============================================================================
# DEBUG PRINTING
# ============================================================================

def print_prediction(
    image_path,
    result,
):
    """
    Print a human-readable prediction for one garment.
    """

    print()
    print("=" * 70)
    print(f"IMAGE: {image_path.name}")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # CATEGORY
    # ------------------------------------------------------------------------

    print()
    print("CATEGORY")
    print("-" * 70)

    for rank, prediction in enumerate(
        result["category"],
        start=1,
    ):

        print(
            f"{rank}. "
            f"{prediction['name']:<20} "
            f"{prediction['probability']:.2%}"
        )

    top = result["top_category"]

    print()
    print(
        f"CLIP CATEGORY: "
        f"{top['name']} "
        f"({top['probability']:.2%})"
    )

    # ------------------------------------------------------------------------
    # GENERAL ATTRIBUTES
    # ------------------------------------------------------------------------

    print()
    print("GENERAL ATTRIBUTES")
    print("-" * 70)

    for group_name, predictions in result[
        "attributes"
    ].items():

        print()
        print(f"[{group_name.upper()}]")

        for rank, prediction in enumerate(
            predictions,
            start=1,
        ):

            print(
                f"  {rank}. "
                f"{prediction['name']:<25} "
                f"{prediction['probability']:.2%}"
            )

    # ------------------------------------------------------------------------
    # CATEGORY-SPECIFIC ATTRIBUTES
    # ------------------------------------------------------------------------

    category_specific = result[
        "category_specific_attributes"
    ]

    if category_specific:

        print()
        print("CATEGORY-SPECIFIC ATTRIBUTES")
        print("-" * 70)

        for group_name, predictions in (
            category_specific.items()
        ):

            print()
            print(f"[{group_name.upper()}]")

            for rank, prediction in enumerate(
                predictions,
                start=1,
            ):

                print(
                    f"  {rank}. "
                    f"{prediction['name']:<30} "
                    f"{prediction['probability']:.2%}"
                )


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(results):

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "model": {
            "architecture": MODEL_NAME,
            "pretrained": PRETRAINED,
            "device": DEVICE,
        },

        "classification": {
            "category_top_k": TOP_K_CATEGORY,
            "attribute_top_k": TOP_K_ATTRIBUTE,
            "logit_scale": LOGIT_SCALE,
        },

        "categories": CATEGORIES,

        "attribute_groups": {
            name: values
            for name, values in ATTRIBUTE_GROUPS.items()
        },

        "category_specific_groups": (
            CATEGORY_SPECIFIC_GROUPS
        ),

        "image_count": len(results),

        "results": results,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(OUTPUT_FILE)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 70)
    print("CLIP GARMENT + ATTRIBUTE ANALYSIS")
    print("=" * 70)

    print()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Wardrobe     : {WARDROBE_DIR}")
    print(f"Output       : {OUTPUT_FILE}")
    print()

    # ------------------------------------------------------------------------
    # FIND IMAGES
    # ------------------------------------------------------------------------

    image_paths = find_images()

    print(
        f"Found {len(image_paths)} wardrobe images"
    )

    if not image_paths:

        print()
        print("No supported images found.")

        print()
        print("Supported extensions:")
        print(
            ", ".join(
                sorted(
                    SUPPORTED_EXTENSIONS
                )
            )
        )

        return 1

    # ------------------------------------------------------------------------
    # SHOW CONFIGURATION
    # ------------------------------------------------------------------------

    print()
    print("CATEGORY LABELS")
    print("-" * 70)

    for category in CATEGORIES:
        print(f"  • {category}")

    print()
    print("ATTRIBUTE GROUPS")
    print("-" * 70)

    for group_name, values in (
        ATTRIBUTE_GROUPS.items()
    ):

        print(
            f"  • {group_name}: "
            f"{len(values)} labels"
        )

    print()
    print("CATEGORY-SPECIFIC GROUPS")
    print("-" * 70)

    for category, groups in (
        CATEGORY_SPECIFIC_GROUPS.items()
    ):

        print(
            f"  • {category}: "
            f"{', '.join(groups.keys())}"
        )

    # ------------------------------------------------------------------------
    # LOAD CLIP
    # ------------------------------------------------------------------------

    try:

        model, preprocess, tokenizer = load_clip()

    except Exception as exc:

        print()
        print("Failed to load CLIP:")
        print(exc)

        import traceback

        traceback.print_exc()

        return 1

    # ------------------------------------------------------------------------
    # BUILD TEXT EMBEDDINGS ONCE
    # ------------------------------------------------------------------------

    try:

        text_embeddings = build_text_embeddings(
            model,
            tokenizer,
        )

    except Exception as exc:

        print()
        print(
            "Failed to build CLIP text embeddings:"
        )
        print(exc)

        import traceback

        traceback.print_exc()

        return 1

    # ------------------------------------------------------------------------
    # ANALYZE WARDROBE
    # ------------------------------------------------------------------------

    results = []

    print()
    print("=" * 70)
    print("ANALYZING WARDROBE")
    print("=" * 70)

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(image_paths)}] "
            f"Processing {image_path.name}"
        )

        try:

            result = analyze_image(
                image_path=image_path,
                model=model,
                preprocess=preprocess,
                text_embeddings=text_embeddings,
            )

            print_prediction(
                image_path,
                result,
            )

            results.append(
                {
                    "image": image_path.name,

                    "path": str(
                        image_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                    "predictions": result,
                }
            )

        except Exception as exc:

            print(
                f"Failed: {image_path.name}"
            )

            print(
                f"Error: {exc}"
            )

            results.append(
                {
                    "image": image_path.name,

                    "path": str(
                        image_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                    "error": str(exc),
                }
            )

    # ------------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------------

    save_results(results)

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    successful = [
        result
        for result in results
        if "predictions" in result
    ]

    failed = [
        result
        for result in results
        if "error" in result
    ]

    print(
        f"Images processed : {len(results)}"
    )

    print(
        f"Successful       : {len(successful)}"
    )

    print(
        f"Failed           : {len(failed)}"
    )

    # ------------------------------------------------------------------------
    # CATEGORY DISTRIBUTION
    # ------------------------------------------------------------------------

    if successful:

        category_counts = {}

        for result in successful:

            category = (
                result["predictions"]
                ["top_category"]
                ["name"]
            )

            category_counts[category] = (
                category_counts.get(
                    category,
                    0,
                )
                + 1
            )

        print()
        print("TOP-PREDICTION DISTRIBUTION")
        print("-" * 70)

        for category, count in sorted(
            category_counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):

            print(
                f"{category:<20} {count}"
            )

    # ------------------------------------------------------------------------
    # FAILED IMAGES
    # ------------------------------------------------------------------------

    if failed:

        print()
        print("FAILED IMAGES")
        print("-" * 70)

        for result in failed:

            print(
                f"{result['image']}: "
                f"{result['error']}"
            )

    # ------------------------------------------------------------------------
    # COMPLETE
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("CLIP ATTRIBUTE ANALYSIS COMPLETE")
    print("=" * 70)

    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())