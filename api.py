from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

from inference import InferenceService


# ================================================================
# FASTAPI APP
# ================================================================

app = FastAPI(
    title="Smart Wardrobe AI API",
    description="API for AI-powered outfit recommendations",
    version="1.0.0",
)


# ================================================================
# CORS
# ================================================================
#
# Allows your local frontend.html file to communicate
# with the FastAPI server running on localhost.
#

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# REQUEST MODEL
# ================================================================

class OutfitRequest(BaseModel):
    occasion: str
    temperature_c: float
    rain: bool


# ================================================================
# CONFIGURATION
# ================================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".jfif",
    ".webp",
    ".bmp",
}


# Always locate the wardrobe folder relative to api.py.
PROJECT_DIR = Path(__file__).resolve().parent

WARDROBE_DIR = PROJECT_DIR / "wardrobe"


# ================================================================
# LOAD WARDROBE
# ================================================================

def load_wardrobe():
    """
    Load all supported clothing images from the wardrobe folder.

    Each wardrobe item is returned as:

        {
            "id": "top.jfif",
            "image": PIL.Image
        }
    """

    if not WARDROBE_DIR.exists():
        raise RuntimeError(
            f"Wardrobe directory not found: {WARDROBE_DIR}"
        )

    image_paths = sorted(
        path
        for path in WARDROBE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    )

    if not image_paths:
        raise RuntimeError(
            f"No supported images found in {WARDROBE_DIR}"
        )

    items = []

    for image_path in image_paths:

        try:

            image = (
                Image.open(image_path)
                .convert("RGB")
            )

            items.append(
                {
                    "id": image_path.name,
                    "image": image,
                }
            )

        except Exception as exc:

            print(
                f"Could not load "
                f"{image_path.name}: {exc}"
            )

    if not items:
        raise RuntimeError(
            "No wardrobe images could be loaded."
        )

    return items


# ================================================================
# HOME / HEALTH CHECK
# ================================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "service": "Smart Wardrobe AI API",
        "version": "1.0.0",
    }


# ================================================================
# SERVE WARDROBE IMAGES
# ================================================================

@app.get("/wardrobe/{filename}")
def get_wardrobe_image(filename: str):
    """
    Serve a wardrobe image using either:

        /wardrobe/top.jfif

    or simply:

        /wardrobe/top

    The second form is useful because the AI recommender
    may return IDs such as "top", "skirt2", etc.
    """

    # Remove any directory components from the requested name.
    # This keeps the request inside the wardrobe directory.

    requested_name = Path(filename).name

    # ------------------------------------------------------------
    # First, try the exact filename.
    # ------------------------------------------------------------

    exact_path = WARDROBE_DIR / requested_name

    if (
        exact_path.exists()
        and exact_path.is_file()
        and exact_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ):
        return FileResponse(exact_path)

    # ------------------------------------------------------------
    # If the exact filename was not found, treat the request
    # as an item ID and match it against the filename stem.
    #
    # Example:
    #
    #     top
    #
    # matches:
    #
    #     top.jfif
    #
    # ------------------------------------------------------------

    requested_stem = Path(requested_name).stem

    for image_path in WARDROBE_DIR.iterdir():

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if image_path.stem == requested_stem:
            return FileResponse(image_path)

    # ------------------------------------------------------------
    # Nothing matched.
    # ------------------------------------------------------------

    raise HTTPException(
        status_code=404,
        detail=f"Wardrobe image not found: {filename}",
    )

@app.get("/api/wardrobe")
def get_wardrobe():

    if not WARDROBE_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail="Wardrobe directory not found.",
        )

    images = []

    for image_path in sorted(WARDROBE_DIR.iterdir()):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        images.append(
            {
                "id": image_path.stem,
                "filename": image_path.name,
                "url": (
                    "http://127.0.0.1:8000/wardrobe/"
                    + image_path.name
                ),
            }
        )

    return {
        "count": len(images),
        "items": images,
    }
# ================================================================
# GENERATE OUTFITS
# ================================================================

@app.post("/api/outfits/generate")
def generate_outfits(request: OutfitRequest):

    print()
    print("=" * 70)
    print("API REQUEST — GENERATE OUTFITS")
    print("=" * 70)

    print(
        f"Occasion: {request.occasion}"
    )

    print(
        f"Temperature: {request.temperature_c}°C"
    )

    print(
        f"Rain: {request.rain}"
    )

    # ------------------------------------------------------------
    # Load wardrobe
    # ------------------------------------------------------------

    items = load_wardrobe()

    print(
        f"Wardrobe items loaded: {len(items)}"
    )

    # ------------------------------------------------------------
    # Create inference service
    # ------------------------------------------------------------

    service = InferenceService()

    # ------------------------------------------------------------
    # Build context
    # ------------------------------------------------------------

    context = {
        "occasion": request.occasion,
        "temperature_c": request.temperature_c,
        "rain": request.rain,
    }

    # ------------------------------------------------------------
    # Run complete AI recommendation pipeline
    # ------------------------------------------------------------

    results = service.compose_outfits(
        items,
        context,
    )

    # ------------------------------------------------------------
    # No recommendations
    # ------------------------------------------------------------

    if not results:

        return {
            "occasion": request.occasion,
            "temperature_c": request.temperature_c,
            "rain": request.rain,
            "wardrobe_items": len(items),
            "outfits": [],
        }

    # ------------------------------------------------------------
    # Keep only the top 5 outfits.
    #
    # We intentionally expose only:
    #
    #     rank
    #     item_ids
    #
    # The internal CLIP / ViT / scoring information stays
    # inside the backend.
    # ------------------------------------------------------------

    outfits = []

    for result in results[:5]:

        if not isinstance(result, dict):
            continue

        outfits.append(
            {
                "rank": result.get("rank"),
                "item_ids": result.get(
                    "item_ids",
                    [],
                ),
            }
        )

    # ------------------------------------------------------------
    # Final response
    # ------------------------------------------------------------

    response = {
        "occasion": request.occasion,
        "temperature_c": request.temperature_c,
        "rain": request.rain,
        "wardrobe_items": len(items),
        "outfits": outfits,
    }

    print()
    print("API RESPONSE")
    print("-" * 70)

    for outfit in outfits:

        print(
            f"#{outfit.get('rank')} "
            f"{outfit.get('item_ids', [])}"
        )

    print("=" * 70)
    print()

    return response