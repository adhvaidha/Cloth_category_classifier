"""
Smart Wardrobe AI API
======================

FastAPI backend that powers the Smart Wardrobe frontend (frontend.html).

This module is intentionally a thin, well-tested HTTP layer on top of the
existing ML pipeline:

    inference.InferenceService   -> CLIP / ResNet / ViT / outfit generation
    filters.*                    -> occasion + weather filtering
    outfit_candidates.*          -> structural candidate generation

None of that logic is duplicated here. The API's job is to:

    - serve the wardrobe (list + images)
    - accept new clothing uploads and run them through the real
      classification pipeline (InferenceService.classify_image)
    - run the real outfit-generation pipeline (InferenceService.compose_outfits)
      and shape its output for the UI
    - persist small pieces of app state that the ML pipeline doesn't own:
      user-confirmed tags, favourites, and the outfit calendar/history.
      This uses flat JSON files under app_data/, which fits the project's
      existing convention of JSON-file storage (see predictions_filtered/,
      candidate_outfits/) rather than pulling in a database dependency.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

from inference import InferenceService
from utils.image_utils import ensure_rgb_image, load_image_from_bytes


# ================================================================
# FASTAPI APP
# ================================================================

app = FastAPI(
    title="Smart Wardrobe AI API",
    description="API for AI-powered outfit recommendations",
    version="2.0.0",
)


# ================================================================
# CORS
# ================================================================
#
# Allows the frontend (opened as a local file, or served from any
# origin/port) to talk to the API running on localhost.
#

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# CONFIGURATION / PATHS
# ================================================================

PROJECT_DIR = Path(__file__).resolve().parent
WARDROBE_DIR = PROJECT_DIR / "wardrobe"
APP_DATA_DIR = PROJECT_DIR / "app_data"

WARDROBE_DIR.mkdir(parents=True, exist_ok=True)
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = APP_DATA_DIR / "wardrobe_metadata.json"
FAVORITES_FILE = APP_DATA_DIR / "favorites.json"
CALENDAR_FILE = APP_DATA_DIR / "calendar.json"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".jfif",
    ".webp",
    ".bmp",
}

# Human-friendly closet categories, and the raw CLIP/DeepFashion category
# names (see inference.ATTRIBUTE_GROUPS / clip_categories) that map to them.
CATEGORY_GROUPS: Dict[str, List[str]] = {
    "tops": ["tee", "shirt", "blouse", "tank", "hoodie", "sweater", "cardigan", "shrug", "t-shirt"],
    "bottoms": ["jeans", "pants", "trousers", "shorts", "skirt"],
    "dresses": ["dress", "jumpsuit", "romper"],
    "jackets": ["jacket", "blazer", "coat", "bomber", "anorak", "parka", "peacoat"],
    "shoes": ["shoes"],
}


def guess_group(category_name: Optional[str], filename: str) -> str:
    """Map a garment to a coarse closet category.

    Prefers the AI-predicted category name; falls back to a filename-based
    guess (matching the previous frontend behaviour) only when no category
    is known yet, e.g. for images added before classification metadata
    existed.
    """

    haystacks = []
    if category_name:
        haystacks.append(category_name.lower())
    haystacks.append(filename.lower())

    for group, keywords in CATEGORY_GROUPS.items():
        for haystack in haystacks:
            if any(keyword in haystack for keyword in keywords):
                return group

    return "other"


# ================================================================
# SIMPLE JSON PERSISTENCE
# ================================================================
#
# One lock guards all app_data JSON files. Traffic on a personal wardrobe
# app is low enough that a single lock is simple, correct, and fast.

_store_lock = threading.Lock()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, data: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def load_metadata() -> Dict[str, Dict[str, Any]]:
    with _store_lock:
        return _read_json(METADATA_FILE, {})


def save_metadata(metadata: Dict[str, Dict[str, Any]]) -> None:
    with _store_lock:
        _write_json(METADATA_FILE, metadata)


def load_favorites() -> List[Dict[str, Any]]:
    with _store_lock:
        return _read_json(FAVORITES_FILE, [])


def save_favorites(favorites: List[Dict[str, Any]]) -> None:
    with _store_lock:
        _write_json(FAVORITES_FILE, favorites)


def load_calendar() -> Dict[str, Dict[str, Any]]:
    with _store_lock:
        return _read_json(CALENDAR_FILE, {})


def save_calendar(entries: Dict[str, Dict[str, Any]]) -> None:
    with _store_lock:
        _write_json(CALENDAR_FILE, entries)


# ================================================================
# INFERENCE SERVICE (singleton)
# ================================================================
#
# Loading CLIP / ResNet / ViT is expensive. The previous implementation
# constructed a brand new InferenceService on every single outfit-generation
# request, which reloaded every model from disk each time. We load it once,
# lazily, on first use.

_service: Optional[InferenceService] = None
_service_lock = threading.Lock()


def get_service() -> InferenceService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = InferenceService()
    return _service


# ================================================================
# WARDROBE HELPERS
# ================================================================

def iter_wardrobe_paths():
    return sorted(
        path
        for path in WARDROBE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_wardrobe():
    """
    Load all supported clothing images from the wardrobe folder as
    PIL images, for use by the inference pipeline.

        [{"id": "top.jfif", "image": PIL.Image}, ...]
    """

    image_paths = iter_wardrobe_paths()

    if not image_paths:
        raise RuntimeError(f"No supported images found in {WARDROBE_DIR}")

    items = []

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            items.append({"id": image_path.name, "image": image})
        except Exception as exc:
            print(f"Could not load {image_path.name}: {exc}")

    if not items:
        raise RuntimeError("No wardrobe images could be loaded.")

    return items


def build_wardrobe_item(image_path: Path, metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    entry = metadata.get(image_path.name, {})
    category = entry.get("category")

    return {
        "id": image_path.stem,
        "filename": image_path.name,
        "url": f"/wardrobe/{image_path.name}",
        "category": category or "unclassified",
        "group": guess_group(category, image_path.name),
        "tags": entry.get("tags", []),
        "name": entry.get("name") or image_path.stem.replace("_", " ").replace("-", " ").title(),
        "confidence": entry.get("confidence"),
        "added_at": entry.get("added_at"),
        "source": entry.get("source", "seed"),
    }


def get_wardrobe_items() -> List[Dict[str, Any]]:
    metadata = load_metadata()
    return [build_wardrobe_item(p, metadata) for p in iter_wardrobe_paths()]


def find_wardrobe_path(item_id: str) -> Path:
    """Resolve an item id (filename OR filename stem) to a real file path."""

    requested_name = Path(item_id).name
    exact = WARDROBE_DIR / requested_name
    if exact.exists() and exact.is_file() and exact.suffix.lower() in SUPPORTED_EXTENSIONS:
        return exact

    stem = Path(requested_name).stem
    for path in iter_wardrobe_paths():
        if path.stem == stem:
            return path

    raise HTTPException(status_code=404, detail=f"Wardrobe item not found: {item_id}")


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return value or "item"


# ================================================================
# REQUEST MODELS
# ================================================================

class OutfitRequest(BaseModel):
    occasion: str
    temperature_c: float
    rain: bool = False


class WardrobeItemUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None


class FavoriteCreate(BaseModel):
    rank: Optional[int] = None
    item_ids: List[str]
    items: Optional[List[Dict[str, Any]]] = None
    compatibility_score: Optional[float] = None
    explanation: Optional[str] = None
    occasion: Optional[str] = None
    temperature_c: Optional[float] = None
    rain: Optional[bool] = None


class CalendarEntryCreate(BaseModel):
    date: str
    item_ids: List[str]
    items: Optional[List[Dict[str, Any]]] = None
    occasion: Optional[str] = None
    compatibility_score: Optional[float] = None
    notes: Optional[str] = None
    worn: bool = False


class CalendarWornUpdate(BaseModel):
    worn: bool


# ================================================================
# HOME / HEALTH CHECK
# ================================================================

@app.get("/")
def home():
    return {
        "status": "running",
        "service": "Smart Wardrobe AI API",
        "version": "2.0.0",
    }


@app.get("/api/status")
def get_status():
    """Model + wardrobe status, used by the dashboard and error states."""

    try:
        service = get_service()
        model_status = service.get_model_status()
    except Exception as exc:  # pragma: no cover - defensive
        model_status = {
            "models_loaded": False,
            "can_recommend": False,
            "errors": [str(exc)],
        }

    wardrobe_count = len(iter_wardrobe_paths())

    return {
        "api": "ok",
        "wardrobe_items": wardrobe_count,
        "model_status": model_status,
    }


# ================================================================
# SERVE WARDROBE IMAGES
# ================================================================

@app.get("/wardrobe/{filename}")
def get_wardrobe_image(filename: str):
    """
    Serve a wardrobe image using either the exact filename
    (``/wardrobe/top.jfif``) or its stem (``/wardrobe/top``), the latter
    being how the AI recommender refers to garments.
    """

    requested_name = Path(filename).name
    exact_path = WARDROBE_DIR / requested_name

    if (
        exact_path.exists()
        and exact_path.is_file()
        and exact_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ):
        return FileResponse(exact_path)

    requested_stem = Path(requested_name).stem
    for image_path in WARDROBE_DIR.iterdir():
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if image_path.stem == requested_stem:
            return FileResponse(image_path)

    raise HTTPException(status_code=404, detail=f"Wardrobe image not found: {filename}")


# ================================================================
# WARDROBE — LIST / STATS
# ================================================================

@app.get("/api/wardrobe")
def get_wardrobe(category: Optional[str] = None, search: Optional[str] = None):
    if not WARDROBE_DIR.exists():
        raise HTTPException(status_code=404, detail="Wardrobe directory not found.")

    items = get_wardrobe_items()

    if category and category != "all":
        items = [item for item in items if item["group"] == category]

    if search:
        needle = search.strip().lower()
        items = [
            item
            for item in items
            if needle in item["name"].lower()
            or needle in item["filename"].lower()
            or needle in item["category"].lower()
            or any(needle in tag.lower() for tag in item["tags"])
        ]

    return {"count": len(items), "items": items}


@app.get("/api/wardrobe/stats")
def get_wardrobe_stats():
    items = get_wardrobe_items()

    by_group: Dict[str, int] = {}
    for item in items:
        by_group[item["group"]] = by_group.get(item["group"], 0) + 1

    recent = sorted(
        items,
        key=lambda item: item.get("added_at") or "",
        reverse=True,
    )[:8]

    favorites_count = len(load_favorites())
    calendar_entries = load_calendar()
    worn_count = sum(1 for entry in calendar_entries.values() if entry.get("worn"))

    return {
        "total_items": len(items),
        "by_group": by_group,
        "recent_items": recent,
        "favorites_count": favorites_count,
        "planned_outfits": len(calendar_entries),
        "worn_outfits": worn_count,
    }


# ================================================================
# WARDROBE — ADD / EDIT / DELETE
# ================================================================

@app.post("/api/wardrobe/items")
async def add_wardrobe_item(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
):
    """
    Upload a new clothing photo.

    The image is saved into the wardrobe folder and run through the real
    CLIP classification pipeline (InferenceService.classify_image) so the
    UI can suggest a category + attributes. The user can then confirm or
    edit those tags (see PATCH /api/wardrobe/items/{item_id}) — nothing
    here is faked or hardcoded.
    """

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image = load_image_from_bytes(raw_bytes, convert_to_rgb=True, raise_on_error=False)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not read that file as an image. Please upload a JPG, PNG, WEBP or BMP photo.",
        )
    image = ensure_rgb_image(image)

    # Build a unique, filesystem-safe filename that keeps a supported
    # extension so it is picked up by iter_wardrobe_paths().
    original_stem = Path(file.filename or "item").stem
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        suffix = ".jpg"

    unique_suffix = uuid.uuid4().hex[:8]
    final_name = f"{slugify(original_stem)}-{unique_suffix}{suffix}"
    destination = WARDROBE_DIR / final_name

    save_format = "JPEG" if suffix in (".jpg", ".jpeg", ".jfif") else None
    if save_format:
        image.save(destination, format=save_format)
    else:
        image.save(destination)

    # Best-effort automatic classification via the existing CLIP pipeline.
    classification: Dict[str, Any] = {}
    classification_error: Optional[str] = None

    try:
        service = get_service()
        if service.clip_loaded:
            classification = service.classify_image(str(destination))
        else:
            classification_error = (
                "The AI classifier isn't loaded, so this item was saved "
                "without an automatic category. You can set it manually below."
            )
    except Exception as exc:
        classification_error = f"Automatic classification failed: {exc}"

    predicted_category = classification.get("category_name") or classification.get("category")
    predicted_confidence = classification.get("category_confidence")

    parsed_tags: List[str] = []
    if tags:
        parsed_tags = [t.strip() for t in tags.split(",") if t.strip()]
    elif classification.get("attributes"):
        # Seed tags from the top color/style/pattern predictions so the
        # user has something concrete to confirm or edit, rather than
        # starting from nothing.
        for group_name in ("color", "style", "pattern"):
            values = classification["attributes"].get(group_name) or []
            if values:
                parsed_tags.append(values[0]["name"])

    metadata = load_metadata()
    metadata[final_name] = {
        "name": (name or original_stem).strip() or original_stem,
        "category": (category or predicted_category or "unclassified"),
        "tags": parsed_tags,
        "confidence": predicted_confidence,
        "added_at": datetime.utcnow().isoformat() + "Z",
        "source": "uploaded",
        "clip_predictions": classification.get("predictions", []),
    }
    save_metadata(metadata)

    item = build_wardrobe_item(destination, metadata)

    return {
        "item": item,
        "classification": classification or None,
        "warning": classification_error,
    }


@app.patch("/api/wardrobe/items/{item_id}")
def update_wardrobe_item(item_id: str, update: WardrobeItemUpdate):
    path = find_wardrobe_path(item_id)
    metadata = load_metadata()
    entry = metadata.get(path.name, {})

    if update.name is not None:
        entry["name"] = update.name
    if update.category is not None:
        entry["category"] = update.category
    if update.tags is not None:
        entry["tags"] = update.tags

    metadata[path.name] = entry
    save_metadata(metadata)

    return {"item": build_wardrobe_item(path, metadata)}


@app.delete("/api/wardrobe/items/{item_id}")
def delete_wardrobe_item(item_id: str):
    path = find_wardrobe_path(item_id)

    try:
        path.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete file: {exc}")

    metadata = load_metadata()
    metadata.pop(path.name, None)
    save_metadata(metadata)

    # Clean up any favourites / calendar entries referencing this item so
    # deleted garments don't leave dangling references in the UI.
    favorites = load_favorites()
    changed_favorites = [
        fav for fav in favorites if path.stem not in fav.get("item_ids", [])
    ]
    if len(changed_favorites) != len(favorites):
        save_favorites(changed_favorites)

    calendar_entries = load_calendar()
    changed = False
    for key, entry in list(calendar_entries.items()):
        if path.stem in entry.get("item_ids", []):
            calendar_entries.pop(key)
            changed = True
    if changed:
        save_calendar(calendar_entries)

    return {"deleted": True, "id": path.stem}


# ================================================================
# GENERATE OUTFITS
# ================================================================

def _build_explanation(outfit: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Turn the backend's score breakdown into a plain-English reason."""

    parts: List[str] = []

    occasion_score = outfit.get("occasion_score", 0.0) or 0.0
    weather_score = outfit.get("weather_score", 0.0) or 0.0
    vit_score = outfit.get("vit_score", 0.0) or 0.0
    garment_score = outfit.get("garment_compatibility_score", 0.0) or 0.0

    occasion = context.get("occasion", "this occasion")
    if occasion_score >= 0.75:
        parts.append(f"strongly matches {occasion}")
    elif occasion_score >= 0.45:
        parts.append(f"suits {occasion} reasonably well")
    else:
        parts.append(f"was one of the few options left for {occasion}")

    if context.get("rain"):
        if weather_score >= 0.6:
            parts.append("holds up in the rain")
    else:
        temp = context.get("temperature_c")
        if weather_score >= 0.75 and temp is not None:
            parts.append(f"fits {temp:.0f}°C weather well")
        elif weather_score < 0.4 and temp is not None:
            parts.append(f"is a compromise for {temp:.0f}°C")

    if vit_score >= 0.65:
        parts.append("the pieces are learned to look good together")

    if garment_score >= 1.0:
        parts.append("covers top and bottom cleanly")

    if not parts:
        return "Balanced pick across occasion, weather and style compatibility."

    sentence = "; ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."


@app.post("/api/outfits/generate")
def generate_outfits(request: OutfitRequest):
    occasion = request.occasion.strip().lower()

    print()
    print("=" * 70)
    print("API REQUEST — GENERATE OUTFITS")
    print("=" * 70)
    print(f"Occasion: {occasion}")
    print(f"Temperature: {request.temperature_c}°C")
    print(f"Rain: {request.rain}")

    try:
        items = load_wardrobe()
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    print(f"Wardrobe items loaded: {len(items)}")

    service = get_service()

    context = {
        "occasion": occasion,
        "temperature_c": request.temperature_c,
        "rain": request.rain,
    }

    results = service.compose_outfits(items, context)

    model_status = service.get_model_status()
    base_response = {
        "occasion": occasion,
        "temperature_c": request.temperature_c,
        "rain": request.rain,
        "wardrobe_items": len(items),
        "ranking_mode": model_status.get("ranking_mode"),
    }

    # ------------------------------------------------------------
    # InferenceService reports fatal errors (e.g. models not loaded)
    # as `[{"error": ..., "details": [...]}]` rather than raising, so
    # detect that shape explicitly instead of silently treating it as
    # a zero-length or malformed outfit list.
    # ------------------------------------------------------------

    if (
        isinstance(results, list)
        and len(results) == 1
        and isinstance(results[0], dict)
        and "error" in results[0]
    ):
        return {
            **base_response,
            "outfits": [],
            "error": results[0]["error"],
            "details": results[0].get("details", []),
        }

    if not results:
        return {**base_response, "outfits": []}

    metadata = load_metadata()

    outfits = []
    for result in results[:5]:
        if not isinstance(result, dict):
            continue

        item_ids = result.get("item_ids", [])
        raw_items = result.get("items", [])

        enriched_items = []
        for raw_item in raw_items:
            item_id = str(raw_item.get("id"))
            try:
                path = find_wardrobe_path(item_id)
                url = f"/wardrobe/{path.name}"
                filename = path.name
            except HTTPException:
                url = f"/wardrobe/{item_id}"
                filename = item_id

            meta = metadata.get(filename, {})
            enriched_items.append(
                {
                    "id": Path(item_id).stem,
                    "filename": filename,
                    "url": url,
                    "name": meta.get("name"),
                    "category": raw_item.get("category", meta.get("category", "unknown")),
                    "category_type": raw_item.get("category_type"),
                    "occasion_score": raw_item.get("occasion_score"),
                    "weather_score": raw_item.get("weather_score"),
                }
            )

        outfits.append(
            {
                "rank": result.get("rank"),
                "item_ids": [Path(str(i)).stem for i in item_ids],
                "items": enriched_items,
                "structure": result.get("structure"),
                "compatibility_score": result.get("compatibility_score"),
                "occasion_score": result.get("occasion_score"),
                "weather_score": result.get("weather_score"),
                "vit_score": result.get("vit_score"),
                "garment_compatibility_score": result.get("garment_compatibility_score"),
                "ranking_mode": result.get("ranking_mode", "ai"),
                "explanation": _build_explanation(result, context),
            }
        )

    response = {**base_response, "outfits": outfits}

    print()
    print("API RESPONSE")
    print("-" * 70)
    for outfit in outfits:
        score = outfit.get("compatibility_score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        print(f"#{outfit.get('rank')} score={score_str} {outfit.get('item_ids', [])}")
    print("=" * 70)
    print()

    return response


# ================================================================
# FAVOURITES
# ================================================================

@app.get("/api/favorites")
def list_favorites():
    return {"favorites": load_favorites()}


@app.post("/api/favorites")
def add_favorite(favorite: FavoriteCreate):
    favorites = load_favorites()
    entry = favorite.dict()
    entry["id"] = uuid.uuid4().hex
    entry["saved_at"] = datetime.utcnow().isoformat() + "Z"
    favorites.append(entry)
    save_favorites(favorites)
    return {"favorite": entry}


@app.delete("/api/favorites/{favorite_id}")
def delete_favorite(favorite_id: str):
    favorites = load_favorites()
    remaining = [f for f in favorites if f.get("id") != favorite_id]
    if len(remaining) == len(favorites):
        raise HTTPException(status_code=404, detail="Favorite not found.")
    save_favorites(remaining)
    return {"deleted": True}


# ================================================================
# CALENDAR / SCHEDULE  (also powers outfit history)
# ================================================================

@app.get("/api/calendar")
def list_calendar():
    return {"entries": load_calendar()}


@app.get("/api/calendar/{date_key}")
def get_calendar_entry(date_key: str):
    entries = load_calendar()
    entry = entries.get(date_key)
    if entry is None:
        raise HTTPException(status_code=404, detail="No outfit planned for that date.")
    return {"entry": entry}


@app.post("/api/calendar")
def upsert_calendar_entry(entry: CalendarEntryCreate):
    try:
        datetime.strptime(entry.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format.")

    entries = load_calendar()
    payload = entry.dict()
    date_key = payload.pop("date")
    payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
    entries[date_key] = payload
    save_calendar(entries)
    return {"entry": payload, "date": date_key}


@app.patch("/api/calendar/{date_key}")
def mark_calendar_entry_worn(date_key: str, update: CalendarWornUpdate):
    entries = load_calendar()
    entry = entries.get(date_key)
    if entry is None:
        raise HTTPException(status_code=404, detail="No outfit planned for that date.")
    entry["worn"] = update.worn
    entry["updated_at"] = datetime.utcnow().isoformat() + "Z"
    entries[date_key] = entry
    save_calendar(entries)
    return {"entry": entry, "date": date_key}


@app.delete("/api/calendar/{date_key}")
def delete_calendar_entry(date_key: str):
    entries = load_calendar()
    if date_key not in entries:
        raise HTTPException(status_code=404, detail="No outfit planned for that date.")
    entries.pop(date_key)
    save_calendar(entries)
    return {"deleted": True, "date": date_key}


@app.get("/api/history")
def get_history():
    """Outfits marked as worn, most recent first."""

    entries = load_calendar()
    worn = [
        {"date": date_key, **entry}
        for date_key, entry in entries.items()
        if entry.get("worn")
    ]
    worn.sort(key=lambda e: e["date"], reverse=True)
    return {"history": worn}
