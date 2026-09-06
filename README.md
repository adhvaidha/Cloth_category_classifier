# Muse — an AI-powered Smart Wardrobe app

Upload your clothes, and let a CLIP + ResNet + ViT pipeline classify them,
generate outfits for an occasion and the weather, and help you plan and
track what you actually wear.

This repository contains:

- **The ML pipeline** (`inference.py`, `filters/`, `outfit_candidates/`,
  `garment_tagger/`) — CLIP-based garment classification, occasion/weather
  filtering, and a ViT-based outfit-compatibility ranker. **Unchanged** from
  the original research code.
- **`api.py`** — a FastAPI backend that exposes that pipeline over HTTP, plus
  small JSON-file-backed persistence for favorites, the outfit calendar, and
  wardrobe metadata (names/tags/categories).
- **`frontend.html` + `app.js`** — a single-page app (no build step) that
  talks to the API: dashboard, closet, add-item, AI stylist, calendar,
  favorites, and history.

---

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Model checkpoints (optional, but required for AI features)

The outfit ranker and automatic classification need two trained checkpoints
that are **not included in this repository** (they're large binary files):

```
models/exports/resnet_item_embedder_best.pth
models/exports/vit_outfit_model_best.pth
```

Train them with `train.py`, or place your own checkpoints at those paths.
CLIP itself (`open_clip_torch`) downloads its pretrained weights
automatically on first run.

**You don't need the checkpoints to explore the app.** Without them:

- The closet, search/filters, calendar, favorites, and history all work
  normally.
- Adding a photo still saves it to your wardrobe; it just won't get an
  automatic category suggestion (you can still tag it manually).
- The AI Stylist and dashboard will show a clear "AI models not ready"
  banner instead of silently failing or faking a result.

## 2. Run the backend

```bash
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

This serves the API at `http://127.0.0.1:8000` and reads/writes images in
`wardrobe/`. On first request it lazily loads CLIP/ResNet/ViT once and keeps
them in memory (previous versions of this API reloaded every model on every
single outfit request — that's fixed).

Check it's healthy:

```bash
curl http://127.0.0.1:8000/api/status
```

## 3. Run the frontend

The frontend is a static file — no build step, no npm install.

```bash
# from the project root, in a second terminal
python -m http.server 5500
```

Then open **http://127.0.0.1:5500/frontend.html** in your browser.

(You can also just double-click `frontend.html` to open it directly as a
`file://` URL — the backend's CORS is open, so both work. Serving it over
HTTP is only slightly friendlier for browser caching/geolocation.)

If your backend isn't at `http://127.0.0.1:8000`, click the status pill in
the top-right of the app and type the correct address — it's saved in your
browser for next time.

---

## Features

**Home** — live wardrobe stats (item count, favorites, planned/worn
outfits), a category breakdown, your most recently added pieces, and quick
links to the Stylist, Add Item, and Calendar.

**Closet** — every wardrobe image, searchable and filterable by category
(tops/bottoms/dresses/jackets/shoes/other), sortable by recency, name, or
category. Click an item to rename it, edit its category/tags, or delete it.

**Add Clothing** — drag-and-drop or click to upload a photo. It's saved to
`wardrobe/` immediately and run through `InferenceService.classify_image`
for a suggested category, confidence score, and color/style/pattern tags,
which you can edit before confirming.

**AI Stylist** — pick an occasion (casual, college, travel, office, formal,
wedding, party, date, workout — matching `filters/occasion.py` exactly),
set a temperature and whether it's raining (or pull live weather via
Open-Meteo + your browser's geolocation — no hardcoded weather), and
generate outfits. Each result shows the actual wardrobe photos, an overall
compatibility score, a breakdown across occasion/weather/style/coverage,
and a plain-English explanation. Save a look to Favorites, plan it on a
date, or cycle through the other ranked options with "Try another."

**Calendar** — a real month grid. Click a day to plan an outfit (from your
saved favorites or by hand-picking wardrobe items), mark it as worn, edit,
or remove it. Planned vs. worn days are visually distinct, and everything
is persisted server-side in `app_data/calendar.json`, so a page refresh
never loses it.

**Favorites** — every outfit you've saved from the Stylist, with the option
to plan it for a date or remove it.

**History** — every outfit you've marked "worn" on the calendar, most
recent first.

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | Backend + model health |
| GET | `/api/wardrobe?category=&search=` | List wardrobe items |
| GET | `/api/wardrobe/stats` | Dashboard stats |
| GET | `/wardrobe/{filename_or_id}` | Serve a wardrobe image |
| POST | `/api/wardrobe/items` | Upload + classify a new item (multipart) |
| PATCH | `/api/wardrobe/items/{id}` | Edit name/category/tags |
| DELETE | `/api/wardrobe/items/{id}` | Delete an item |
| POST | `/api/outfits/generate` | Run the outfit pipeline (`occasion`, `temperature_c`, `rain`) |
| GET/POST | `/api/favorites` | List / save a favorite outfit |
| DELETE | `/api/favorites/{id}` | Remove a favorite |
| GET/POST | `/api/calendar` | List all planned outfits / plan one for a date |
| GET/PATCH/DELETE | `/api/calendar/{YYYY-MM-DD}` | Get, mark worn, or remove a specific day |
| GET | `/api/history` | Outfits marked as worn |

## Project layout

```
api.py                  FastAPI backend (HTTP layer only)
inference.py             CLIP / ResNet / ViT pipeline (unchanged)
filters/                 Occasion + weather filtering (unchanged)
outfit_candidates/       Structural outfit-candidate generation (unchanged)
garment_tagger/          DeepFashion attribute tagging (unchanged)
utils/                   Image loading helpers
frontend.html            Single-page app shell + styles
app.js                   Frontend application logic
wardrobe/                Your clothing photos
app_data/                Generated at runtime: metadata, favorites, calendar (gitignored)
requirements.txt         Python dependencies
```
