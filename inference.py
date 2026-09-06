import json
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import open_clip

from utils.transforms import build_inference_transform
from models.resnet_embedder import ResNetItemEmbedder
from models.vit_outfit import OutfitCompatibilityModel
from utils.tag_system import TagProcessor
from utils.image_utils import ensure_rgb_image, validate_image_format



# ---------------------------------------------------------------------------
# CLIP visual attribute vocabularies
# Reused from the validated standalone attribute test.
# ---------------------------------------------------------------------------

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

TOP_K_ATTRIBUTE = 5
LOGIT_SCALE = 100.0

# ---------------------------------------------------------------------------
# Existing project phases
# ---------------------------------------------------------------------------


TOP_K_CATEGORY = 5

# Final context-aware outfit ranking weights.
OCCASION_WEIGHT = 0.40
VIT_WEIGHT = 0.35
WEATHER_WEIGHT = 0.15
GARMENT_COMPATIBILITY_WEIGHT = 0.10


from filters.filter_engine import filter_wardrobe

from outfit_candidates.generator import (
    generate_candidates,
    load_category_slot_map,
)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


# ---------------------------------------------------------------------------
# Inference Service
# ---------------------------------------------------------------------------

class InferenceService:
    """
    End-to-end inference pipeline.

    Pipeline:

        Image
          ↓
        CLIP zero-shot garment classification
          ↓
        Phase 2 filtering
          ↓
        Phase 3 structural candidate generation
          ↓
        ResNet item embeddings
          ↓
        ViT compatibility scoring
          ↓
        ranked outfits

    CLIP is used as the Phase 1 garment classifier.
    Filename-based category detection is intentionally disabled.
    DeepFashion is kept out of the critical path.
    """

    def __init__(self) -> None:
        self.device = _get_device()

        self.transform = build_inference_transform()

        self.embed_dim = int(os.getenv("EMBED_DIM", "512"))

        self.resnet_version = "resnet_v1"
        self.vit_version = "vit_v1"

        # ---------------------------------------------------------------
        # Model status
        # ---------------------------------------------------------------

        self.models_loaded = False
        self.model_errors: List[str] = []

        # ---------------------------------------------------------------
        # Tag processing
        # ---------------------------------------------------------------

        self.tag_processor = TagProcessor()

        # ---------------------------------------------------------------
        # CLIP Phase 1 classifier
        # ---------------------------------------------------------------

        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.clip_categories: List[str] = []
        self.clip_loaded = False

        self._load_clip()

        print("ℹ️ Filename-based category detection disabled")

        # ---------------------------------------------------------------
        # Load Phase 4 models
        # ---------------------------------------------------------------

        self.resnet, self.resnet_loaded = self._load_resnet()
        self.vit, self.vit_loaded = self._load_vit()

        if self.resnet_loaded:
            self.resnet = self.resnet.to(self.device).eval()

        if self.vit_loaded:
            self.vit = self.vit.to(self.device).eval()

        # Disable gradients
        for model in (self.resnet, self.vit):
            if model is not None:
                for parameter in model.parameters():
                    parameter.requires_grad_(False)

        # ---------------------------------------------------------------
        # DeepFashion is intentionally NOT loaded for production inference.
        # It remains available elsewhere in the project for comparison.
        # ---------------------------------------------------------------

        self.deepfashion_model = None
        self.deepfashion_cat_names = None
        self.deepfashion_attr_names = None
        self.deepfashion_attr_types = None
        self.deepfashion_loaded = False

        # ---------------------------------------------------------------
        # Overall status
        # ---------------------------------------------------------------

        self.models_loaded = (
            self.clip_loaded
            and self.resnet_loaded
            and self.vit_loaded
        )

        if not self.models_loaded:
            self.model_errors = []

            if not self.clip_loaded:
                self.model_errors.append(
                    "CLIP: classifier not loaded"
                )

            if not self.resnet_loaded:
                self.model_errors.append(
                    "ResNet: trained embedding checkpoint not loaded"
                )

            if not self.vit_loaded:
                self.model_errors.append(
                    "ViT: trained compatibility checkpoint not loaded"
                )

    # ===================================================================
    # Phase 4 model loading
    # ===================================================================

    @staticmethod
    def _extract_state_dict(checkpoint):
        """Extract a model state_dict from common checkpoint formats."""
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
            return checkpoint
        return checkpoint

    @staticmethod
    def _clean_state_dict(state_dict):
        """Remove common DataParallel / DDP prefixes."""
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            cleaned[key] = value
        return cleaned

    def _load_resnet(self) -> Tuple[nn.Module, bool]:
        """Load the trained ResNet item embedding checkpoint."""
        checkpoint_path = Path(
            os.getenv(
                "RESNET_CHECKPOINT",
                "models/exports/resnet_item_embedder_best.pth",
            )
        )

        print(
            f"📁 Loading local ResNet checkpoint: {checkpoint_path}"
        )

        try:
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"ResNet checkpoint not found: {checkpoint_path}"
                )

            model = ResNetItemEmbedder(
                embedding_dim=self.embed_dim,
                backbone=os.getenv(
                    "RESNET_BACKBONE",
                    "resnet50",
                ),
                pretrained=False,
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
            )
            state_dict = self._clean_state_dict(
                self._extract_state_dict(checkpoint)
            )

            missing, unexpected = model.load_state_dict(
                state_dict,
                strict=False,
            )

            if missing:
                print("⚠️ ResNet missing keys:")
                for key in missing[:20]:
                    print(f"   - {key}")

            if unexpected:
                print("⚠️ ResNet unexpected keys:")
                for key in unexpected[:20]:
                    print(f"   - {key}")

            model = model.to(self.device).eval()

            print(
                "✅ ResNet local checkpoint loaded successfully"
            )
            return model, True

        except Exception as exc:
            print(
                f"❌ Failed to load ResNet checkpoint: {exc}"
            )
            import traceback
            traceback.print_exc()
            return None, False

    def _load_vit(self) -> Tuple[nn.Module, bool]:
        """Load the trained ViT outfit compatibility checkpoint."""
        checkpoint_path = Path(
            os.getenv(
                "VIT_CHECKPOINT",
                "models/exports/vit_outfit_model_best.pth",
            )
        )

        print(
            f"📁 Loading local ViT checkpoint: {checkpoint_path}"
        )

        try:
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"ViT checkpoint not found: {checkpoint_path}"
                )

            model = OutfitCompatibilityModel(
                embedding_dim=self.embed_dim,
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
            )
            state_dict = self._clean_state_dict(
                self._extract_state_dict(checkpoint)
            )

            missing, unexpected = model.load_state_dict(
                state_dict,
                strict=False,
            )

            if missing:
                print("⚠️ ViT missing keys:")
                for key in missing[:20]:
                    print(f"   - {key}")

            if unexpected:
                print("⚠️ ViT unexpected keys:")
                for key in unexpected[:20]:
                    print(f"   - {key}")

            model = model.to(self.device).eval()

            print(
                "✅ ViT local checkpoint loaded successfully"
            )
            return model, True

        except Exception as exc:
            print(
                f"❌ Failed to load ViT checkpoint: {exc}"
            )
            import traceback
            traceback.print_exc()
            return None, False

    @torch.inference_mode()
    def embed_images(
        self,
        images: List[Image.Image],
    ) -> np.ndarray:
        """Generate one normalized ResNet embedding per image."""
        if not self.resnet_loaded or self.resnet is None:
            raise RuntimeError(
                "ResNet model is not loaded."
            )

        if not images:
            return np.empty(
                (0, self.embed_dim),
                dtype=np.float32,
            )

        tensors = []
        for image in images:
            image = ensure_rgb_image(image)
            tensors.append(self.transform(image))

        batch = torch.stack(
            tensors,
            dim=0,
        ).to(self.device)

        embeddings = self.resnet(batch)
        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        return (
            embeddings.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    # ===================================================================
    # CLIP Phase 1
    # ===================================================================

    def _load_clip(self) -> None:
        """Load OpenCLIP and precompute category/attribute text features."""

        self.clip_categories = [
            "Tee", "Shirt", "Blouse", "Tank", "Hoodie",
            "Sweater", "Cardigan", "Shrug", "Jacket", "Blazer",
            "Coat", "Jeans", "Pants", "Shorts", "Skirt", "Dress",
            "Jumpsuit", "Romper", "Shoes",
        ]

        model_name = os.getenv("CLIP_MODEL", "ViT-B-32")
        pretrained = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")

        try:
            print(f"📁 Loading CLIP classifier: {model_name} / {pretrained}")

            # IMPORTANT: create_model_and_transforms returns only
            # (model, train_transform, validation_transform).
            self.clip_model, _, self.clip_preprocess = (
                open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained, device=self.device
                )
            )
            self.clip_tokenizer = open_clip.get_tokenizer(model_name)
            self.clip_model = self.clip_model.to(self.device).eval()

            for parameter in self.clip_model.parameters():
                parameter.requires_grad_(False)

            self.clip_loaded = True
            print("✅ CLIP classifier loaded successfully")
            print(f"   Categories: {len(self.clip_categories)}")

            self._build_clip_text_embeddings()

        except Exception as exc:
            self.clip_model = None
            self.clip_preprocess = None
            self.clip_tokenizer = None
            self.clip_loaded = False
            print(f"❌ Failed to load CLIP classifier: {exc}")
            import traceback
            traceback.print_exc()

    @torch.inference_mode()
    def _build_clip_text_embeddings(self) -> None:
        """Encode category and visual-attribute prompts once."""
        self.clip_text_embeddings = {}

        def encode_group(labels, prompt_builder):
            prompts = [prompt_builder(label) for label in labels]
            tokens = self.clip_tokenizer(prompts).to(self.device)
            features = self.clip_model.encode_text(tokens)
            return nn.functional.normalize(features, dim=-1)

        self.clip_text_embeddings["category"] = {
            "labels": self.clip_categories,
            "features": encode_group(
                self.clip_categories,
                lambda label: f"a photo of a {label.lower()} clothing item",
            ),
        }

        for group_name, labels in ATTRIBUTE_GROUPS.items():
            self.clip_text_embeddings[group_name] = {
                "labels": labels,
                "features": encode_group(
                    labels,
                    lambda label: f"a photo of a clothing item that is {label}",
                ),
            }

        for category, groups in CATEGORY_SPECIFIC_GROUPS.items():
            self.clip_text_embeddings[category] = {}
            for group_name, labels in groups.items():
                self.clip_text_embeddings[category][group_name] = {
                    "labels": labels,
                    "features": encode_group(
                        labels,
                        lambda label: f"a photo of a clothing item that is {label}",
                    ),
                }

        print(
            f"🧬 Prepared {len(ATTRIBUTE_GROUPS)} general CLIP attribute groups "
            f"+ category-specific groups"
        )

    @staticmethod
    def _softmax_predictions(similarity, labels, top_k=TOP_K_ATTRIBUTE):
        probabilities = torch.softmax(similarity * LOGIT_SCALE, dim=0)
        k = min(top_k, len(labels))
        values, indices = torch.topk(probabilities, k=k)
        return [
            {
                "name": labels[index.item()],
                "category": labels[index.item()],
                "category_name": labels[index.item()],
                "probability": float(value.item()),
                "confidence": float(value.item()),
                "similarity": float(similarity[index].item()),
            }
            for value, index in zip(values, indices)
        ]

    def _predict_clip_group(self, image_features, group_data):
        similarity = (image_features @ group_data["features"].T).squeeze(0)
        return self._softmax_predictions(
            similarity, group_data["labels"], TOP_K_ATTRIBUTE
        )

    @torch.inference_mode()
    def _predict_clip_attributes(self, image_features, category):
        attributes = {}

        for group_name in ATTRIBUTE_GROUPS:
            attributes[group_name] = self._predict_clip_group(
                image_features, self.clip_text_embeddings[group_name]
            )

        category_key = category.lower().strip()
        # The category vocabulary uses "Tee" while the standalone test uses
        # "t-shirt". Map aliases without using filenames.
        category_aliases = {
            "tee": "t-shirt",
            "tank": "tank top",
        }
        category_key = category_aliases.get(category_key, category_key)

        category_groups = CATEGORY_SPECIFIC_GROUPS.get(category_key, {})
        category_specific = {}
        if category_groups and category_key in self.clip_text_embeddings:
            for group_name in category_groups:
                category_specific[group_name] = self._predict_clip_group(
                    image_features,
                    self.clip_text_embeddings[category_key][group_name],
                )

        attributes.update(category_specific)
        return attributes, category_specific

    def _clip_prompts(self) -> List[str]:
        """
        Prompts used for zero-shot garment classification.

        Keep the wording stable during evaluation so comparisons between
        runs are meaningful.
        """
        return [
            f"a photo of a {category.lower()} clothing item"
            for category in self.clip_categories
        ]

    @torch.inference_mode()
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """Run CLIP category + complete visual attribute analysis.

        The image path is used only to load the image. It is NEVER inspected
        to determine the garment category or attributes.
        """
        if not self.clip_loaded or self.clip_model is None:
            raise RuntimeError("CLIP classifier is not loaded.")

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        image_features = self.clip_model.encode_image(image_tensor)
        image_features = nn.functional.normalize(image_features, dim=-1)

        category_group = self.clip_text_embeddings["category"]
        similarities = (image_features @ category_group["features"].T).squeeze(0)
        predictions = self._softmax_predictions(
            similarities, category_group["labels"], TOP_K_CATEGORY
        )
        top = predictions[0]

        predicted_category = top.get(
            "category",
            top.get("name", top.get("category_name")),
        )

        if not predicted_category:
            raise RuntimeError(
                "CLIP returned a category prediction without a category name."
            )

        attributes, category_specific = self._predict_clip_attributes(
            image_features, predicted_category
        )

        print(
            f"   ✅ CLIP prediction: {predicted_category} "
            f"({top['probability']:.2%})"
        )
        print("   🧬 CLIP visual attributes:")
        for group_name in ("color", "formality", "style", "pattern", "material", "season"):
            values = attributes.get(group_name, [])
            if values:
                summary = ", ".join(
                    f"{item['name']}={item['probability']:.1%}"
                    for item in values[:3]
                )
                print(f"      {group_name}: {summary}")

        for group_name, values in category_specific.items():
            summary = ", ".join(
                f"{item['name']}={item['probability']:.1%}"
                for item in values[:3]
            )
            print(f"      {group_name}: {summary}")

        return {
            "category": predicted_category,
            "category_name": predicted_category,
            "category_confidence": top["probability"],
            "confidence": top["probability"],
            "predictions": [
                {
                    "category": item["category"] if "category" in item else item["name"],
                    "category_name": item["category"] if "category" in item else item["name"],
                    "probability": item["probability"],
                    "confidence": item["probability"],
                    "similarity": item["similarity"],
                }
                for item in predictions
            ],
            "attributes": attributes,
            "category_specific_attributes": category_specific,
            "classifier": "openclip",
            "model": os.getenv("CLIP_MODEL", "ViT-B-32"),
            "pretrained": os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k"),
        }

    # ===================================================================
    # DEBUG / DIAGNOSTICS
    # ===================================================================

    def _debug_clip_prediction(self, filename: str, prediction: Dict[str, Any]) -> None:
        print("\n" + "." * 70)
        print(f"🔬 DEBUG CLIP: {filename}")
        print("." * 70)
        if not isinstance(prediction, dict):
            print(f"⚠️ Unexpected prediction type: {type(prediction)}")
            print(repr(prediction))
            return
        print("Top-level keys:", list(prediction.keys()))
        for key, value in prediction.items():
            k = str(key).lower()
            if any(x in k for x in ("categor", "class", "prob", "score", "label", "similar")):
                print(f"{key}: {value}")
        print("FULL PREDICTION:")
        print(json.dumps(prediction, indent=2, default=str))

    def _debug_phase2_result(self, phase2_result: Any) -> None:
        print("\n" + "." * 70)
        print("🔬 DEBUG PHASE 2 — ALL GARMENT DECISIONS")
        print("." * 70)
        try:
            data = phase2_result.to_dict()
        except Exception:
            data = phase2_result if isinstance(phase2_result, dict) else {}
        print("Phase 2 type:", type(phase2_result))
        print("Request:", data.get("request", {}))
        garments = data.get("garments", [])
        print(f"Garments: {len(garments)}")
        for i, garment in enumerate(garments, 1):
            print(f"\n[{i}]")
            print(json.dumps(garment, indent=2, default=str))

    def _debug_phase3_result(self, phase3_result: Any, candidates: List[Any], survivors: List[Any]) -> None:
        print("\n" + "." * 70)
        print("🔬 DEBUG PHASE 3 — SURVIVORS AND CANDIDATES")
        print("." * 70)
        print("Phase 3 type:", type(phase3_result))
        print(f"Survivors: {len(survivors)}")
        for i, survivor in enumerate(survivors, 1):
            print(f"\nSURVIVOR {i}:")
            print(json.dumps(survivor, indent=2, default=str) if isinstance(survivor, dict) else repr(survivor))
        print(f"\nCandidates: {len(candidates)}")
        for i, candidate in enumerate(candidates, 1):
            if isinstance(candidate, dict):
                items = candidate.get("items", [])
                structure = candidate.get("structure", "unknown")
            else:
                items = list(getattr(candidate, "items", []))
                structure = getattr(candidate, "structure", "unknown")
            suspicious = len(items) == 1 or len({str(x).lower() for x in items}) < len(items)
            print(f"{'⚠️ SUSPICIOUS' if suspicious else 'CANDIDATE'} #{i}: structure={structure}, items={items}")

    def _debug_ranking(self, ranked: List[Dict[str, Any]]) -> None:
        print("\n" + "." * 70)
        print("🔬 DEBUG PHASE 4 — RANKING")
        print("." * 70)
        if not ranked:
            print("No ranked candidates.")
            return
        scores = [float(r["compatibility_score"]) for r in ranked]
        print(f"Score range: min={min(scores):.8f}, max={max(scores):.8f}, spread={max(scores)-min(scores):.8f}")
        for result in ranked:
            print(f"RANK #{result['rank']:03d} score={result['compatibility_score']:.8f} structure={result.get('structure')} items={result.get('item_ids')}")

    # ===================================================================
    # Phase 1 (CLIP) + Phase 2
    # ===================================================================

    def _run_phase1_and_phase2(
        self,
        items: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Run CLIP on every input garment, then run the existing
        Phase 2 filter.

        A temporary predictions directory is used because the existing
        Phase 2 API expects predictions_dir.
        """

        occasion = context.get(
            "occasion",
            "casual",
        )

        temperature_c = float(
            context.get(
                "temperature_c",
                context.get("temperature", 25),
            )
        )

        rain = bool(
            context.get(
                "rain",
                False,
            )
        )

        temp_dir = Path(
            tempfile.mkdtemp(
                prefix="cloth_phase1_"
            )
        )

        print(
            f"📁 Temporary Phase 1 predictions: {temp_dir}"
        )

        # ---------------------------------------------------------------
        # Run CLIP on each image
        # ---------------------------------------------------------------

        for index, item in enumerate(items):
            image = item.get("image")

            if image is None:
                raise ValueError(
                    f"Item {index} does not contain an image."
                )

            image_id = item.get("id")

            if not image_id:
                image_id = f"item_{index}.jpg"

            # Only use the provided ID as a temporary file name.
            # It is NEVER used for categorization.
            filename = Path(str(image_id)).name

            if not filename:
                filename = f"item_{index}.jpg"

            image_path = temp_dir / filename

            ensure_rgb_image(image).save(
                image_path,
                format="JPEG",
            )

            print(
                f"🔍 Phase 1: CLIP → {filename}"
            )

            prediction = self.classify_image(
                str(image_path)
            )

            self._debug_clip_prediction(filename, prediction)

            prediction_path = (
                temp_dir / f"{Path(filename).stem}.json"
            )

            with open(
                prediction_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    prediction,
                    file,
                    indent=2,
                )

        # ---------------------------------------------------------------
        # Existing Phase 2 implementation
        # ---------------------------------------------------------------

        print(
            "🔍 Phase 2: filtering CLIP predictions..."
        )

        phase2_result = filter_wardrobe(
            temp_dir,
            occasion,
            temperature_c,
            rain,
        )

        print("✅ Phase 2 filtering complete")

        self._debug_phase2_result(phase2_result)

        return temp_dir, phase2_result

    # ===================================================================
    # Phase 4 — rank Phase 3 candidates
    # ===================================================================

    @torch.inference_mode()
    def _rank_phase3_candidates(
        self,
        candidates: List[Any],
        survivors: List[Any],
        image_lookup: Dict[str, Image.Image],
        metadata_lookup: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        use_vit: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Context-aware Phase 4 ranking.

        Final score (use_vit=True, the ResNet+ViT checkpoints are loaded):
            40% occasion suitability
            35% learned ViT outfit compatibility
            15% weather suitability
            10% structural/garment compatibility

        Fallback score (use_vit=False, checkpoints not available):
            Occasion / weather / garment-coverage weights above,
            renormalized to sum to 1.0 with the ViT term dropped. This is
            still real, wardrobe-specific ranking driven by CLIP attribute
            evidence and the existing occasion/weather rules — not a fake
            or random result — it just doesn't have the learned "do these
            items look good together" signal until the checkpoints exist.

        Category names are not hardcoded as occasion rejections.
        Phase 2 visual-attribute evidence supplies the context signal.
        """
        if not candidates:
            return []

        occasion = str(context.get("occasion", "casual")).strip().lower()
        temperature_c = float(
            context.get("temperature_c", context.get("temperature", 25))
        )
        rain = bool(context.get("rain", False))

        def as_dict(value):
            if isinstance(value, dict):
                return value
            try:
                return value.to_dict()
            except Exception:
                return {}

        def prob(values, names):
            names = {str(x).strip().lower() for x in names}
            best = 0.0
            if not isinstance(values, list):
                return best
            for item in values:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", item.get("category", ""))).strip().lower()
                if name in names:
                    try:
                        best = max(best, float(item.get("probability", 0.0)))
                    except (TypeError, ValueError):
                        pass
            return best

        def contextual_scores(metadata):
            data = as_dict(metadata)
            details = data.get("details", {})
            if not isinstance(details, dict):
                details = {}

            attrs = (
                data.get("attributes")
                or details.get("attributes")
                or data.get("visual_attributes")
                or {}
            )
            if not isinstance(attrs, dict):
                attrs = {}

            occasion_verdicts = (
                data.get("occasion_verdicts")
                or details.get("occasion_verdicts")
                or []
            )
            weather_verdicts = (
                data.get("weather_verdicts")
                or details.get("weather_verdicts")
                or []
            )

            # Start from Phase 2's explicit verdict.
            occasion_score = 0.50
            verdicts = [
                str(v.get("verdict", "")).lower()
                for v in occasion_verdicts
                if isinstance(v, dict)
            ]
            if "hard_reject" in verdicts:
                occasion_score = 0.0
            elif "preferred" in verdicts:
                occasion_score = 1.0
            elif "soft_penalty" in verdicts:
                occasion_score = 0.40
            elif "neutral" in verdicts:
                occasion_score = 0.60
            else:
                decision = str(
                    data.get("decision", data.get("phase2_decision", ""))
                ).lower()
                if decision == "filtered":
                    occasion_score = 0.0
                elif decision == "preferred":
                    occasion_score = 1.0
                elif decision == "suitable":
                    occasion_score = 0.70

            weather_score = 0.50
            wverdicts = [
                str(v.get("verdict", "")).lower()
                for v in weather_verdicts
                if isinstance(v, dict)
            ]
            if "hard_reject" in wverdicts:
                weather_score = 0.0
            elif "preferred" in wverdicts:
                weather_score = 1.0
            elif "soft_penalty" in wverdicts:
                weather_score = 0.40
            elif "neutral" in wverdicts:
                weather_score = 0.60

            formality = attrs.get("formality", [])
            style = attrs.get("style", [])

            professional = max(
                prob(formality, {"business casual", "formal", "very formal"}),
                prob(style, {"elegant", "classic", "minimalist", "preppy"}),
            )
            casual = max(
                prob(formality, {"very casual", "casual"}),
                prob(style, {"casual", "sporty", "streetwear", "bohemian"}),
            )

            # Category-specific CLIP attributes contribute evidence without
            # turning category names into hardcoded rejection rules.
            professional_terms = {
                "office blouse", "office shirt", "formal shirt",
                "formal trousers", "office trousers", "formal pants",
                "tailored pants", "tailored trousers", "formal blazer",
                "office blazer", "tailored blazer", "business dress",
                "office dress", "formal dress", "pencil skirt",
                "formal skirt", "formal-looking shorts", "formal shoes",
                "loafers",
            }
            casual_terms = {
                "casual shorts", "sport shorts", "denim shorts",
                "relaxed shorts", "casual t-shirt", "graphic t-shirt",
                "sporty t-shirt", "casual shirt", "casual blouse",
                "baggy jeans", "ripped jeans", "cargo pants",
                "casual pants", "sporty jacket", "denim jacket",
                "running shoes",
            }

            for group_name, values in attrs.items():
                if group_name in {
                    "formality", "style", "color", "pattern", "material",
                    "season", "fit", "sleeve", "length",
                }:
                    continue
                professional = max(professional, prob(values, professional_terms))
                casual = max(casual, prob(values, casual_terms))

            if occasion in {"office", "formal", "business", "interview"}:
                attribute_score = float(
                    np.clip(0.50 + 0.55 * professional - 0.65 * casual, 0.0, 1.0)
                )
            elif occasion in {"college", "casual", "everyday"}:
                attribute_score = float(
                    np.clip(0.50 + 0.35 * casual + 0.15 * professional, 0.0, 1.0)
                )
            elif occasion in {"party", "date", "night out", "wedding"}:
                attribute_score = float(
                    np.clip(0.50 + 0.45 * professional + 0.35 * casual, 0.0, 1.0)
                )
            else:
                attribute_score = 0.50

            occasion_score = float(
                np.clip(0.70 * occasion_score + 0.30 * attribute_score, 0.0, 1.0)
            )

            season = attrs.get("season", [])
            warm = prob(season, {"summer clothing"})
            cold = prob(season, {"winter clothing"})

            if temperature_c >= 27:
                weather_attribute = float(np.clip(0.50 + 0.35 * warm - 0.35 * cold, 0.0, 1.0))
            elif temperature_c <= 15:
                weather_attribute = float(np.clip(0.50 + 0.35 * cold - 0.35 * warm, 0.0, 1.0))
            else:
                weather_attribute = 0.50

            if rain:
                weather_attribute = float(np.clip(weather_attribute * 0.90, 0.0, 1.0))

            weather_score = float(
                np.clip(0.75 * weather_score + 0.25 * weather_attribute, 0.0, 1.0)
            )

            return occasion_score, weather_score

        # Resolve all garment IDs referenced by Phase 3.
        image_names = set()
        for candidate in candidates:
            names = candidate.get("items", []) if isinstance(candidate, dict) else getattr(candidate, "items", [])
            image_names.update(names)

        resolved_names = {}
        for name in image_names:
            key = str(name)
            if key in image_lookup:
                resolved_names[key] = key
            elif Path(key).name in image_lookup:
                resolved_names[key] = Path(key).name
            elif Path(key).stem in image_lookup:
                resolved_names[key] = Path(key).stem

        ordered_names = list(resolved_names.keys())
        images = [image_lookup[resolved_names[name]] for name in ordered_names]

        print(
            f"🔗 Phase 4 resolved {len(ordered_names)} "
            f"of {len(image_names)} referenced garment IDs"
        )

        if len(ordered_names) < len(image_names):
            missing = sorted(
                str(name) for name in image_names
                if str(name) not in resolved_names
            )
            print(f"⚠️ Phase 4 could not resolve image IDs: {missing}")

        embedding_lookup: Dict[str, Any] = {}
        if use_vit:
            embeddings = self.embed_images(images)
            embedding_lookup = dict(zip(ordered_names, embeddings))

        scored = []

        for candidate_index, candidate in enumerate(candidates, start=1):
            if isinstance(candidate, dict):
                item_names = candidate.get("items", [])
                structure = candidate.get("structure", "unknown")
            else:
                item_names = list(getattr(candidate, "items", []))
                structure = getattr(candidate, "structure", "unknown")

            if not item_names:
                continue

            if use_vit and any(name not in embedding_lookup for name in item_names):
                continue

            vit_raw = None
            vit_score = None

            if use_vit:
                token_array = np.stack(
                    [embedding_lookup[name] for name in item_names],
                    axis=0,
                )
                tokens = torch.tensor(
                    token_array,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                vit_raw = float(self.vit.score_compatibility(tokens).item())
                # Preserve ordering while normalizing to [0,1].
                vit_score = float(torch.sigmoid(torch.tensor(vit_raw)).item())

            outfit_items = []
            occasion_scores = []
            weather_scores = []

            for name in item_names:
                metadata = metadata_lookup.get(name, {})
                if not metadata:
                    path = Path(str(name))
                    metadata = metadata_lookup.get(
                        path.name,
                        metadata_lookup.get(path.stem, {}),
                    )

                occ, weather = contextual_scores(metadata)
                occasion_scores.append(occ)
                weather_scores.append(weather)

                outfit_items.append({
                    "id": name,
                    "category": metadata.get(
                        "category",
                        metadata.get("category_name", "unknown"),
                    ),
                    "category_type": metadata.get("slot", "unknown"),
                    "phase2_decision": metadata.get(
                        "phase2_decision",
                        metadata.get("decision"),
                    ),
                    "occasion_score": occ,
                    "weather_score": weather,
                })

            occasion_score = float(np.mean(occasion_scores)) if occasion_scores else 0.0
            weather_score = float(np.mean(weather_scores)) if weather_scores else 0.0

            slots = {
                str(item["category_type"]).strip().lower()
                for item in outfit_items
            }
            if {"upper", "lower"} <= slots or {"top", "bottom"} <= slots:
                garment_score = 1.0
            elif slots & {"upper", "lower", "top", "bottom", "layer", "outerwear"}:
                garment_score = 0.60
            else:
                garment_score = 0.40

            if use_vit:
                final_score = (
                    OCCASION_WEIGHT * occasion_score
                    + VIT_WEIGHT * vit_score
                    + WEATHER_WEIGHT * weather_score
                    + GARMENT_COMPATIBILITY_WEIGHT * garment_score
                )
                score_weights = {
                    "occasion": OCCASION_WEIGHT,
                    "vit": VIT_WEIGHT,
                    "weather": WEATHER_WEIGHT,
                    "garment_compatibility": GARMENT_COMPATIBILITY_WEIGHT,
                }
            else:
                # No learned compatibility signal available — renormalize
                # the remaining rule-based weights so they still sum to 1.0.
                fallback_total = OCCASION_WEIGHT + WEATHER_WEIGHT + GARMENT_COMPATIBILITY_WEIGHT
                final_score = (
                    OCCASION_WEIGHT * occasion_score
                    + WEATHER_WEIGHT * weather_score
                    + GARMENT_COMPATIBILITY_WEIGHT * garment_score
                ) / fallback_total
                score_weights = {
                    "occasion": OCCASION_WEIGHT / fallback_total,
                    "vit": 0.0,
                    "weather": WEATHER_WEIGHT / fallback_total,
                    "garment_compatibility": GARMENT_COMPATIBILITY_WEIGHT / fallback_total,
                }

            scored.append({
                "candidate_index": candidate_index,
                "item_ids": item_names,
                "items": outfit_items,
                "structure": structure,
                "compatibility_score": float(final_score),
                "final_score": float(final_score),
                "occasion_score": float(occasion_score),
                "vit_score_raw": vit_raw,
                "vit_score": vit_score,
                "weather_score": float(weather_score),
                "garment_compatibility_score": float(garment_score),
                "score_weights": score_weights,
                "ranking_mode": "ai" if use_vit else "rule_based",
                "outfit_size": len(item_names),
            })

        scored.sort(key=lambda result: result["final_score"], reverse=True)

        for rank, result in enumerate(scored, start=1):
            result["rank"] = rank

        return scored

    # ===================================================================
    # Main end-to-end pipeline
    # ===================================================================

    @torch.inference_mode()
    def compose_outfits(
        self,
        items: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Run the COMPLETE outfit recommendation pipeline.

        Phase 1:
            CLIP zero-shot garment classification

        Phase 2:
            Existing wardrobe filtering

        Phase 3:
            Existing structural candidate generation

        Phase 4:
            ResNet embeddings + ViT compatibility ranking

        IMPORTANT:
            Phase 4 does NOT create its own candidates.
        """

        print("\n" + "=" * 70)
        print("END-TO-END OUTFIT INFERENCE")
        print("=" * 70)

        print(
            f"Device: {self.device}"
        )

        print(
            f"Items received: {len(items)}"
        )

        print(
            f"Context: {context}"
        )

        if not self.clip_loaded:
            # CLIP is the hard requirement: Phase 1 classification and the
            # occasion/weather rules both depend on it. ResNet/ViT are
            # optional — see the use_vit fallback below.
            error = {
                "error": "Models not loaded",
                "details": self.model_errors,
            }

            print(
                f"❌ {error}"
            )

            return [error]

        use_vit = self.resnet_loaded and self.vit_loaded
        if not use_vit:
            print(
                "⚠️ ResNet/ViT checkpoints not loaded — ranking outfits by "
                "occasion/weather/coverage rules only (see model_errors)."
            )

        if not items:
            print("⚠️ No wardrobe items supplied")
            return []

        # ---------------------------------------------------------------
        # Validate all images
        # ---------------------------------------------------------------

        normalized_items: List[
            Dict[str, Any]
        ] = []

        for index, item in enumerate(items):
            image = item.get("image")

            if image is None:
                print(
                    f"⚠️ Skipping item {index}: no image"
                )
                continue

            image = ensure_rgb_image(image)

            item_id = item.get(
                "id",
                f"item_{index}.jpg",
            )

            normalized_items.append(
                {
                    "id": str(item_id),
                    "image": image,
                }
            )

        if not normalized_items:
            return []

        # ---------------------------------------------------------------
        # Phase 1 (CLIP) + Phase 2
        # ---------------------------------------------------------------

        print("\n" + "-" * 70)
        print("PHASE 1 (CLIP) + PHASE 2")
        print("-" * 70)

        temp_predictions_dir, phase2_result = (
            self._run_phase1_and_phase2(
                normalized_items,
                context,
            )
        )

        # ---------------------------------------------------------------
        # Phase 3
        # ---------------------------------------------------------------

        print("\n" + "-" * 70)
        print("PHASE 3")
        print("-" * 70)

        category_slot_map = load_category_slot_map(
            Path.cwd()
        )

        phase2_dict = phase2_result.to_dict()

        phase3_result = generate_candidates(
            phase2_dict,
            category_slot_map,
            include_layering=True,
        )

        # ---------------------------------------------------------------
        # Extract candidate/survivor structures.
        #
        # The existing Phase 3 code exposes these as objects, while
        # the saved JSON representation uses dictionaries.
        # ---------------------------------------------------------------

        if isinstance(phase3_result, dict):
            candidates = phase3_result.get(
                "candidate_outfits",
                [],
            )

            survivors = phase3_result.get(
                "garments_entering_phase3",
                [],
            )

        else:
            candidates = getattr(
                phase3_result,
                "candidate_outfits",
                [],
            )

            survivors = getattr(
                phase3_result,
               "garments_entering_phase3",
                [],
            )

        print(
            f"✅ Phase 3 produced "
            f"{len(candidates)} candidate outfits"
        )

        self._debug_phase3_result(phase3_result, candidates, survivors)

        if not candidates:
            print(
                "⚠️ Phase 3 produced no candidates"
            )
            return []

        # ---------------------------------------------------------------
        # Build metadata from Phase 2/3 survivors.
        # ---------------------------------------------------------------

        metadata_lookup: Dict[
            str,
            Dict[str, Any],
        ] = {}

        if isinstance(survivors, list):
            for survivor in survivors:
                if isinstance(survivor, dict):
                    image_name = survivor.get(
                        "image",
                        survivor.get("id"),
                    )

                    if image_name:
                        metadata_lookup[
                            str(image_name)
                        ] = survivor

                else:
                    image_name = getattr(
                        survivor,
                        "image",
                        getattr(
                            survivor,
                            "id",
                            None,
                        ),
                    )

                    if image_name:
                        metadata_lookup[
                            str(image_name)
                        ] = {
                            "category": getattr(
                                survivor,
                                "category_name",
                                getattr(
                                    survivor,
                                    "category",
                                    "unknown",
                                ),
                            ),
                            "slot": getattr(
                                survivor,
                                "slot",
                                "unknown",
                            ),
                            "phase2_decision": getattr(
                                survivor,
                                "phase2_decision",
                                None,
                            ),
                        }

        # ---------------------------------------------------------------
        # Build image lookup.
        #
        # Phase 3 stores image basenames such as bottom1.jfif.
        # Match them to the original input IDs without using the name
        # to infer a category.
        # ---------------------------------------------------------------

        image_lookup: Dict[str, Image.Image] = {}

        for item in normalized_items:
            item_id = str(item["id"])
            image = item["image"]
            path = Path(item_id)

            image_lookup[path.name] = image
            image_lookup[path.stem] = image
            image_lookup[item_id] = image

        # ---------------------------------------------------------------
        # Phase 4
        # ---------------------------------------------------------------

        print("\n" + "-" * 70)
        if use_vit:
            print("PHASE 4 — CONTEXT-AWARE RESNET + VIT RANKING")
        else:
            print("PHASE 4 — RULE-BASED RANKING (ResNet/ViT checkpoints not loaded)")
        print("-" * 70)

        ranked = self._rank_phase3_candidates(
            candidates,
            survivors,
            image_lookup,
            metadata_lookup,
            context,
            use_vit=use_vit,
        )

        print(
            f"✅ Ranked {len(ranked)} Phase 3 candidates"
        )

        self._debug_ranking(ranked)

        print("\nFINAL RANKING")

        for result in ranked:
            print(
                f"#{result['rank']} "
                f"score={result['compatibility_score']:.6f} "
                f"{result['item_ids']}"
            )

        # ---------------------------------------------------------------
        # Cleanup temporary Phase 1 predictions
        # ---------------------------------------------------------------

        try:
            import shutil

            shutil.rmtree(
                temp_predictions_dir,
                ignore_errors=True,
            )

        except Exception:
            pass

        return ranked

    # ===================================================================
    # Status
    # ===================================================================

    def get_model_status(
        self,
    ) -> Dict[str, Any]:
        """Return current inference model status."""

        return {
            "models_loaded": self.models_loaded,
            "deepfashion_loaded": False,
            "resnet_loaded": self.resnet_loaded,
            "vit_loaded": self.vit_loaded,
            "clip_loaded": self.clip_loaded,
            "filename_category_detection": False,
            "errors": self.model_errors,
            # CLIP alone is enough to generate outfits (rule-based ranking);
            # the full learned ranking additionally needs ResNet + ViT.
            "can_recommend": self.clip_loaded,
            "ranking_mode": "ai" if self.models_loaded else ("rule_based" if self.clip_loaded else "unavailable"),
            "device": self.device,
            "resnet_model": self.resnet is not None,
            "vit_model": self.vit is not None,
            "deepfashion_model": False,
            "clip_model": self.clip_model is not None,
            "clip_categories": self.clip_categories,
        }

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING END-TO-END OUTFIT INFERENCE")
    print("=" * 70)

    wardrobe_dir = Path("wardrobe")

    # Automatically load all supported image files.
    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".jfif",
        ".webp",
        ".bmp",
    }

    image_paths = sorted(
        path
        for path in wardrobe_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )

    print(
        f"📦 Found {len(image_paths)} wardrobe images"
    )

    if not image_paths:
        print(
            f"❌ No supported images found in {wardrobe_dir}"
        )
        raise SystemExit(1)

    items = []

    for image_path in image_paths:
        try:
            print(f"📷 Loading: {image_path.name}")

            image = Image.open(image_path).convert("RGB")

            items.append(
                {
                    "id": image_path.name,
                    "image": image,
                }
            )

        except Exception as exc:
            print(
                f"⚠️ Could not load {image_path.name}: {exc}"
            )

    print(
        f"✅ Successfully loaded {len(items)} images"
    )

    service = InferenceService()

    context = {
        "occasion": "party",
        "temperature_c": 25,
        "rain": False,
    }

    results = service.compose_outfits(
        items,
        context,
    )

    print("\n" + "=" * 70)
    print("FINAL OUTFIT RANKING")
    print("=" * 70)

    if not results:
        print("⚠️ No outfit recommendations generated.")
    elif isinstance(results, dict) and results.get("error"):
        print(f"❌ Inference failed: {results.get('error')}")
        for detail in results.get("details", []):
            print(f"   - {detail}")
    else:
        for result in results:
            if not isinstance(result, dict):
                print(f"⚠️ Unexpected result: {result!r}")
                continue
            try:
                score = float(result.get("compatibility_score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            print(
                f"#{result.get('rank', '?')} "
                f"| score={score:.6f} "
                f"| {result.get('item_ids', [])}"
            )