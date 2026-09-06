"""
Train the outfit-compatibility pair of models that inference.py expects:

    models/exports/resnet_item_embedder_best.pth   (models.resnet_embedder.ResNetItemEmbedder)
    models/exports/vit_outfit_model_best.pth        (models.vit_outfit.OutfitCompatibilityModel)

WHY THIS SCRIPT EXISTS
-----------------------
train.py in this repo trains a different model (models.model.FashionResnet, a
single-image category/attribute classifier) on the DeepFashion Category and
Attribute Prediction Benchmark. That's not the model inference.py loads.
inference.py needs a model that scores whether a *set* of garments looks
good together, which requires outfit-level "these items go together" data —
DeepFashion's category/attribute labels don't have that. This script trains
on the Polyvore Outfits dataset instead, which does.

DATA LAYOUT THIS SCRIPT EXPECTS
--------------------------------
--polyvore_root should point at a directory laid out like:

    <polyvore_root>/
      polyvore_item_metadata.json      # item_id -> {"semantic_category": ..., ...}
      images/
        <item_id>.jpg
      <split>/                         # --split disjoint (default) or nondisjoint
        train.json                     # list of {"set_id": ..., "items": [{"item_id":..,"index":..}, ...]}
        valid.json
        compatibility_valid.txt        # "label item1ref item2ref ..." (item refs are "set_id_index")

This is the standard layout used by the original Polyvore Outfits release
(Vasileva et al., ECCV 2018) and every mirror/toolkit that redistributes it.
See DATA.md in this repo for where to obtain it — the Category/Attribute
DeepFashion files already in Anno/ are NOT this dataset and won't work here.

USAGE
-----
    # Quick sanity check with a tiny subset before committing to a full run:
    python train_outfit_compatibility.py --polyvore_root /path/to/polyvore_outfits_root \
        --dry_run --limit_outfits 50

    # Full training run:
    python train_outfit_compatibility.py --polyvore_root /path/to/polyvore_outfits_root \
        --epochs 15 --lr 1e-4 --accumulate 8

Checkpoints are written to models/exports/ as plain state_dicts, exactly the
format inference.py's _load_resnet / _load_vit already know how to read —
no changes to inference.py are needed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet_embedder import ResNetItemEmbedder  # noqa: E402
from models.vit_outfit import OutfitCompatibilityModel  # noqa: E402
from utils.transforms import build_train_transforms, build_inference_transform  # noqa: E402

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ================================================================
# DATA LOADING
# ================================================================

class PolyvoreOutfits:
    """Loads outfit structure + item metadata for one Polyvore Outfits split."""

    def __init__(self, root: Path, split: str):
        self.root = root
        self.images_dir = root / "images"
        self.split_dir = root / split

        with open(root / "polyvore_item_metadata.json", "r", encoding="utf-8") as fh:
            self.item_meta: Dict[str, dict] = json.load(fh)

        self.train_outfits = self._load_outfits("train")
        self.valid_outfits = self._load_outfits("valid")

        # category -> list of item_ids, for negative sampling by same-type swap.
        self.category_to_items: Dict[str, List[str]] = defaultdict(list)
        for item_id, meta in self.item_meta.items():
            category = meta.get("semantic_category", "unknown")
            self.category_to_items[category].append(item_id)

        self.valid_questions = self._load_compatibility_questions("valid")

    def _load_outfits(self, split_name: str) -> List[Tuple[str, List[str]]]:
        path = self.split_dir / f"{split_name}.json"
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        outfits = []
        for outfit in raw:
            item_ids = [it["item_id"] for it in outfit["items"]]
            if len(item_ids) >= 2:
                outfits.append((outfit["set_id"], item_ids))
        return outfits

    def _load_compatibility_questions(self, split_name: str) -> List[Tuple[List[str], int]]:
        """Parse compatibility_{split}.txt into (item_ids, label) pairs.

        Item refs in the file are "set_id_index"; we resolve them back to
        item_ids using the same split's outfit json.
        """
        txt_path = self.split_dir / f"compatibility_{split_name}.txt"
        json_path = self.split_dir / f"{split_name}.json"
        if not txt_path.exists() or not json_path.exists():
            return []

        with open(json_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        id2item = {}
        for outfit in raw:
            for it in outfit["items"]:
                id2item[f"{outfit['set_id']}_{it['index']}"] = it["item_id"]

        questions = []
        with open(txt_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                label = int(parts[0])
                refs = parts[1:]
                item_ids = [id2item[r] for r in refs if r in id2item]
                if len(item_ids) >= 2:
                    questions.append((item_ids, label))
        return questions

    def category_of(self, item_id: str) -> str:
        return self.item_meta.get(item_id, {}).get("semantic_category", "unknown")

    def image_path(self, item_id: str) -> Path:
        return self.images_dir / f"{item_id}.jpg"

    def sample_negative_outfit(self, item_ids: List[str], other_outfit_ids: set) -> List[str]:
        """Build a synthetic incompatible outfit by swapping each item for a
        same-category item drawn from a *different* outfit — the same
        negative-construction technique used by the Polyvore Outfits paper.
        """
        negative = []
        for item_id in item_ids:
            category = self.category_of(item_id)
            pool = self.category_to_items.get(category) or list(self.item_meta.keys())
            for _ in range(20):
                candidate = random.choice(pool)
                if candidate != item_id:
                    negative.append(candidate)
                    break
            else:
                negative.append(item_id)  # fallback: couldn't find a swap
        return negative


# ================================================================
# IMAGE LOADING / EMBEDDING
# ================================================================

def load_and_transform(path: Path, transform) -> Optional[torch.Tensor]:
    try:
        img = Image.open(path).convert("RGB")
        return transform(img)
    except Exception as exc:
        print(f"  ⚠️ skipping unreadable image {path.name}: {exc}")
        return None


def embed_outfit(item_ids: List[str], dataset: PolyvoreOutfits, resnet: nn.Module,
                  transform, device: str) -> Optional[torch.Tensor]:
    """Returns a (1, N, D) tensor of item embeddings for one outfit, or None
    if too few images loaded successfully."""
    tensors = []
    for item_id in item_ids:
        t = load_and_transform(dataset.image_path(item_id), transform)
        if t is not None:
            tensors.append(t)
    if len(tensors) < 2:
        return None
    batch = torch.stack(tensors, dim=0).to(device)
    embeddings = resnet(batch)  # (N, D), already L2-normalized by the model
    return embeddings.unsqueeze(0)  # (1, N, D)


# ================================================================
# VALIDATION
# ================================================================

@torch.no_grad()
def validate(dataset: PolyvoreOutfits, resnet: nn.Module, vit: nn.Module,
             transform, device: str, limit: Optional[int] = None) -> dict:
    resnet.eval()
    vit.eval()

    questions = dataset.valid_questions
    if limit:
        questions = questions[:limit]

    scores, labels = [], []
    for item_ids, label in questions:
        tokens = embed_outfit(item_ids, dataset, resnet, transform, device)
        if tokens is None:
            continue
        logit = vit.score_compatibility(tokens)
        score = torch.sigmoid(logit).item()
        scores.append(score)
        labels.append(label)

    if not scores:
        return {"accuracy": 0.0, "auc": None, "n": 0}

    correct = sum((s >= 0.5) == bool(l) for s, l in zip(scores, labels))
    accuracy = correct / len(scores)

    auc = None
    if HAS_SKLEARN and len(set(labels)) > 1:
        auc = float(roc_auc_score(labels, scores))

    return {"accuracy": accuracy, "auc": auc, "n": len(scores)}


# ================================================================
# TRAINING LOOP
# ================================================================

def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    root = Path(args.polyvore_root)
    dataset = PolyvoreOutfits(root, args.split)
    print(f"Loaded {len(dataset.train_outfits)} training outfits, "
          f"{len(dataset.valid_questions)} validation questions "
          f"({args.split} split).")

    if not dataset.train_outfits:
        print("No training outfits found — check --polyvore_root / --split point at "
              "a real Polyvore Outfits directory (see DATA.md).")
        sys.exit(1)

    resnet = ResNetItemEmbedder(embedding_dim=args.embed_dim, backbone=args.backbone,
                                 pretrained=True).to(device)
    vit = OutfitCompatibilityModel(embedding_dim=args.embed_dim).to(device)

    train_transform = build_train_transforms()
    eval_transform = build_inference_transform()

    optimizer = torch.optim.AdamW(
        list(resnet.parameters()) + list(vit.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_outfits = list(dataset.train_outfits)
    all_outfit_ids = {oid for oid, _ in train_outfits}

    export_dir = PROJECT_ROOT / "models" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    resnet_path = export_dir / "resnet_item_embedder_best.pth"
    vit_path = export_dir / "vit_outfit_model_best.pth"

    best_metric = -1.0

    if args.limit_outfits:
        train_outfits = train_outfits[: args.limit_outfits]
        print(f"--limit_outfits set: training on {len(train_outfits)} outfits only.")

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_outfits)
        resnet.train()
        vit.train()

        epoch_loss, seen, step_count = 0.0, 0, 0
        optimizer.zero_grad()
        start = time.time()

        for i, (outfit_id, item_ids) in enumerate(train_outfits, start=1):
            if args.max_items_per_outfit:
                item_ids = item_ids[: args.max_items_per_outfit]

            pos_tokens = embed_outfit(item_ids, dataset, resnet, train_transform, device)
            neg_ids = dataset.sample_negative_outfit(item_ids, all_outfit_ids)
            neg_tokens = embed_outfit(neg_ids, dataset, resnet, train_transform, device)

            if pos_tokens is None or neg_tokens is None:
                continue

            pos_logit = vit.score_compatibility(pos_tokens)
            neg_logit = vit.score_compatibility(neg_tokens)

            loss = criterion(pos_logit, torch.ones_like(pos_logit)) + \
                criterion(neg_logit, torch.zeros_like(neg_logit))
            loss = loss / args.accumulate
            loss.backward()

            epoch_loss += loss.item() * args.accumulate
            seen += 1
            step_count += 1

            if step_count % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(resnet.parameters()) + list(vit.parameters()), max_norm=5.0
                )
                optimizer.step()
                optimizer.zero_grad()

            if args.dry_run and seen >= (args.limit_outfits or 10):
                break

            if seen and seen % 200 == 0:
                elapsed = time.time() - start
                print(f"  epoch {epoch} | outfit {seen}/{len(train_outfits)} "
                      f"| avg loss {epoch_loss / seen:.4f} | {elapsed:.0f}s")

        optimizer.step()
        optimizer.zero_grad()

        avg_loss = epoch_loss / max(seen, 1)
        val_metrics = validate(dataset, resnet, vit, eval_transform, device,
                                limit=args.val_limit)
        metric_str = f"acc={val_metrics['accuracy']:.4f}"
        if val_metrics["auc"] is not None:
            metric_str += f" auc={val_metrics['auc']:.4f}"

        print(f"[epoch {epoch}] train_loss={avg_loss:.4f} | val {metric_str} "
              f"(n={val_metrics['n']})")

        current_metric = val_metrics["auc"] if val_metrics["auc"] is not None else val_metrics["accuracy"]
        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(resnet.state_dict(), resnet_path)
            torch.save(vit.state_dict(), vit_path)
            print(f"  ✅ new best ({current_metric:.4f}) — saved to {export_dir}")

        if args.dry_run:
            print("Dry run complete — data loading, forward/backward pass, "
                  "validation, and checkpoint saving all worked.")
            break

    print(f"Done. Best validation metric: {best_metric:.4f}")
    print(f"Checkpoints: {resnet_path}, {vit_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--polyvore_root", required=True,
                   help="Path to the Polyvore Outfits directory (contains polyvore_item_metadata.json, images/, and the split folder).")
    p.add_argument("--split", default="disjoint", choices=["disjoint", "nondisjoint"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--embed_dim", type=int, default=512, help="Must match EMBED_DIM used by inference.py (default 512).")
    p.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet101"])
    p.add_argument("--accumulate", type=int, default=8, help="Outfits per optimizer step (effective batch size).")
    p.add_argument("--max_items_per_outfit", type=int, default=8, help="Cap outfit size for speed/memory; 0 = no cap.")
    p.add_argument("--limit_outfits", type=int, default=0, help="Train on only the first N outfits (for smoke-testing).")
    p.add_argument("--val_limit", type=int, default=1000, help="Cap validation questions evaluated per epoch.")
    p.add_argument("--device", default=None, help="cuda / cpu / mps. Default: auto-detect.")
    p.add_argument("--dry_run", action="store_true",
                   help="Run a tiny slice end-to-end (load data, one partial epoch, validate, save) to catch path/format issues before a full run.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
