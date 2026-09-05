"""
deepfashion_infer.py

Phase 1: IMAGE -> DEEPFASHION -> STRUCTURED JSON

This module performs the SAME inference as the existing, verified
`test_on_image.py`, but returns a structured Python dict / JSON
instead of a matplotlib PNG.

It intentionally reuses the existing repository code as the source of
truth for anything that determines *correctness* of the predictions:

  - `models.model.FashionResnet`          -> model architecture
  - `train.load_model`                    -> checkpoint loading
  - `data.input.get_ctg_name`             -> category label source
  - `data.dataset_fashion.get_attr_names_and_types` -> attribute labels/types

The only code duplicated from `test_on_image.py` is the small
`load_image()` preprocessing function and the category-index
off-by-one correction (`['n/a'] + cat_names[0:-1]`), because
`test_on_image.py` itself calls `parse_args()` / `argparse` at import
time and cannot be imported as a library without side effects. Both
pieces are copied verbatim (not reinterpreted) so behavior matches
exactly.

Nothing here retrains, redownloads, or modifies the model or
checkpoint.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import (
    Compose,
    Resize,
    Normalize,
    ToTensor,
    Lambda,
    TenCrop,
)

# The repo's modules (models/, data/, train.py, ...) are imported using
# their existing package-relative style, so this script must be able to
# see the project root on sys.path regardless of the current working
# directory it's invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import FashionResnet  # noqa: E402
from data.dataset_fashion import get_attr_names_and_types  # noqa: E402
from data import input as df_input  # noqa: E402
from train import load_model  # noqa: E402

from garment_tagger.schema import (  # noqa: E402
    NamedProbability,
    BoundingBox,
    GarmentTagResult,
)

ATTR_TYPE_NAMES = ["texture", "fabric", "shape", "part", "style"]

DEFAULT_CHECKPOINT = PROJECT_ROOT / "results" / "pretrained_model" / "epoch_340.pth"
DEFAULT_RESNET_TYPE = "resnet18"  # verified architecture for epoch_340.pth


def load_labels(df_dir):
    """Load category and attribute label vocabularies.

    Reuses the existing repository's loaders and preserves the exact
    category-indexing correction already present in test_on_image.py
    (the model has an unused/no-op category slot, so we pad with
    'n/a' and drop the last raw label to stay at 50 categories).

    :param df_dir: directory containing the 'Anno' folder
    :returns: (cat_names, attr_names, attr_types)
    """
    df_dir = str(df_dir)
    cat_names = df_input.get_ctg_name(df_dir)[0]
    # Same correction as test_on_image.py:
    cat_names = ["n/a"] + cat_names[0:-1]

    attr_names, attr_types = get_attr_names_and_types(df_dir)
    attr_names = np.asarray(attr_names)
    attr_types = torch.FloatTensor(np.asarray(attr_types))

    return cat_names, attr_names, attr_types


def load_image(filename, ensemble=True):
    """Load and preprocess an image exactly like test_on_image.py.

    :param filename: path to the image
    :param ensemble: if True, perform ten-crop and return a
      (10, 3, 224, 224) tensor; otherwise a single (1, 3, 224, 224)
      center-resized tensor.
    :returns: preprocessed image tensor
    """
    transform_list = []
    if ensemble:
        norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform_list.append(Resize((256, 256)))
        transform_list.append(TenCrop(224))
        transform_list.append(
            Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))
        )
        transform_list.append(
            Lambda(lambda crops: torch.stack([norm(crop) for crop in crops]))
        )
    else:
        transform_list.append(Resize((224, 224)))
        transform_list.append(ToTensor())
        transform_list.append(Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transformer = Compose(transform_list)

    filepath = os.path.abspath(str(filename))
    img = Image.open(filepath)
    img = img.convert("RGB")
    img = transformer(img)
    if ensemble:
        return img
    else:
        return img.unsqueeze(0)


def build_model(checkpoint, resnet_type=DEFAULT_RESNET_TYPE, num_categories=50, num_attrs=1000):
    """Build FashionResnet and load the verified checkpoint (CPU).

    :param checkpoint: path to the .pth checkpoint file
    :param resnet_type: backbone type; must match the checkpoint
      (epoch_340.pth was trained with resnet18)
    :returns: (model in eval mode, epoch stored in the checkpoint)
    """
    model = FashionResnet(num_categories, num_attrs, resnet_type)
    epoch = load_model(model, str(checkpoint), optimizer=None, devices=[])
    # NOTE: test_on_image.py never calls model.eval() before running
    # inference, so the model stays in its default training-mode state.
    # Because FashionResnet uses BatchNorm, eval() vs train() actually
    # changes the numbers here (train mode uses the current 10-crop
    # batch statistics rather than the checkpoint's running stats).
    # To exactly reproduce test_on_image.py's verified output, we
    # deliberately match that behavior and do NOT call model.eval().
    return model, epoch


def run_inference(
    image_path,
    df_dir=PROJECT_ROOT,
    checkpoint=DEFAULT_CHECKPOINT,
    resnet_type=DEFAULT_RESNET_TYPE,
    softmax_temp=1.0,
    top_k=3,
    model=None,
    cat_names=None,
    attr_names=None,
    attr_types=None,
):
    """Run the same inference as test_on_image.py's plot_classification,
    but return a structured dict instead of drawing a PNG.

    A model / label set can optionally be passed in (e.g. when tagging
    many images in a loop) to avoid reloading the checkpoint per image.

    :param image_path: path to the image to classify
    :param df_dir: directory containing the 'Anno' folder
    :param checkpoint: path to the .pth checkpoint
    :param resnet_type: resnet backbone (must match checkpoint)
    :param softmax_temp: softmax temperature for category classification
    :param top_k: how many predictions to keep, per category and per
      attribute group (configurable, default 3)
    :returns: dict matching schema.GarmentTagResult
    """
    image_path = Path(image_path).resolve()

    if cat_names is None or attr_names is None or attr_types is None:
        cat_names, attr_names, attr_types = load_labels(df_dir)

    if model is None:
        model, _epoch = build_model(checkpoint, resnet_type)

    with torch.no_grad():
        img = load_image(filename=image_path, ensemble=True)
        out_cls, out_bin, out_bbox = model(img)

        # Average the ten-crop predictions, exactly like test_on_image.py.
        out_cls = out_cls.mean(dim=0, keepdim=True)
        out_bin = out_bin.mean(dim=0, keepdim=True)
        # Central crop's bbox (index 4), same as test_on_image.py.
        out_bbox = out_bbox[4:5]

    # ---- Category (softmax over 50 classes) ----
    cls_probs = torch.softmax(out_cls / softmax_temp, dim=1)[0]
    cat_topk_values, cat_topk_indices = cls_probs.topk(top_k)
    category = [
        NamedProbability(name=cat_names[idx], probability=float(val))
        for val, idx in zip(cat_topk_values.tolist(), cat_topk_indices.tolist())
    ]

    # ---- Attributes (sigmoid over 1000 attrs, grouped by type) ----
    attributes = {}
    for j in range(1, 6):
        out_bin_subset = out_bin.clone()
        out_bin_subset[:, (attr_types != j)] = -1000.0
        this_topk_values, this_topk_indices = out_bin_subset.topk(top_k, 1, True, True)
        this_topk_values = torch.sigmoid(this_topk_values[0])
        this_topk_indices = this_topk_indices.numpy()[0]
        this_attr_names = attr_names[this_topk_indices]

        group_name = ATTR_TYPE_NAMES[j - 1]
        attributes[group_name] = [
            NamedProbability(name=str(name), probability=float(prob))
            for name, prob in zip(this_attr_names, this_topk_values.tolist())
        ]

    # ---- Bounding box ----
    sz = img.shape[-1]  # 224, the crop size
    normalized_bbox = out_bbox[0].tolist()
    pixel_bbox = (out_bbox[0] * sz).tolist()
    bounding_box = BoundingBox(
        normalized=normalized_bbox,
        pixels_224=pixel_bbox,
        crop_size=sz,
    )

    result = GarmentTagResult(
        image=str(image_path),
        checkpoint=str(Path(checkpoint).resolve()),
        resnet_type=resnet_type,
        category=category,
        attributes=attributes,
        bounding_box=bounding_box,
    )
    return result.to_dict()


def save_json(result_dict, output_path):
    """Save a result dict as JSON, creating parent directories as needed.

    Uses pathlib throughout so this is safe on Windows (avoids mixing
    '/' and '\\' into a malformed path).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: run the verified DeepFashion classifier on "
        "one image and emit structured JSON (no PNG)."
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image to classify."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to write the resulting JSON. If omitted, JSON is only printed.",
    )
    parser.add_argument(
        "--df_dir", type=str, default=str(PROJECT_ROOT),
        help="Directory containing the 'Anno' folder (default: project root).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
        help="Path to the model checkpoint (default: epoch_340.pth).",
    )
    parser.add_argument(
        "--resnet_type", type=str, default=DEFAULT_RESNET_TYPE,
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="Must match the checkpoint's architecture (epoch_340.pth -> resnet18).",
    )
    parser.add_argument(
        "--softmax_temp", type=float, default=1.0, help="Softmax temperature for category."
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="How many predictions to return per category/attribute group.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_inference(
        image_path=args.image,
        df_dir=args.df_dir,
        checkpoint=args.checkpoint,
        resnet_type=args.resnet_type,
        softmax_temp=args.softmax_temp,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2))
    if args.output:
        out_path = save_json(result, args.output)
        print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
