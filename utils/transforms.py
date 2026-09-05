from typing import Tuple

import torchvision.transforms as T


def build_inference_transform(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_train_transforms(image_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize(int(image_size * 1.1), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
