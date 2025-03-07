from .extended import ExtendedVisionDataset
from typing import Any, Callable, List, Optional, Tuple, Union
from pathlib import Path
import random

class Ultralytics(ExtendedVisionDataset):
    image_paths: list[tuple[int, str]]
    def __init__(
        self,
        *,
        split: str,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        root = Path(root)
        labels_file = root / "labels.txt"
        with open(labels_file, "r") as f:
            self.labels = [l.strip() for l in f.readlines()]
        data_folder = Path(root) / Path(split)

        self.image_paths = []

        for i, cls in enumerate(self.labels):
            for image_path in (data_folder / cls).iterdir():
                self.image_paths.append((i, str(image_path)))
        random.seed(42)
        random.shuffle(self.image_paths)

    def get_image_data(self, index: int) -> bytes:
        with open(self.image_paths[index][1], mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Any:
        return self.image_paths[index][0]

    def __len__(self) -> int:
        return len(self.image_paths)
