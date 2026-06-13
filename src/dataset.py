import json
import cv2
import torch

from pathlib import Path
from torch.utils.data import Dataset


class GCPDataset(Dataset):

    SHAPE_MAP = {
        "Cross": 0,
        "Square": 1,
        "L-Shape": 2
    }

    def __init__(
        self,
        json_path,
        image_root,
        indices=None
    ):

        self.image_root = Path(image_root)

        with open(json_path) as f:
            labels = json.load(f)

        self.samples = []
        all_samples = []

        for path, info in labels.items():

            if "verified_shape" not in info:
                continue

            all_samples.append(
                (path, info)
            )
        
        if indices is None:
            self.samples = all_samples
        else:
            self.samples = [
                all_samples[i]
                for i in indices
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, info = self.samples[idx]

        img_path = (
            self.image_root / path
        )

        image = cv2.imread(str(img_path))

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ORIGINAL dimensions
        h, w = image.shape[:2]

        x = info["mark"]["x"]
        y = info["mark"]["y"]

        x_norm = x / w
        y_norm = y / h

        # ONLY AFTER normalization
        image = cv2.resize(
            image,
            (1024,1024)
        )

        shape = self.SHAPE_MAP[
            info["verified_shape"]
        ]

        image = (
            torch.tensor(image)
            .permute(2,0,1)
            .float()
            / 255.0
        )

        coords = torch.tensor(
            [x_norm, y_norm],
            dtype=torch.float32
        )

        shape = torch.tensor(
            shape,
            dtype=torch.long
        )

        return image, coords, shape