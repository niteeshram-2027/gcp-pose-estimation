import torch
import torch.nn as nn

from tqdm import tqdm

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from dataset import (
    GCPDataset,
    train_transform,
    val_transform
)

from model import GCPModel


JSON_PATH = "/content/drive/MyDrive/skylark_gcp/gcp_marks.json"
IMAGE_ROOT = "/content/drive/MyDrive/skylark_gcp/train_dataset"

BATCH_SIZE = 4
EPOCHS = 1
LR = 1e-4


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)


full_dataset = GCPDataset(
    JSON_PATH,
    IMAGE_ROOT
)

indices = list(range(len(full_dataset)))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42
)

train_dataset = GCPDataset(
    JSON_PATH,
    IMAGE_ROOT,
    train_idx,
    train_transform
)

val_dataset = GCPDataset(
    JSON_PATH,
    IMAGE_ROOT,
    val_idx,
    val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2
)

model = GCPModel().to(device)


shape_map = {
    "Cross": 0,
    "Square": 1,
    "L-Shape": 2
}

labels = [
    shape_map[
        sample[1]["verified_shape"]
    ]
    for sample in full_dataset.samples
]

weights = compute_class_weight(
    class_weight="balanced",
    classes=[0,1,2],
    y=labels
)

weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)


coord_loss_fn = nn.SmoothL1Loss()

shape_loss_fn = nn.CrossEntropyLoss(
    weight=weights
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

best_val_loss = float("inf")


for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    loop = tqdm(train_loader)

    for images, coords, shapes in loop:

        images = images.to(device)

        coords = coords.to(device)

        shapes = shapes.to(device)

        pred_coords, pred_shapes = model(
            images
        )

        coord_loss = coord_loss_fn(
            pred_coords,
            coords
        )

        shape_loss = shape_loss_fn(
            pred_shapes,
            shapes
        )

        loss = (
            coord_loss
            + 0.5 * shape_loss
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        loop.set_description(
            f"Epoch {epoch+1}"
        )

        loop.set_postfix(
            loss=loss.item()
        )

    train_loss /= len(train_loader)

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for (
            images,
            coords,
            shapes
        ) in val_loader:

            images = images.to(device)

            coords = coords.to(device)

            shapes = shapes.to(device)

            pred_coords, pred_shapes = model(
                images
            )

            coord_loss = coord_loss_fn(
                pred_coords,
                coords
            )

            shape_loss = shape_loss_fn(
                pred_shapes,
                shapes
            )

            loss = (
                coord_loss
                + 0.5 * shape_loss
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train={train_loss:.4f} | "
        f"Val={val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("Saved Best Model")