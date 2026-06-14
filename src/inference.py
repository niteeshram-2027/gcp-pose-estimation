import json
import cv2
import torch

from pathlib import Path

from model import GCPModel


TEST_ROOT = "/content/drive/MyDrive/skylark_gcp/test_dataset"

MODEL_PATH = "best_model.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

shape_map = {
    0: "Cross",
    1: "Square",
    2: "L-Shape"
}


model = GCPModel().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

results = {}

all_images = sorted(
    list(Path(TEST_ROOT).rglob("*.JPG"))
)

print("Total Images:", len(all_images))

for img_path in all_images:

    image = cv2.imread(str(img_path))

    h, w = image.shape[:2]

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image_resized = cv2.resize(
        image_rgb,
        (768, 768)
    )

    tensor = (
        torch.tensor(image_resized)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        coords, shape_logits = model(
            tensor
        )

    x_norm = coords[0][0].item()
    y_norm = coords[0][1].item()

    x_pixel = x_norm * w
    y_pixel = y_norm * h

    shape_idx = torch.argmax(
        shape_logits,
        dim=1
    ).item()

    shape_name = shape_map[
        shape_idx
    ]

    relative_path = str(
        img_path.relative_to(TEST_ROOT)
    ).replace("\\", "/")

    results[relative_path] = {
        "mark": {
            "x": float(x_pixel),
            "y": float(y_pixel)
        },
        "verified_shape": shape_name
    }

with open(
    "predictions.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print("Saved predictions.json")