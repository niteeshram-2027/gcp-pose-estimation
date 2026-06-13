from src.dataset import GCPDataset

dataset = GCPDataset(
    json_path="gcp_marks.json",
    image_root="data/train_dataset"
)

print("Dataset Size:", len(dataset))

image, coords, shape = dataset[0]

print("Image Shape:", image.shape)
print("Coordinates:", coords)
print("Shape:", shape)

for i in range(5):

    image, coords, shape = dataset[i]

    print()
    print(image.shape)
    print(coords)
    print(shape)