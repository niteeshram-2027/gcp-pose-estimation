from dataset import GCPDataset
from torch.utils.data import DataLoader

dataset = GCPDataset(
    "../gcp_marks.json",
    "../data/train_dataset"
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

images, coords, shapes = next(iter(loader))

print(images.shape)
print(coords.shape)
print(shapes.shape)