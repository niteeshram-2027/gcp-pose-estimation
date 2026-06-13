import torch

from src.model import GCPModel

model = GCPModel()

dummy = torch.randn(
    2,
    3,
    1024,
    1024
)

coords, shape_logits = model(dummy)

print("Coords Shape:")
print(coords.shape)

print()

print("Shape Logits Shape:")
print(shape_logits.shape)