import torch

from model import GCPModel

model = GCPModel()

dummy = torch.randn(
    1,
    3,
    1024,
    1024
)

coords, logits = model(dummy)

print(coords.shape)
print(logits.shape)