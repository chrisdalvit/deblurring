import torch
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import DDANet
from utils import get_transforms, loss_fn
from data import train_dataloader

img = Image.open("./data/sample.jpg")

transforms = get_transforms()
X = transforms(img).unsqueeze(0)

dda = DDANet(
    in_channels=3,
    hid_channels=32,
    kernel_size=3,
    sam_groups=4,
    attention_size=3
)

lr = 1e-4
min_lr = 1e-6
epochs = 2
test_dataloader = train_dataloader("./data/GOPRO", batch_size=4)
optimizer = torch.optim.Adam(dda.parameters(), lr=lr)
schedule = CosineAnnealingLR(optimizer, len(test_dataloader), eta_min=min_lr)

dda.train()
for epoch in range(epochs):
    for X, y in test_dataloader:
        outputs = dda(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        print(loss)
    schedule.step()