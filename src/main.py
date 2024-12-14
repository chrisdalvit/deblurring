import argparse
import torch
from model import DDANet
from utils import loss_fn
from data import train_dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int)
parser.add_argument("--lr_start", type=float, default=1e-4)
parser.add_argument("--lr_min", type=float, default=1e-6)

def main():
    args = parser.parse_args()    
    test_dataloader = train_dataloader("./data/GOPRO", batch_size=4)
    dda = DDANet(
        in_channels=3,
        hid_channels=32,
        kernel_size=3,
        sam_groups=8,
        attention_size=3
    )
    optimizer = torch.optim.Adam(dda.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(test_dataloader), eta_min=args.min_lr)
    dda.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for X, y in test_dataloader:
            outputs = dda(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        schedule.step()
        print(loss / len(test_dataloader))