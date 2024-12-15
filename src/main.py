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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    test_dataloader = train_dataloader("./data/GOPRO", batch_size=4)
    dda = DDANet(
        in_channels=3,
        hid_channels=32,
        kernel_size=3,
        sam_groups=8,
        attention_size=3
    ).to(device)
    optimizer = torch.optim.Adam(dda.parameters(), lr=args.lr_start)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(test_dataloader), eta_min=args.lr_min)
    dda.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for X, y in test_dataloader:
            outputs = dda(X.to(device))
            outputs = tuple(x.to(device) for x in outputs)
            loss = loss_fn(outputs, y.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        schedule.step()
        avg_epoch_loss = epoch_loss / len(test_dataloader)
        print(f"Avg epoch loss: {avg_epoch_loss.item()}")
        
if __name__ == "__main__":
    main()
