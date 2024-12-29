import argparse
import time
import torch
import numpy as np
from model import DDANet
from utils import loss_fn
from data import get_train_dataloader, get_test_dataloader
from skimage.metrics import peak_signal_noise_ratio

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--lr_start", type=float, default=1e-4)
parser.add_argument("--lr_min", type=float, default=1e-6)
parser.add_argument("--save", action='store_true')

def main():
    start = time.time()
    args = parser.parse_args()   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    train_loader = get_train_dataloader("./data/GOPRO", batch_size=4)
    test_loader = get_test_dataloader("./data/GOPRO", batch_size=1, num_workers=0)
    dda = DDANet(
        in_channels=3,
        hid_channels=32,
        kernel_size=3
    ).to(device)
    optimizer = torch.optim.Adam(dda.parameters(), lr=args.lr_start)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=args.lr_min)
    dda.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = dda(X.to(device))
            outputs = tuple(x.to(device) for x in outputs)
            loss = loss_fn(outputs, y.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        schedule.step()
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Avg train loss: {avg_epoch_loss.item()}")
        
    with torch.no_grad():
        dda.eval()
        psnrs = []
        for X, y, _ in test_loader:
            outputs = dda(X.to(device))
            outputs = tuple(x.to(device) for x in outputs)
            preds = outputs[0].cpu().numpy()
            labels = y.cpu().numpy()
            psnr = peak_signal_noise_ratio(labels, preds, data_range=1) 
            psnrs.append(psnr)
        print('The average PSNR is %.2f dB' % (np.mean(psnrs)))
    end = time.time()
    print(f"Duration: {(end-start) / 60 } minutes")
    
    if args.save:
        torch.save({
            'epoch': args.epochs,
            'model_state_dict': dda.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f"{args.name}.pt"
        )
        
if __name__ == "__main__":
    main()
