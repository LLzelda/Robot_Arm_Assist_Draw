#!/usr/bin/env python3
"""
Train a tiny U‑Net to predict the pen‑tip’s next‑waypoint heat‑map.
Assumes your training pairs are stored as *.npz files with keys:
    img :  H×W×3  uint8   (BGR or RGB is fine)
    hm  :  H×W     float32 (0‑1 heat‑map)
"""

import cv2, glob, os, argparse, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ---------- command‑line ----------
ap = argparse.ArgumentParser()
ap.add_argument('--root',   required=True, help='folder with *.npz pairs')
ap.add_argument('--epochs', type=int, default=25)
ap.add_argument('--batch',  type=int, default=16)
args = ap.parse_args()

# ---------- dataset ----------
class WaypointSet(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(f'{root}/*.npz'))
        self.tf_img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),  # RGB → 1‑channel
            transforms.ToTensor(),                       # → 1×H×W float
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10,
                                    translate=(0.05, 0.05)),
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        d   = np.load(self.files[idx])
        img = d['img']                              # H×W×3  uint8
        hm  = d['hm'].astype(np.float32)            # H×W     float32
        img = self.tf_img(img)                      # 1×H×W   float32 [0,1]
        hm  = torch.from_numpy(hm).unsqueeze(0)     # 1×H×W
        return img, hm

# ---------- U‑Net ----------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci,co,3,1,1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.Conv2d(co,co,3,1,1), nn.BatchNorm2d(co), nn.ReLU())
        self.e1 = blk(1,32);  self.p1 = nn.MaxPool2d(2)
        self.e2 = blk(32,64); self.p2 = nn.MaxPool2d(2)
        self.b  = blk(64,128)
        self.u2 = nn.ConvTranspose2d(128,64,2,2); self.d2 = blk(128,64)
        self.u1 = nn.ConvTranspose2d(64,32,2,2);  self.d1 = blk(64,32)
        self.o  = nn.Conv2d(32,1,1)

    def forward(self,x):
        e1=self.e1(x)
        e2=self.e2(self.p1(e1))
        b = self.b(self.p2(e2))
        d2=self.d2(torch.cat([self.u2(b),e2],1))
        d1=self.d1(torch.cat([self.u1(d2),e1],1))
        return self.o(d1)        # logits

# ---------- train ----------
ds  = WaypointSet(args.root)
dl  = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
# net = UNet().cuda()
# opt = torch.optim.Adam(net.parameters(), 3e-4)
# crit= nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.]).cuda())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net  = UNet().to(device)
opt  = torch.optim.Adam(net.parameters(), 3e-4)
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.], device=device))

for ep in range(args.epochs):
    net.train()
    epoch_loss = 0
    for img, hm in tqdm(dl, desc=f'Epoch {ep+1}/{args.epochs}'):   # <‑‑ wrap here
        img, hm = img.to(device), hm.to(device)
        opt.zero_grad()
        loss = crit(net(img), hm)
        loss.backward(); opt.step()
        epoch_loss += loss.item() * img.size(0)
    print(f'\nEpoch {ep+1}/{args.epochs}  loss {epoch_loss/len(ds):.4f}')
    torch.save(net.state_dict(), f'unet_ep{ep+1:02d}.pth')

