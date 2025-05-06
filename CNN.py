# train_cnn_waypoint.py
import cv2, os, glob, math, random, torch, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------- 1. dataset ----------
class WaypointSet(Dataset):
    def __init__(self, npz_root):
        self.files = sorted(glob.glob(f'{npz_root}/*.npz'))
        self.tf = transforms.Compose([
            transforms.ToTensor(),                      # HWC [0,255] → CHW [0,1]
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10,
                                    translate=(0.05,0.05)),
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        img = d['img'].astype(np.uint8)        # H×W
        hm  = d['hm'].astype(np.float32)       # H×W in [0,1]
        img = self.tf(img)
        hm  = torch.from_numpy(np.expand_dims(hm,0))
        return img, hm

# ---------- 2. model ----------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(ch_in,ch_out):
            return nn.Sequential(
                nn.Conv2d(ch_in,ch_out,3,1,1), nn.BatchNorm2d(ch_out), nn.ReLU(),
                nn.Conv2d(ch_out,ch_out,3,1,1), nn.BatchNorm2d(ch_out), nn.ReLU())
        self.enc1 = blk(1,32);  self.pool1 = nn.MaxPool2d(2)
        self.enc2 = blk(32,64); self.pool2 = nn.MaxPool2d(2)
        self.bott = blk(64,128)
        self.up2  = nn.ConvTranspose2d(128,64,2,2)
        self.dec2 = blk(128,64)
        self.up1  = nn.ConvTranspose2d(64,32,2,2)
        self.dec1 = blk(64,32)
        self.outc = nn.Conv2d(32,1,1)

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bott(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2],1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1],1))
        return self.outc(d1)                    # logits

# ---------- 3. train loop ----------
def train(root='data_npz', epochs=25):
    ds   = WaypointSet(root)
    dl   = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
    net  = UNet().cuda()
    opt  = torch.optim.Adam(net.parameters(), 3e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.]).cuda())

    for ep in range(epochs):
        net.train(); running=0
        for img,hm in dl:
            img,hm = img.cuda().float(), hm.cuda()
            opt.zero_grad()
            out = net(img)
            loss= crit(out, hm)
            loss.backward(); opt.step()
            running += loss.item()*img.size(0)
        print(f'Epoch {ep+1}: loss {running/len(ds):.4f}')
        torch.save(net.state_dict(), f'unet_ep{ep+1:02d}.pth')

if __name__ == '__main__':
    train('waypoint_npz')
