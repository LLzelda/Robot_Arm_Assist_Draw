"""
Train the waypoint-prediction U-Net.

Two dataset modes
-----------------
1.  Pairs mode (old):  --pairs  <dir_with_00000.npz …>
    npz must contain keys  {img: H×W×3 uint8,  hm: H×W float32}

2.  Raw-image mode (new):
    --imgs <image_dir>  --hms <heatmap_dir>
    - Every image file must have a matching heat-map file
      with the *same stem*:
         frame_0123.jpeg  →  frame_0123.npy   (or .npz / .png)
    - Accepted heat-map formats
         • .npy  (H×W float32)
         • .npz  (stored under key 'hm')
         • .png  (8-bit, scaled 0-255)        <-- easiest to inspect
"""

import argparse, glob, os, cv2, numpy as np, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ---------- cmd ----------
ap = argparse.ArgumentParser()
grp = ap.add_mutually_exclusive_group(required=True)
grp.add_argument('--pairs', help='folder with *.npz pairs')
grp.add_argument('--imgs',  help='folder with raw JPEG/PNG')
ap.add_argument('--hms', help='folder with heat-maps (only with --imgs)')
ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--batch',  type=int, default=16)
ap.add_argument('--workers',type=int, default=0)
args = ap.parse_args()

# ---------- data ----------
to_gray = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(1),
    transforms.Resize((480,640)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=8, translate=(0.04,0.04)),
])

def load_heat(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path).astype(np.float32)
    if ext == '.npz':
        return np.load(path)['hm'].astype(np.float32)
    if ext in ('.png', '.jpg', '.jpeg'):
        hm = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        return hm / 255.0                    # scale to 0-1
    raise ValueError(f'Unsupported heat-map {path}')

class PairSet(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(f'{root}/*.npz'))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        img = to_gray(d['img'])
        hm  = torch.from_numpy(d['hm']).unsqueeze(0)
        return img, hm

class RawSet(Dataset):
    def __init__(self, img_dir, hm_dir):
        self.imgs = sorted(glob.glob(f'{img_dir}/*.*g'))
        self.hm_dir = hm_dir
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        hm_path = glob.glob(os.path.join(self.hm_dir, stem + '.*'))[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        hm  = load_heat(hm_path)
        img = to_gray(img)
        hm  = torch.from_numpy(hm).unsqueeze(0)
        return img, hm

if args.pairs:
    ds = PairSet(args.pairs)
else:
    if not args.hms:
        ap.error('--imgs needs --hms')
    ds = RawSet(args.imgs, args.hms)

# ---------- model ----------
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
        return self.o(d1)

# ---------- train ----------
dl = DataLoader(ds,
                batch_size=args.batch,
                shuffle=True,
                num_workers=args.workers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net  = UNet().to(device)
opt  = torch.optim.Adam(net.parameters(), 3e-4)
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.], device=device))

for ep in range(args.epochs):
    net.train(); epoch_loss = 0.0
    for img, hm in tqdm(dl, desc=f'Epoch {ep+1}/{args.epochs}'):
        img, hm = img.to(device), hm.to(device)
        opt.zero_grad()
        loss = crit(net(img), hm)
        loss.backward(); opt.step()
        epoch_loss += loss.item()*img.size(0)
    print(f'\nEpoch {ep+1}/{args.epochs}  loss {epoch_loss/len(ds):.4f}')
    torch.save(net.state_dict(), f'unet_ep{ep+1:02d}.pth')
