#!/usr/bin/env python3
"""
Create (image, heat‑map) .npz pairs from numbered JPEG frames WITHOUT MediaPipe.
Assumptions:
  • Pen tip (marker cap) covered by a distinct neon sticker.
  • Sticker HSV ranges specified via --hsv "Hmin,Hmax,Smin,Smax,Vmin,Vmax".
  • Waypoint for frame t is the centroid detected in frame t+1.
Usage:
  python frames2npz_no_mediapipe.py \
      --src "/path/combined_frames" \
      --dst npz_pairs \
      --hsv "40,85,100,255,100,255"       # neon‑green example
"""
import cv2, os, glob, numpy as np, argparse, tqdm

def gaussian(shape, center, sigma=5):
    y,x = np.ogrid[:shape[0], :shape[1]]
    return np.exp(-((x-center[0])**2 + (y-center[1])**2)/(2*sigma**2))

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument('--src',  required=True)
ap.add_argument('--dst',  required=True)
ap.add_argument('--hsv',  required=True,
                help='comma‑sep 6 values Hmin,Hmax,Smin,Smax,Vmin,Vmax')
ap.add_argument('--sigma', type=int, default=5)
args = ap.parse_args(); os.makedirs(args.dst, exist_ok=True)
hmin,hmax,smin,smax,vmin,vmax = map(int, args.hsv.split(','))

files = sorted(glob.glob(os.path.join(args.src,'frame_*.jp*g')))
if len(files) < 3: raise RuntimeError('Need >=3 frames')

centroids = []
for f in tqdm.tqdm(files, desc='detect'):
    bgr = cv2.imread(f)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask= cv2.inRange(hsv, (hmin,smin,vmin), (hmax,smax,vmax))
    cnts,_= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        centroids.append(None); continue
    c   = max(cnts, key=cv2.contourArea)
    M   = cv2.moments(c)
    if M['m00'] == 0:
        centroids.append(None); continue
    u = int(M['m10']/M['m00']); v = int(M['m01']/M['m00'])
    centroids.append((u,v))

# ---------- build npz ----------
valid = 0
for i in tqdm.tqdm(range(len(files)-1), desc='save'):
    if centroids[i] is None or centroids[i+1] is None:
        continue
    img  = cv2.imread(files[i])                   # keep BGR
    u,v  = centroids[i+1]                        # look‑ahead waypoint
    hm   = gaussian(img.shape[:2], (u,v), sigma=args.sigma).astype(np.float32)
    out  = os.path.join(args.dst, f'{i:05d}.npz')
    np.savez_compressed(out, img=img, hm=hm)
    valid += 1
print('Done. Saved', valid, 'labelled frames →', args.dst)
