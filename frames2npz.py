#!/usr/bin/env python3
"""
Create cnn‑trainable .npz pairs from a folder of numbered JPEGs:
  frame_0001.jpeg, frame_0002.jpeg, ...
  • Uses MediaPipe Hands to locate index‑finger tip
  • Uses (tip[t+1]) as the "next waypoint" for frame t
Output: dst/00001.npz  with  keys: img (H×W×3 uint8), hm (H×W float32)
"""
import cv2, os, glob, numpy as np, mediapipe as mp, tqdm, argparse

def gaussian(shape, center, sigma=5):
    y,x = np.ogrid[:shape[0], :shape[1]]
    return np.exp(-((x-center[0])**2 + (y-center[1])**2)/(2*sigma**2))

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True, help='folder with frame_*.jpeg')
ap.add_argument('--dst', required=True, help='output folder for .npz')
ap.add_argument('--sigma', type=int, default=5)
args = ap.parse_args(); os.makedirs(args.dst, exist_ok=True)

files = sorted(glob.glob(os.path.join(args.src,'frame_*.jp*g')))

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6)


tip_px = []   # (u,v) list per frame
for f in tqdm.tqdm(files, desc='detect'):
    bgr = cv2.imread(f); h,w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = mp_hands.process(rgb)
    if not res.multi_hand_landmarks:
        tip_px.append(None); continue
    pt = res.multi_hand_landmarks[0].landmark[8]   # index‑finger tip
    tip_px.append((int(pt.x*w), int(pt.y*h)))

# second pass: build npz
valid = 0
for i in tqdm.tqdm(range(len(files)-1), desc='save'):
    img = cv2.imread(files[i])
    if tip_px[i] is None or tip_px[i+1] is None:
        continue
    u,v   = tip_px[i+1]                     # waypoint = tip in next frame
    hm    = gaussian(img.shape[:2], (u,v), sigma=args.sigma).astype(np.float32)
    out   = os.path.join(args.dst, f'{i:05d}.npz')
    np.savez_compressed(out, img=img, hm=hm)
    valid += 1

print(f'Wrote {valid} labelled frames to {args.dst}')
