#!/usr/bin/env python3
import cv2, numpy as np, argparse, os

ap = argparse.ArgumentParser()
ap.add_argument('--img', required=True)
args = ap.parse_args()

orig = cv2.imread(args.img)
if orig is None:
    raise FileNotFoundError(args.img)
hsv  = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)

def nothing(x): pass
cv2.namedWindow('mask')
# trackbars:  H 0‑179, S 0‑255, V 0‑255
cv2.createTrackbar('Hmin','mask', 0,179,nothing)
cv2.createTrackbar('Hmax','mask',179,179,nothing)
cv2.createTrackbar('Smin','mask', 0,255,nothing)
cv2.createTrackbar('Smax','mask',255,255,nothing)
cv2.createTrackbar('Vmin','mask', 0,255,nothing)
cv2.createTrackbar('Vmax','mask',255,255,nothing)

print("\nAdjust sliders until ONLY the sticker is white.\nPress 's' to save values, 'q' to quit.")
values = None
while True:
    hmin = cv2.getTrackbarPos('Hmin','mask')
    hmax = cv2.getTrackbarPos('Hmax','mask')
    smin = cv2.getTrackbarPos('Smin','mask')
    smax = cv2.getTrackbarPos('Smax','mask')
    vmin = cv2.getTrackbarPos('Vmin','mask')
    vmax = cv2.getTrackbarPos('Vmax','mask')

    mask = cv2.inRange(hsv, (hmin,smin,vmin), (hmax,smax,vmax))
    cv2.imshow('mask', mask)

    k = cv2.waitKey(1)&0xFF
    if k == ord('s'):
        values = (hmin,hmax,smin,smax,vmin,vmax)
        break
    elif k == ord('q'):
        break
cv2.destroyAllWindows()
if values:
    print("\nUse this --hsv string:\n\"%d,%d,%d,%d,%d,%d\"" % values)
