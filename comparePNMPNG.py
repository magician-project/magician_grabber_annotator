#!/usr/bin/env python3

import sys
import cv2
import numpy as np

# Import function from readData.py
from readData import readPolarPNMToRGBALive


def load_and_convert(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return readPolarPNMToRGBALive(img)


def compare_images(pnm_path, png_path):
    rgba_pnm = load_and_convert(pnm_path)
    rgba_png = load_and_convert(png_path)

    if rgba_pnm.shape != rgba_png.shape:
        print("FAIL")
        print(f"Shape mismatch: {rgba_pnm.shape} vs {rgba_png.shape}")
        return False

    if rgba_pnm.dtype != rgba_png.dtype:
        print("FAIL")
        print(f"Dtype mismatch: {rgba_pnm.dtype} vs {rgba_png.dtype}")
        return False

    if np.array_equal(rgba_pnm, rgba_png):
        print("OK")
        return True
    else:
        diff = np.abs(rgba_pnm.astype(np.int32) - rgba_png.astype(np.int32))
        max_diff = np.max(diff)
        print("FAIL")
        print(f"Max pixel difference: {max_diff}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("  compare_polar_images.py input.pnm input.png")
        sys.exit(2)

    pnm_path = sys.argv[1]
    png_path = sys.argv[2]

    success = compare_images(pnm_path, png_path)
    sys.exit(0 if success else 1)
