import os
import glob
import argparse
import numpy as np
import cv2
from readData import readPolarPNMToRGBA, averagePolarRGBAtoGray
# ---------------------- Feature utils ---------------------- #
def make_feature_extractor():
    if hasattr(cv2, "SIFT_create"):
        sift = cv2.SIFT_create(nfeatures=8000, nOctaveLayers=5, contrastThreshold=0.04, edgeThreshold=25 )
        return sift, cv2.NORM_L2, "SIFT"
    else:
        orb = cv2.ORB_create(nfeatures=8000)
        return orb, cv2.NORM_HAMMING, "ORB"

def detect_and_describe(detector, img):
    kps, desc = detector.detectAndCompute(img, None)
    if kps is None:
        kps = []
    return np.array([kp.pt for kp in kps], dtype=np.float32), desc, kps

def ratio_test_matcher(descA, descB, normType, ratio=0.75):
    if descA is None or descB is None or len(descA) == 0 or len(descB) == 0:
        return []

    bf = cv2.BFMatcher(normType, crossCheck=False)
    knn = bf.knnMatch(descA, descB, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            # Skip this keypoint — not enough matches
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def estimate_homography(ptsA, ptsB, matches, ransac_thresh=4.0, maxIters=4000, conf=0.999):
    if len(matches) < 8:
        return None, None, np.inf, 0
    src = np.float32([ptsA[m.queryIdx] for m in matches])
    dst = np.float32([ptsB[m.trainIdx] for m in matches])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh, maxIters=maxIters, confidence=conf)
    if H is None:
        return None, None, np.inf, 0
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 8:
        return None, None, np.inf, 0
    src_in = src[inliers]
    dst_in = dst[inliers]
    proj = cv2.perspectiveTransform(src_in.reshape(-1,1,2), H).reshape(-1,2)
    err = np.sqrt(np.sum((proj - dst_in)**2, axis=1)).mean()
    return H, inliers, float(err), int(inliers.sum())

# ---------------------- Visualization ---------------------- #
def visualize_sift_keypoints(img, kps, win_name="SIFT Keypoints"):
    vis = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow(win_name, vis)
    cv2.waitKey(1)

def visualize_matches(img1, img2, kps1, kps2, matches, inlier_mask=None, win_name="Matches"):
    if inlier_mask is not None:
        # Convert mask safely to a Python list of ints
        matchesMask = [int(x) for x in inlier_mask.ravel().tolist()]
    else:
        matchesMask = None

    vis = cv2.drawMatches(
        img1, kps1,
        img2, kps2,
        matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imshow(win_name, vis)
    cv2.waitKey(1)

# ---------------------- Panorama utils ---------------------- #
def warp_corners(size, H):
    h, w = size
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1,2)
    return warped

def translate_H(H, tx, ty):
    T = np.array([[1,0,-tx],[0,1,-ty],[0,0,1]], dtype=np.float64)
    return T @ H

def compute_panorama_bounds(images, H_to_0):
    all_pts = []
    for img, H in zip(images, H_to_0):
        pts = warp_corners(img.shape[:2], H)
        all_pts.append(pts)
    all_pts = np.vstack(all_pts)
    min_x, min_y = np.floor(all_pts.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_pts.max(axis=0)).astype(int)
    return (min_x, min_y, max_x, max_y)

def feather_blend(dst, mask, src):
    binary = (src > 0).astype(np.uint8)
    if binary.any():
        inv = 1 - binary
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
        w = dist / (dist.max() + 1e-6)
        w = np.clip(w, 0.05, 1.0)
        dst += src * w
        mask += w
    return dst, mask

# ---------------------- Stitching core ---------------------- #
def collect_images(input_dir):
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.pnm","*.tif","*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(input_dir, e)))
    files = sorted(files)
    imgs = []
    for f in files:
        #im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        im = averagePolarRGBAtoGray(readPolarPNMToRGBA(f))
        if im is None:
            continue
        imgs.append(im)
    if not imgs:
        raise RuntimeError("No readable images.")
    return imgs, files

def build_best_edges(images, next_x, ransac_thresh, feature_name, vis=False):
    detector, normType, _ = make_feature_extractor()
    keypoints = []
    descriptors = []
    kps_full = []

    print(f"[INFO] Detecting {feature_name} features...")
    for idx, img in enumerate(images):
        pts, desc, kps = detect_and_describe(detector, img)
        keypoints.append(pts)
        descriptors.append(desc)
        kps_full.append(kps)
        print(f"  Frame {idx}: {len(pts)} keypoints")
        if vis:
            visualize_sift_keypoints(img, kps, f"Keypoints Frame") #  {idx}

    n = len(images)
    best_edges = {i: None for i in range(n)}

    for i in range(n-1):
        best = None
        for j in range(i+1, min(n, i+1+next_x)):
            matches = ratio_test_matcher(descriptors[i], descriptors[j], normType)
            H, inliers, err, ninl = estimate_homography(keypoints[i], keypoints[j], matches)
            if vis:
                visualize_matches(images[i], images[j], kps_full[i], kps_full[j], matches, inliers, f"Matches") #  {i}->{j}
            if H is None:
                continue
            score = (ninl, -err)
            if (best is None) or (score > best["score"]):
                best = {"j": j, "H": H, "ninl": ninl, "err": err, "score": score}
        if best is not None:
            best_edges[i] = (best["j"], best["H"])
            print(f"  Edge {i}->{best['j']}: inliers={best['ninl']}  err={best['err']:.3f}")
        else:
            print(f"  Edge {i}->(none): insufficient matches")

    return best_edges

def compute_global_homographies(best_edges, n):
    H_to_0 = [None]*n
    H_to_0[0] = np.eye(3)
    for i in range(1, n):
        if best_edges[i-1] is not None:
            j, H = best_edges[i-1]
            H_to_0[i] = H_to_0[i-1] @ H
        else:
            H_to_0[i] = H_to_0[i-1]
    return H_to_0

def stitch_sequence(input_dir, next_x=3, ransac_thresh=4.0, out_path="panorama.png", vis=False):
    images, files = collect_images(input_dir)
    print(f"[INFO] Loaded {len(images)} frames.")
    _, _, feature_name = make_feature_extractor()
    best_edges = build_best_edges(images, next_x, ransac_thresh, feature_name, vis)

    print("[INFO] Computing global homographies to reference (frame 0)...")
    H_to_0 = compute_global_homographies(best_edges, len(images))

    print("[INFO] Estimating panorama bounds...")
    min_x, min_y, max_x, max_y = compute_panorama_bounds(images, H_to_0)
    width  = int(max_x - min_x + 1)
    height = int(max_y - min_y + 1)
    print(f"[INFO] Panorama size: {width} x {height}")
    H_adj = [translate_H(H, min_x, min_y) for H in H_to_0]

    acc = np.zeros((height, width), dtype=np.float32)
    wts = np.zeros((height, width), dtype=np.float32)

    print("[INFO] Warping and blending frames...")
    for idx, (img, H) in enumerate(zip(images, H_adj)):
        warped = cv2.warpPerspective(img, H, (width, height))
        warped_f = warped.astype(np.float32)/255.0
        acc, wts = feather_blend(acc, wts, warped_f)
        print(f"  Blended frame {idx} ({os.path.basename(files[idx])})")

    pano = (acc / (wts + 1e-6))
    pano = np.clip(pano*255, 0, 255).astype(np.uint8)
    pano = cv2.medianBlur(pano, 3)

    if not (out_path.lower().endswith(".png") or out_path.lower().endswith(".jpg") or out_path.lower().endswith(".jpeg")):
      out_path = out_path + ".png"


    cv2.imwrite(out_path, pano)
    print(f"[DONE] Saved panorama to {out_path}")

    if vis:
        cv2.imshow("Final Panorama", pano)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ---------------------- CLI ---------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stitch a sequence of grayscale frames with SIFT visualization.")
    ap.add_argument("--input", required=True, help="Folder with grayscale frames")
    ap.add_argument("--next_x", type=int, default=3, help="How many next frames to compare")
    ap.add_argument("--ransac_thresh", type=float, default=4.0, help="RANSAC reprojection threshold")
    ap.add_argument("--out", default="panorama.png", help="Output panorama filename")
    ap.add_argument("--vis", action="store_true", help="Enable SIFT and match visualization")
    args = ap.parse_args()

    stitch_sequence(args.input, args.next_x, args.ransac_thresh, args.out, args.vis)

