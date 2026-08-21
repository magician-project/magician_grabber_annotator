#!/usr/bin/python3

""" 
Author : "Ammar Qammaz"
Copyright : "2022 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 
"""

import sys
import numpy as np
import cv2




def get_png_comment(filename):
    """Extract 'Comment' metadata from a PNG file using Pillow."""
    from PIL import Image
    try:
        img = Image.open(filename)
        comment = img.info.get("Comment", "")
        return comment
    except Exception as e:
        print(f"Warning: Could not read comment from {filename}: {e}")
        return ""

def check_threshold(array, threshold):
    """Return a binary mask where pixels exceed threshold in any channel, plus count."""
    mask = np.any(array > threshold, axis=-1)
    count = int(np.sum(mask))  # number of pixels above threshold
    return (mask.astype(np.uint8) * 255), count


def check_threshold_count(array, threshold):
    """Return a binary mask where pixels exceed threshold in any channel, plus count."""
    mask = np.any(array > threshold)
    count = int(np.sum(mask))  # number of pixels above threshold
    return count

def check_variation(tile, threshold):
    """Return a single-channel map where variation above threshold is highlighted."""
    std_dev = np.std(tile, axis=-1)  # std per pixel across channels
    mask = (std_dev > threshold).astype(np.uint8) * 255
    return mask


def sobel_edges(array):
    """Apply Sobel filter to each channel and sum magnitudes."""
    edges = []
    for c in range(array.shape[2]):
        ch = array[:, :, c].astype(np.float32)
        sx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)
        edges.append(mag)
    edge_map = np.sum(edges, axis=0)
    edge_map = (255 * edge_map / np.max(edge_map)).astype(np.uint8)
    return edge_map


def sum_channels(array):
    """Return grayscale image representing sum across channels."""
    summed = np.sum(array, axis=-1)
    summed = (255 * summed / np.max(summed)).astype(np.uint8)
    return summed



def make_comment_panel(text, height, width=550, bg_color=(30, 30, 30), text_color=(0, 255, 0)):
    """
    Create a small CV Mat to display the comment text.
    The panel will have the same height as the input image.
    """
    panel = np.full((height, width, 3), bg_color, dtype=np.uint8)

    if not text:
        text = "(No comment)"

    # Wrap text if it's too long
    words = text.split()
    lines = []
    line = ""
    for word in words:
        if len(line + word) < 35:
            line += word + " "
        else:
            lines.append(line)
            line = word + " "
    lines.append(line)

    y = 3
    for ln in lines:
        cv2.putText(panel, ln.strip(), (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)
        y += 18

    return panel


def process_image(filename, threshold=128, var_threshold=5.0):
    # Load PNM (with 4 channels if available)
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)  # preserves channels
    if img is None:
        raise ValueError(f"Could not load {filename}")

    img = img.astype(np.float32)

    # Run functions
    threshold_map, threshold_count = check_threshold(img, threshold)
    threshold_count = check_threshold_count(img, threshold)
    variation_map = check_variation(img, var_threshold)
    sobel_map  = sobel_edges(img)
    summed_map = sum_channels(img)

    # Normalize input for visualization (ignore alpha if 4 channels)
    input_img = (255 * (img / np.max(img))).astype(np.uint8)
    if input_img.shape[2] == 4:
        input_img = input_img[:, :, :3]  # drop alpha

    # Convert masks to BGR so we can concat
    def to_bgr(x):
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)

    threshold_bgr = to_bgr(threshold_map)
    variation_bgr = to_bgr(variation_map)
    sobel_bgr     = to_bgr(sobel_map)
    summed_bgr    = to_bgr(summed_map)

    # Overlay threshold result as text
    text = f"{threshold_count}"
    cv2.putText( threshold_bgr, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA )


    # Get PNG comment if available
    comment_text = get_png_comment(filename)
    comment_panel = make_comment_panel(comment_text, input_img.shape[0]) 

    # Concatenate horizontally
    combined = np.concatenate(
        [input_img, threshold_bgr, variation_bgr, sobel_bgr, summed_bgr, comment_panel], axis=1
    )

    return combined


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_pnm.py input1.pnm input2.pnm ...")
        sys.exit(1)

    output_file = "examination.png"
    input_files = sys.argv[1:]

    results = [process_image(f) for f in input_files]
    combined_vertical = np.concatenate(results, axis=0)

    cv2.imwrite(output_file, combined_vertical)
    print(f"Saved result to {output_file}")

