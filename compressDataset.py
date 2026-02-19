import os
import sys
import shutil
import cv2
import numpy as np


# --- Debayer ---
def debayerPolarImage(image):
    polarization_90_deg   = image[0::2, 0::2]
    polarization_45_deg   = image[0::2, 1::2]
    polarization_135_deg  = image[1::2, 0::2]
    polarization_0_deg    = image[1::2, 1::2]
    return polarization_0_deg, polarization_45_deg, polarization_90_deg, polarization_135_deg


# --- Write PNG ---
def write_polar_png_from_pnm(pnm_path: str, out_png_path: str):
    img = cv2.imread(pnm_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read: {pnm_path}")
    if img.ndim != 2:
        raise ValueError(f"Expected 1-channel image, got shape {img.shape} for {pnm_path}")

    p0, p45, p90, p135 = debayerPolarImage(img)
    rgba = np.stack([p0, p45, p90, p135], axis=-1)

    os.makedirs(os.path.dirname(out_png_path) or ".", exist_ok=True)
    ok = cv2.imwrite(out_png_path, rgba)
    if not ok:
        raise IOError(f"Failed to write: {out_png_path}")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_tree_and_convert_pnm(input_dir: str, output_dir: str) -> int:
    """
    Walks input_dir recursively:
      - Converts *.pnm -> *.png (same relative path) into output_dir
      - Copies all other files as-is (same relative path) into output_dir
      - Copies subdirectories structure
      - Skips copying *.pnm (because converted)
      - Skips *.pnm.json being treated as pnm (it will be copied as a normal file)
    Returns: number of converted pnm files
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    converted = 0

    for root, dirs, files in os.walk(input_dir):
        rel_root = os.path.relpath(root, input_dir)
        out_root = output_dir if rel_root == "." else os.path.join(output_dir, rel_root)
        ensure_dir(out_root)

        # Ensure directories exist in output (os.walk already gives dirs)
        for d in dirs:
            ensure_dir(os.path.join(out_root, d))

        for fname in files:
            in_path = os.path.join(root, fname)
            out_path = os.path.join(out_root, fname)

            # Convert only files that end EXACTLY with ".pnm"
            if fname.lower().endswith(".pnm"):
                out_png = os.path.join(out_root, os.path.splitext(fname)[0] + ".png")
                write_polar_png_from_pnm(in_path, out_png)
                print(f"Converted: {os.path.join(rel_root, fname)} -> {os.path.join(rel_root, os.path.basename(out_png))}")
                converted += 1
                continue

            # Otherwise copy as-is (json/csv/whatever, including *.pnm.json)
            shutil.copy2(in_path, out_path)
            # Optional: print copies (can be noisy)
            # print(f"Copied   : {os.path.join(rel_root, fname)}")

    return converted


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python convert.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) >= 3 else None

    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory")
        sys.exit(1)

    if output_dir is None:
        print("Error: output_directory is required for safe copying (to avoid clobbering).")
        print("Usage: python convert.py <input_directory> <output_directory>")
        sys.exit(1)

    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        print("Error: input_directory and output_directory must be different when copying the whole tree.")
        sys.exit(1)

    ensure_dir(output_dir)

    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")

    num = copy_tree_and_convert_pnm(input_dir, output_dir)

    print(f"\nDone. Converted {num} .pnm files and copied everything else.")


if __name__ == "__main__":
    main()

