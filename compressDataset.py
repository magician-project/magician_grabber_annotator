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

def _atomic_replace_dir(src_new: str, dst_final: str):
    """
    Replace dst_final with src_new safely:
      - move dst_final -> backup
      - move src_new  -> dst_final
      - delete backup
    """
    dst_final = os.path.abspath(dst_final)
    src_new = os.path.abspath(src_new)

    if not os.path.isdir(src_new):
        raise RuntimeError(f"Temp output directory does not exist: {src_new}")

    parent = os.path.dirname(dst_final)
    base = os.path.basename(dst_final.rstrip(os.sep))
    backup = os.path.join(parent, f".{base}.backup_old")

    # Ensure no stale backup
    if os.path.exists(backup):
        shutil.rmtree(backup)

    # Move original aside (if exists)
    if os.path.exists(dst_final):
        os.replace(dst_final, backup)

    # Move new into place
    os.replace(src_new, dst_final)

    # Remove backup
    if os.path.exists(backup):
        shutil.rmtree(backup)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python compressDataset.py <input_directory> [output_directory]")
        print("")
        print("Behavior:")
        print("  - If output_directory is given: copy+convert into output_directory")
        print("  - If only input_directory is given: convert IN PLACE (safe swap)")
        sys.exit(1)

    input_dir  = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) >= 3 else None

    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory")
        sys.exit(1)

    # --- In-place mode ---
    if output_dir is None:
        parent = os.path.dirname(input_dir)
        base = os.path.basename(input_dir.rstrip(os.sep))
        temp_out = os.path.join(parent, f".{base}.tmp_compress")

        if os.path.exists(temp_out):
            print(f"Error: temp folder already exists: {temp_out}")
            print("Delete it if it's leftover from a previous run.")
            sys.exit(1)

        print(f"Input directory (in-place): {input_dir}")
        print(f"Temp output directory      : {temp_out}")

        ensure_dir(temp_out)
        num = copy_tree_and_convert_pnm(input_dir, temp_out)

        print(f"\nConverted {num} .pnm files. Swapping temp into place...")
        _atomic_replace_dir(temp_out, input_dir)

        print("Done (in-place).")
        return

    # --- Two-path mode (copy) ---
    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        print("Error: input_directory and output_directory must be different when using two-path mode.")
        sys.exit(1)

    ensure_dir(output_dir)

    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")

    num = copy_tree_and_convert_pnm(input_dir, output_dir)

    print(f"\nDone. Converted {num} .pnm files and copied everything else.")

if __name__ == "__main__":
    main()

