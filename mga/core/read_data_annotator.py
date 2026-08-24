#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

THIS FILE WAS CALLED readData.py UNTIL 2026-08-03. It was renamed because
magician_vision_classifier has a readData.py of its own, and the old name silently
shadowed it: wxAnnotator.py *appends* the classifier directory to sys.path while the
annotator's own directory is sys.path[0] (and, with the site's PYTHONPATH starting in
an empty entry, the cwd too), so the classifier's `from readData import
readPolarPNMToRGBALive` in classifierPnm.py picked up OUR file instead of its own.
Editing a function here could therefore change live classification while the
classifier repo looked untouched. After the rename each repo loads its own copy —
verified byte-identical classifier output either way (same heatmap md5, same 112
activations on FORTH_MIX_650 frame 0 with mix_convnext_tiny).

DO NOT rename this back, and do not add a readData.py to this directory.

The two files are NOT copies of each other and must not be resynced. This one is the
newer: it carries 9 functions the classifier's has never had
(resolve_annotation_json_path, list_image_files, loadImageAndJSON, loadImage,
repackPolarToMosaic, averagePolarRGBAtoGray, get_md5, check_threshold_count,
select_most_different_tiles) and its tileImages() takes the severity / clean-class /
quarantine_px / defect_tiles_per_point arguments datasetCreator.py passes. The
classifier's still has the old 9-argument tileImages plus two functions we do not use
(highlightImage, loadMoreImages). Overwriting this file with that one would break the
annotator outright.

Nothing here is shared code any more, so everything is free to diverge — with one
caveat worth knowing when debugging a disagreement between the annotator's own
rendering and what the classifier sees: debayerPolarImage() and
readPolarPNMToRGBALive() still exist on both sides and are expected to agree (same
2x2 mosaic order, same [0,45,90,135] channel assignment). They were verified
identical on real mosaic and packed frames on 2026-08-02; the only difference is that
ours preserves the input dtype where the classifier's forces uint8, which is moot for
the 8-bit frames in use.
"""

import sys
import os
import gc
import json
import cv2
import numpy as np 

"""
Check if a file exists
"""
def checkIfFileExists(filename):
    if filename is None:
          return False
    return os.path.isfile(filename) 

"""
Check if a path exists
"""
def checkIfPathExists(filename):
    if filename is None:
          return False
    return os.path.exists(filename) 

"""
Check if a path exists
"""
def checkIfPathIsDirectory(filename):
    if filename is None:
          return False
    return os.path.isdir(filename) 


"""
Do a CRC on data to prevent data corruption training errors
"""
def get_md5(file_path):
    # Construct the command
    command = f"md5sum {file_path}"
    
    # Execute the command and capture the output
    output = os.popen(command).read()
    
    # Parse the output to extract the MD5 hash
    md5_hash = output.split()[0]
    
    return md5_hash


def list_image_files(directory):
    """
    Retrieve a list of all files in the specified directory.

    Parameters:
    - directory (str): The path to the directory.

    Returns:
    - files (list): A list of file names in the directory.
    """

    image_extensions = ['.png', '.pnm', '.jpg', '.jpeg']
    image_files = []

    try:
        # Iterate over all files and directories in the specified directory
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)

            # Check if it's a file (not a directory) and has a valid image extension
            if os.path.isfile(filepath) and any(filename.lower().endswith(ext) for ext in image_extensions):
               if "foreground.png" in filepath:
                   print("Omitting ",filepath," since it is a foreground file!")
               else:
                   image_files.append(filepath) 

    except OSError as e:
        print(f"Error reading directory '{directory}': {e}")
    
    image_files.sort() # Always sort files 

    return image_files

def resolve_annotation_json_path(image_path: str, prefer_existing: bool = True) -> str:
    """
    Resolve the annotation JSON path for a given image.

    Supports multiple historical naming schemes:

    New style:
        image.ext -> image.ext.json

    Legacy styles:
        image.ext -> image.pnm.json
        image.ext -> image.png.json
        image.ext -> image.jpg.json

    If prefer_existing=True the first existing annotation file is returned.
    Otherwise the default new-style path is returned.
    """

    if image_path is None:
        return None

    root, ext = os.path.splitext(image_path)
    ext = ext.lower()

    candidates = []

    # 1️⃣ preferred modern format
    candidates.append(f"{image_path}.json")

    # 2️⃣ legacy variants (dataset history compatibility)
    legacy_exts = ["", ".pnm", ".png", ".jpg", ".jpeg"]

    for e in legacy_exts:
        candidates.append(f"{root}{e}.json")

    # remove duplicates while preserving order
    candidates = list(dict.fromkeys(candidates))

    if prefer_existing:
        for c in candidates:
            if os.path.isfile(c):
                return c

    # default location for saving annotations
    return candidates[0]


def annotation_json_path(image_path):
    """The annotation JSON path to use for an image: any existing historical
    naming scheme (see resolve_annotation_json_path), otherwise the new-style
    stem.json (colorFrame_0_00047.json). Shared by every wx_annotator path
    that reads or writes per-frame annotations."""
    jp = resolve_annotation_json_path(image_path, prefer_existing=True)
    if not jp or not checkIfFileExists(jp):
        jp = os.path.splitext(image_path)[0] + ".json"
    return jp


def read_annotation_json(json_path):
    """The annotation dict from json_path, or {} when missing or unreadable.
    Callers never have to care about the difference."""
    if checkIfFileExists(json_path):
        try:
            with open(json_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def dataset_images(streamer, filepath):
    """Frame list of the open dataset: the streamer's directoryList when
    present, else the image files next to `filepath` (a dataset opened via
    --from may not have a directory-mode streamer)."""
    images = list(getattr(streamer, "directoryList", None) or [])
    if not images and filepath and os.path.isfile(filepath):
        images = list_image_files(os.path.dirname(filepath))
    return images


"""
Function under construction 
"""
def select_most_different_tiles(tiles, X, random_state=42):
    from sklearn.decomposition import PCA
    """
    Select X tiles that are maximally different from each other
    using farthest-point sampling.

    Args:
        tiles: np.ndarray of shape (N, H, W, C)
        X: number of diverse tiles to select
        random_state: int, reproducibility
    
    Returns:
        selected_tiles: np.ndarray of shape (X, H, W, C)
        selected_indices: list of indices into original tiles
    """
    rng = np.random.RandomState(random_state)

    N = len(tiles)
    tiles = np.array(tiles, dtype=np.float32)
    tiles_flat = tiles.reshape(N, -1)

    # Optional: reduce dimensionality for speed
    if tiles_flat.shape[1] > 50:
        pca = PCA(n_components=50, random_state=random_state)
        reduced = pca.fit_transform(tiles_flat)
    else:
        reduced = tiles_flat

    # Step 1: start with a random tile
    selected_indices = [rng.randint(0, N)]
    distances = np.full(N, np.inf)

    # Step 2: iteratively add the farthest tile
    for _ in range(1, X):
        # update distances: min distance to any selected
        last_idx = selected_indices[-1]
        dist_to_last = np.linalg.norm(reduced - reduced[last_idx], axis=1)
        distances = np.minimum(distances, dist_to_last)

        # pick the farthest one
        next_idx = np.argmax(distances)
        selected_indices.append(next_idx)

    selected_tiles = tiles[selected_indices]
    return selected_tiles, selected_indices


def check_threshold(array, threshold):
    # Check if any pixel in any channel is above the threshold
    return np.any(array > threshold)

def check_threshold_count(array, threshold):
    """Return a binary mask where pixels exceed threshold in any channel, plus count."""
    mask = np.any(array > threshold)
    count = int(np.sum(mask))  # number of pixels above threshold
    return count

def check_variation(tile, threshold):
    # Calculate the standard deviation of pixel values in each channel
    std_dev = np.std(tile, axis=(0, 1))
    
    # Check if the standard deviation is greater than zero in any channel
    return np.any(std_dev > threshold)

def tileImages(image, 
               json_file, 
               tile_size=32, 
               border=0,
               step=3, 
               clean_step=None,
               low_value_tile_threshold=30, 
               ignoreBackground=False,
               mergeSameKindOfDefectsRegardlessOfCount=True,
               includeTilesAnnotatedByAI=True,
               use_severity=False,
               use_clean_class=True,
               quarantine_px=0,
               defect_tiles_per_point=0,
               debug=False):
    """
    Extract tiles from an image, oversampling tiles containing defects and
    undersampling clean tiles to balance the dataset.

    defect_step: step size for tiles containing defects (smaller -> more tiles)
    clean_step: step size for tiles without defects (larger -> fewer tiles)
    quarantine_px: tiles that come CLOSER than this to a defect point without
        containing it are dropped entirely (neither clean nor defect). The dent
        extends beyond the click point, so such tiles likely contain defect pixels
        and would poison the clean class. Keep this well below the pen-ring radius
        (~150 px) so ring-ink tiles stay in the clean class (they are wanted hard
        negatives: the robot works alongside humans, ink can appear in production).
        0 disables (legacy behaviour). Clean/RLClean points never quarantine.
    defect_tiles_per_point: cap on how many (randomly chosen) tiles of the dense
        defect enumeration are kept per annotation point. 0 = keep all
        (~(tile_size/step)^2 per point).

    Sampling structure (two phases):
      1. CLEAN grid — the whole frame scanned WITHOUT overlap (clean_step = tile_size
         by default) in BOTH axes: uniform area coverage, no near-duplicate cleans.
      2. DEFECT enumeration — for every defect point, tiles are generated at
         defect_step offsets in BOTH x and y so the defect appears at many positions
         inside the tile (the old scan only overlapped horizontally, and its row
         advance depended on the last tile of the row).
    """

    import json
    import numpy as np

    tiles                 = []
    tile_classes          = []
    tile_info             = []
    tiles_annotated_by_ai = 0

    # Load point clicks and their classes
    with open(json_file) as json_data:
        data = json.load(json_data)
        point_clicks      = data.get("pointClicks", [])
        point_classes     = data.get("pointClasses", [])
        points_severities = data.get("pointSeverities", [])

    height, width, channels = image.shape

    defect_step=step
    if clean_step is None:
          clean_step = tile_size

    def label_tile(start_x, start_y):
        """Resolve the class text / AI flag / quarantine flag for the tile whose
        top-left corner is (start_x, start_y). Same labeling rules as always."""
        end_x, end_y = start_x + tile_size, start_y + tile_size
        tile_text = ""
        tileAnnotatedByAI = 0
        nearDefect = False   # inside the quarantine band of a defect point

        for idx, (xFull, yFull) in enumerate(point_clicks):
            xAct, yAct = xFull // 2, yFull // 2  # <- keep your scaling
            if quarantine_px > 0 and point_classes[idx] not in ("Clean", "RLClean"):
                # distance from the point to the tile rectangle
                dx = max(start_x - xAct, 0, xAct - (end_x - 1))
                dy = max(start_y - yAct, 0, yAct - (end_y - 1))
                if 0 < dx * dx + dy * dy < quarantine_px * quarantine_px:
                    nearDefect = True
            if start_x <= xAct < end_x and start_y <= yAct < end_y:

                thisTileDescription = point_classes[idx]

                # RLClean: automatic RL annotation — treat as Clean when
                # includeTilesAnnotatedByAI is True, otherwise drop it.
                if thisTileDescription == "RLClean":
                   if includeTilesAnnotatedByAI:
                       thisTileDescription = "Clean"
                   else:
                       tileAnnotatedByAI = 1  # will be filtered below

                if (points_severities[idx]=="AI"):
                   tileAnnotatedByAI = 1
                   #points_severities[idx]="Class A" #<- Maybe package this with the rest ?

                #If we care about severities, this will make description of class:
                # PositiveDentClassA
                if (use_severity):
                   if (point_classes[idx]!="Clean"): #Clean tiles have no severity :P
                      thisTileDescription += points_severities[idx]

                #If we want we can consider different amounts of defects on a tile as different classes
                if not mergeSameKindOfDefectsRegardlessOfCount:
                   tile_text += thisTileDescription
                else:
                   if (tile_text == ""):
                        tile_text += thisTileDescription
                   elif (tile_text == thisTileDescription):
                        pass #Merge descriptions for the same class appearing again and again
                   else:
                        tile_text += thisTileDescription #Combinations of classes get a new class description

        return tile_text, tileAnnotatedByAI, nearDefect

    def register(tile, tile_text, tileAnnotatedByAI, start_x, start_y):
        nonlocal tiles_annotated_by_ai
        tiles.append(tile)
        tile_classes.append(tile_text)
        if debug:
            #If debuging mode is on also produce tile_info data to identify where the tile came from
            if (tileAnnotatedByAI):
                tiles_annotated_by_ai += 1
            # top-left corner (start_x, start_y) of the tile in the DEMOSAICED (half-res)
            # image the classifier operates on -- multiply by 2 for mosaic/click coords.
            tile_info.append("%s(%u,%u)"%(json_file,start_x,start_y))

    # --- PHASE 1: CLEAN grid — whole frame WITHOUT overlap in either axis ------------
    # (defect-containing tiles are skipped here; phase 2 samples those densely)
    y = border
    while y <= height - tile_size - border:
        x = border
        while x <= width - tile_size - border:
            tile = image[y:y + tile_size, x:x + tile_size]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size \
               and check_threshold_count(tile, low_value_tile_threshold) > 0:
                tile_text, tileAnnotatedByAI, nearDefect = label_tile(x, y)
                is_clean = tile_text in ("", "Clean")
                if not is_clean:
                    pass  # contains a defect point -> phase 2 handles it densely
                elif (not use_clean_class and tile_text == ""):
                    pass #We dont want to use the class Clean so we ignore it!
                elif (tileAnnotatedByAI and not includeTilesAnnotatedByAI):
                    pass #Ignore this tile that has been annotated by AI
                elif nearDefect:
                    # Quarantine band: too close to a defect point to be trusted as
                    # clean, but the point itself is outside -> drop the tile
                    pass
                elif ignoreBackground and tile_text == "":
                    pass # Skip background if requested
                else:
                    register(tile, tile_text, tileAnnotatedByAI, x, y)
            x += clean_step
        y += clean_step

    # --- PHASE 2: DEFECT enumeration — dense offsets in BOTH x and y per point -------
    import random as _random
    seen = set()
    for idx, (xFull, yFull) in enumerate(point_clicks):
        if point_classes[idx] in ("Clean", "RLClean"):
            continue                      # clean-ish points are phase-1 material
        xAct, yAct = int(xFull // 2), int(yFull // 2)
        # all top-left corners for which the point falls inside the tile
        x_lo = int(max(border, xAct - tile_size + 1))
        x_hi = int(min(xAct, width - tile_size - border))
        y_lo = int(max(border, yAct - tile_size + 1))
        y_hi = int(min(yAct, height - tile_size - border))
        if globals().get('CENTER_DEFECT', False):
            # Tile-size experiment: emit ONE tile per point with the defect at the
            # tile centre, so smaller tiles can be recovered by an offline centre-crop
            # (a centred N-tile is a valid centred (N-k)-tile). Trades positional
            # jitter for a clean field-of-view-only comparison. Clamp to keep in-bounds.
            cx = min(max(xAct - tile_size // 2, x_lo), x_hi)
            cy = min(max(yAct - tile_size // 2, y_lo), y_hi)
            offsets = [(cx, cy)]
        else:
            offsets = [(x0, y0)
                       for y0 in range(y_lo, y_hi + 1, defect_step)
                       for x0 in range(x_lo, x_hi + 1, defect_step)]
            if defect_tiles_per_point and len(offsets) > defect_tiles_per_point:
                offsets = _random.sample(offsets, defect_tiles_per_point)
        for (x0, y0) in offsets:
            if (x0, y0) in seen:
                continue                  # already produced for a nearby point
            tile = image[y0:y0 + tile_size, x0:x0 + tile_size]
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size \
               or check_threshold_count(tile, low_value_tile_threshold) <= 0:
                continue
            tile_text, tileAnnotatedByAI, _near = label_tile(x0, y0)
            if tile_text in ("", "Clean"):
                continue                  # defence: point rounding put it outside
            if (tileAnnotatedByAI and not includeTilesAnnotatedByAI):
                continue                  #Ignore this tile that has been annotated by AI
            seen.add((x0, y0))
            register(tile, tile_text, tileAnnotatedByAI, x0, y0)

    return tiles, tile_classes, tile_info, tiles_annotated_by_ai

def saveTiles(tiles,tile_classes):
    # Display or save the tiles as needed
    for i, (tile, tile_class) in enumerate(zip(tiles, tile_classes)):
      if tile is not None:
        if (tile.shape[0]==tile_size) and (tile.shape[1]==tile_size):  
          cv2.imwrite(f'tiles/tile_{i}{tile_class}.png', tile)
        else:
         print(f'Incorrect dimensions for tile {i}: {tile.shape}')
         print(f'tiles/tile_{i}{tile_class}.png') 


def loadMoreClasses(filename,classes_dict):
    with open("%s.json"%filename) as json_data:
        data          = json.load(json_data)
        point_clicks  = data.get("pointClicks", [])
        point_classes = data.get("pointClasses", [])
        for cl in point_classes:
           #print("Add `",cl,"` class ")
           classes_dict[cl]=True 
    return classes_dict 

def loadMoreClassesFromTiles(tile_classes,classes_dict):
    for cl in tile_classes:
           #print("Add `",cl,"` class ")
           classes_dict[cl]=True 
    return classes_dict 

def convertClassDictToOneHotList(classes_dict,tile_classes):
    classToIndex = dict()
    classToIndex[""]=0
    for i,key in enumerate(classes_dict.keys(), start=1):
       classToIndex[key]=i
    
    numberOfClasses = len(classToIndex)+1 #+1 is the none class
    numberOfSamples = len(tile_classes)
    print("We have ",numberOfSamples," samples with ",numberOfClasses," classes")
    onehot = np.full([numberOfSamples,numberOfClasses],fill_value=0,dtype=np.float32,order='C')
 
    for i in range(numberOfSamples):
        #if (tile_classes[i]!=""):
          onehot[i][classToIndex[tile_classes[i]]] = 1.0
     
    return onehot,numberOfClasses 

def debayerPolarImage(image): 
 # Split the A, B, C, and D values into separate monochrome images
 polarization_90_deg   = image[0::2, 0::2]
 polarization_45_deg   = image[0::2, 1::2]
 polarization_135_deg  = image[1::2, 0::2]
 polarization_0_deg    = image[1::2, 1::2]
 return polarization_0_deg,polarization_45_deg,polarization_90_deg,polarization_135_deg      

def repackPolarToMosaic(p0, p45, p90, p135):
    h, w = p0.shape
    mosaic = np.empty((h * 2, w * 2), dtype=p0.dtype)
    mosaic[0::2, 0::2] = p90
    mosaic[0::2, 1::2] = p45
    mosaic[1::2, 0::2] = p135
    mosaic[1::2, 1::2] = p0
    return mosaic


"""
def readPolarPNMToRGBALive(image):
    # Load the image
    image = np.squeeze(image)

    height, width = image.shape

    # Split into polarization images
    polarization_0_deg, polarization_45_deg, polarization_90_deg, polarization_135_deg = debayerPolarImage(image)

    # Create an RGBA image
    rgba_image = np.zeros((int(height/2),int(width/2), 4), dtype=np.uint8)

    # Assign each polarization image to a specific channel
    rgba_image[:, :, 0] = polarization_0_deg
    rgba_image[:, :, 1] = polarization_45_deg
    rgba_image[:, :, 2] = polarization_90_deg
    rgba_image[:, :, 3] = polarization_135_deg
    return rgba_image
"""


def readPolarPNMToRGBALive(image):
    """
    Accepts either:
      (A) DoFP mosaic single-channel image (H×W)  -> debayers to (H/2×W/2×4)
      (B) Already-packed polarization image (H×W×4) -> returned as-is

    Channel convention for already-packed PNGs (as written by cv2.imwrite on np.stack([p0,p45,p90,p135])):
      ch0=p0, ch1=p45, ch2=p90, ch3=p135
    """
    image = np.squeeze(image)

    # Case (B): already RGBA/polar packed
    if (image.ndim == 3) and (image.shape[2] == 4):
        return image

    # Case (A): classic mosaic (must be 2D)
    if image.ndim != 2:
        raise ValueError(f"readPolarPNMToRGBALive: expected 2D mosaic or 4-channel image, got shape {image.shape}")

    height, width = image.shape

    # Split into polarization images
    polarization_0_deg, polarization_45_deg, polarization_90_deg, polarization_135_deg = debayerPolarImage(image)

    # Create an RGBA image (preserve dtype)
    rgba_image = np.zeros((height // 2, width // 2, 4), dtype=image.dtype)

    # Assign each polarization image to a specific channel
    rgba_image[:, :, 0] = polarization_0_deg
    rgba_image[:, :, 1] = polarization_45_deg
    rgba_image[:, :, 2] = polarization_90_deg
    rgba_image[:, :, 3] = polarization_135_deg
    return rgba_image





def readPolarPNMToRGBA(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print("Error: Unable to read the image.")
        return None 

    return readPolarPNMToRGBALive(image)

def averagePolarRGBAtoGray(rgba_image):
    """
    Converts a 4-channel polarization RGBA image into a single-channel
    averaged grayscale image.

    Parameters:
        rgba_image (numpy.ndarray): 3D array (H x W x 4) representing the 4-channel polarization image.

    Returns:
        gray_image (numpy.ndarray): 2D array (H x W) representing the averaged grayscale image.
    """
    if rgba_image is None or rgba_image.ndim != 3 or rgba_image.shape[2] != 4:
        raise ValueError("Input must be a 4-channel (H x W x 4) image")

    # Compute the mean across the 4 polarization channels
    gray_image = np.mean(rgba_image.astype(np.float32), axis=2)

    # Convert back to uint8 (if original image is in that range)
    gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)

    return gray_image


def loadImageAndJSON(filename,
                     json_filename,
                     i,
                     border=0,
                     tile_size=32,
                     step=4,
                     low_value_tile_threshold=30,
                     debug=False,
                     includeTilesAnnotatedByAI=True,
                     use_severity=False,
                     use_clean_class=True,
                     ignoreBackground=False,
                     quarantine_px=0,
                     defect_tiles_per_point=0):

    tiles        = []
    tile_classes = []
    tile_info    = []
    tiles_annotated_by_ai = 0

    if (".png" in filename) or (".pnm" in filename) or (".jpeg" in filename) or (".jpg" in filename):

        rgba_image = readPolarPNMToRGBA(filename)

        if rgba_image is None:
            print(filename, " is not an image ")
            return tiles, tile_classes, tile_info, tiles_annotated_by_ai

        # Use supplied JSON instead of guessing
        tiles, tile_classes, tile_info, tiles_annotated_by_ai = tileImages(
            rgba_image,
            json_filename,
            border=border,
            tile_size=tile_size,
            step=step,
            low_value_tile_threshold=low_value_tile_threshold,
            debug=debug,
            includeTilesAnnotatedByAI=includeTilesAnnotatedByAI,
            use_severity=use_severity,
            use_clean_class=use_clean_class,
            ignoreBackground=ignoreBackground,
            quarantine_px=quarantine_px,
            defect_tiles_per_point=defect_tiles_per_point
        )

        del rgba_image

    return tiles, tile_classes, tile_info, tiles_annotated_by_ai


def loadImage(filename,
              i,
              border=0,
              tile_size=32,
              step=4,
              low_value_tile_threshold=30,
              debug=False,
              includeTilesAnnotatedByAI=True,
              use_severity=False,
              use_clean_class=True,
              ignoreBackground=False):

    json_filename = "%s.json" % filename

    return loadImageAndJSON(
        filename,
        json_filename,
        i,
        border=border,
        tile_size=tile_size,
        step=step,
        low_value_tile_threshold=low_value_tile_threshold,
        debug=debug,
        includeTilesAnnotatedByAI=includeTilesAnnotatedByAI,
        use_severity=use_severity,
        use_clean_class=use_clean_class,
        ignoreBackground=ignoreBackground
    )

def count_class_appearances(onehot, num_classes):
    score = list()
    for i in range(0, num_classes):
        score.append(0) 
    
    num_samples=onehot.shape[0]
    for sampleID in range(0,num_samples):
        for i in range(0,num_classes-1):
           if (onehot[sampleID][i]>0): 
              score[i]=score[i]+1

    return score


#12-0.02
#262 /235 - weld
#534 - black


if __name__ == '__main__':
  step = 10
  tile_size=48 
  tiles=[]
  tile_classes=[]
  class_dict=dict()


  for index, arg in enumerate(sys.argv[1:], start=1):
     if (checkIfFileExists(arg) and checkIfFileExists("%s.json"%arg)):
        print(f"Loading Classes / Argument {index}: {arg}")
        class_dict         = loadMoreClasses(arg,class_dict)
     else:
        print(f"NOT LOADING Argument {index}: {arg}")

  for index, arg in enumerate(sys.argv[1:], start=1):
     if (checkIfFileExists(arg) and checkIfFileExists("%s.json"%arg)):
        print(f"Loading Images / Argument {index}: {arg}")
        tiles,tile_classes = loadMoreImages(arg,index,tiles=tiles,tile_classes=tile_classes,tile_size=tile_size,step=step)
     else:
        print(f"NOT LOADING Argument {index}: {arg}")

  print("Do class update based on Tiles : ",len(tiles))
  class_dict = loadMoreClassesFromTiles(tile_classes,class_dict)
          
 

  print("Tiles : ",len(tiles))
  print("Tile Classes : ",len(tile_classes))
  print("Unique Classes : ",len(class_dict.keys()))

  print("Classes : ",class_dict.keys())
  onehot,num_classes = convertClassDictToOneHotList(class_dict,tile_classes)
  #print("One Hot : ",onehot)
  

  class_appearances = count_class_appearances(onehot,num_classes)
  print("Class Appearances:", class_appearances)
   
  class_weights = class_appearances
  for i in range(len(class_weights)):
    class_weights[i] /= len(tile_classes)
  print("Class Representation:", class_weights)     

  class_weights = class_appearances
  for i in range(len(class_weights)):
    if (class_weights[i]!=0):
      class_weights[i] = 1/class_weights[i]
  print("Class Weights:", class_weights)

  print("Completed Work")
  sys.exit(0)
