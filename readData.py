#!/usr/bin/python3

""" 
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 
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
    return os.path.isfile(filename) 

"""
Check if a path exists
"""
def checkIfPathExists(filename):
    return os.path.exists(filename) 

"""
Check if a path exists
"""
def checkIfPathIsDirectory(filename):
    return os.path.isdir(filename) 

 

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
               debug=False):
    """
    Extract tiles from an image, oversampling tiles containing defects and 
    undersampling clean tiles to balance the dataset.
    
    defect_step: step size for tiles containing defects (smaller -> more tiles)
    clean_step: step size for tiles without defects (larger -> fewer tiles)
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

    # Loop through the image
    y = border
    while y <= height - tile_size - border:
        x = border
        while x <= width - tile_size - border:
            start_x, end_x = x, x + tile_size
            start_y, end_y = y, y + tile_size

            tile = image[start_y:end_y, start_x:end_x]

            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                threshold_count = check_threshold_count(tile, low_value_tile_threshold)
                if threshold_count > 0:
                    # Check if tile contains any defect points
                    tile_text = ""
                    tileAnnotatedByAI = 0

                    for idx, (xFull, yFull) in enumerate(point_clicks):
                        xAct, yAct = xFull // 2, yFull // 2  # <- keep your scaling
                        if start_x <= xAct < end_x and start_y <= yAct < end_y:

                            thisTileDescription = point_classes[idx]
 
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

                    if (not use_clean_class and tile_text == ""):
                        pass #We dont want to use the class Clean so we ignore it!
                    elif (tileAnnotatedByAI and not includeTilesAnnotatedByAI):
                        pass #Ignore this tile that has been annotated by AI
                    elif ignoreBackground and tile_text == "":
                        # Skip background if requested
                        step_size = clean_step
                    else:
                        #Register the processed tile to our lists
                        tiles.append(tile)
                        tile_classes.append(tile_text)
                        if debug:
                           #If debuging mode is on also produce tile_info data to identify where the tile came from
                           AiAnnotationDebugString = ""
                           if (tileAnnotatedByAI):
                                AiAnnotationDebugString = "AIAnnotated "
                                tiles_annotated_by_ai += 1
                           tile_info.append("%s(%u,%u)"%(json_file,start_x,end_x))
                        # Use smaller step if tile contains defect
                        step_size = defect_step if tile_text != "" else clean_step
                else:
                    # Tile has too few pixels above threshold -> treat as background
                    step_size = clean_step
            else:
                step_size = clean_step

            x += step_size
        y += step_size

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
  # Initialize per-call lists (avoid mutable defaults)
  tiles        = []
  tile_classes = []
  tile_info    = []
  tiles_annotated_by_ai = 0

  if (".png" in filename) or (".pnm" in filename) or (".jpeg" in filename) or (".jpeg" in filename):
   rgba_image = readPolarPNMToRGBA(filename) 
   if rgba_image is None:
    print(filename," is not an image ")
    return tiles, tile_classes, tile_info, tiles_annotated_by_ai # Return nothing
   else:
    #If we successfully read an image, cut it into tiles 
    tiles, tile_classes, tile_info, tiles_annotated_by_ai = tileImages(rgba_image,
                                                "%s.json"%filename,
                                                border=border,
                                                tile_size=tile_size,
                                                step=step,
                                                low_value_tile_threshold=low_value_tile_threshold,
                                                debug=debug,
                                                includeTilesAnnotatedByAI=includeTilesAnnotatedByAI,
                                                use_severity=use_severity,
                                                use_clean_class=use_clean_class,
                                                ignoreBackground=ignoreBackground)

    #Remove RGBA image
    del rgba_image

    return tiles, tile_classes, tile_info, tiles_annotated_by_ai


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

 
def checkIfFileExists(filename):
    return os.path.isfile(filename) 


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
