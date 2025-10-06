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


def highlightImage(image, json_file,tile_size=32,step=3):
    point_clicks = list()
    point_classes = list()

    # Extract point clicks and classes from the JSON data
    with open(json_file) as json_data:
        data = json.load(json_data)
        point_clicks = data.get("pointClicks", [])
        point_classes = data.get("pointClasses", [])

    # Get the dimensions of the image
    height, width, channels = image.shape

    # Calculate the number of tiles in each dimension
    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size

    # Loop through the image and extract tiles
    for y in range(0, height-tile_size, step):
        start_y = y
        end_y   = y + tile_size
        for x in range(0, width-tile_size, step):
          start_x = x
          end_x   = x + tile_size
            
          # Calculate the coordinates for each tile
          #print("(",start_x,",",end_x,") - (",start_y,",",end_y,")")
          # Extract the tile from the image
          if (end_x-start_x==tile_size) and (end_y-start_y==tile_size):  
            tile_text = ""
            i = 0 
            for xFull,yFull in point_clicks:
               xAct = xFull // 2
               yAct = yFull // 2
               if (start_x<=xAct) and (xAct<=end_x) and (start_y<=yAct) and (yAct<=end_y):
                     #if (tile_text==""):
                     #     tile_text = "_class"
                     tile_text = tile_text + point_classes[i]
               i = i+1

            if (tile_text!=""):
              image[start_y:end_y,start_x:end_x]=255
          else:
            print("")            

    return image

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

def tileImages(image, json_file,tile_size=32,border=0,step=3,low_value_tile_threshold=30,ignoreBackground=False):
    # Initialize per-call lists (avoid mutable defaults)
    tiles = []
    tile_classes = []

    point_clicks = list()
    point_classes = list()

    # Extract point clicks and classes from the JSON data
    with open(json_file) as json_data:
        data = json.load(json_data)
        point_clicks  = data.get("pointClicks", [])
        point_classes = data.get("pointClasses", [])

    # Get the dimensions of the image
    height, width, channels = image.shape

    # Calculate the number of tiles in each dimension
    num_tiles_x = width  // tile_size
    num_tiles_y = height // tile_size

    # Loop through the image and extract tiles
    for y in range(border, height-tile_size-border, step):
        start_y = y
        end_y   = y + tile_size
        for x in range(border, width-tile_size-border,step):
          start_x = x
          end_x   = x + tile_size
            
          # Calculate the coordinates for each tile
          #print("(",start_x,",",end_x,") - (",start_y,",",end_y,")")
          # Extract the tile from the image
          tile = image[start_y:end_y,start_x:end_x] #<- Why is this the correct order ?
          if (tile.shape[0]==tile_size) and (tile.shape[1]==tile_size):
           threshold_count = check_threshold_count(tile,low_value_tile_threshold)
           #if check_threshold(tile,30): #<- If there is any data in this tile
           if threshold_count > 0: #<- If there is any data in this tile
            tile_text = ""
            i = 0 
            for xFull,yFull in point_clicks:
               xAct = xFull // 2
               yAct = yFull // 2
               if (start_x<=xAct) and (xAct<=end_x) and (start_y<=yAct) and (yAct<=end_y):
                     #if (tile_text==""):
                     #     tile_text = "_class"
                     tile_text = tile_text + point_classes[i]
               i = i+1

            # Append the tile and info to the list
            if (ignoreBackground) and (tile_text == ""):
               #print("Skipping tile")
               pass
            else:
               tiles.append(tile)
               tile_classes.append(tile_text)

    return tiles,tile_classes


def saveTiles(tiles,tile_classes):
    # Display or save the tiles as needed
    for i, (tile, tile_class) in enumerate(zip(tiles, tile_classes)):
      if tile is not None:
        if (tile.shape[0]==tile_size) and (tile.shape[1]==tile_size):  
          cv2.imwrite(f'tiles/tile_{i}{tile_class}.png', tile)
        else:
         print(f'Incorrect dimensions for tile {i}: {tile.shape}')
         print(f'tiles/tile_{i}{tile_class}.png') 



def debayerPolarImage(image): 
 # Split the A, B, C, and D values into separate monochrome images
 polarization_90_deg   = image[0::2, 0::2]
 polarization_45_deg   = image[0::2, 1::2]
 polarization_135_deg  = image[1::2, 0::2]
 polarization_0_deg    = image[1::2, 1::2]
 return polarization_0_deg,polarization_45_deg,polarization_90_deg,polarization_135_deg      

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



def readPolarPNMToRGBALive(image):
    # Load the image
    image = np.squeeze(image)

    height, width = image.shape

    # Split into polarization images
    from readData import debayerPolarImage
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


def loadImage(filename,i,border=0,tile_size=32,step=4,low_value_tile_threshold=30,ignoreBackground=False):
  # Initialize per-call lists (avoid mutable defaults)
  tiles = []
  tile_classes = []

  if (".png" in filename) or (".pnm" in filename) or (".jpeg" in filename) or (".jpeg" in filename):
   rgba_image = readPolarPNMToRGBA(filename) 
   if rgba_image is None:
    print(filename," is not an image ")
    return tiles,tile_classes # Return already acquired tiles 
   else:
    # Save the combined image
    #output_filename = "sample_%05u.png" % i
    #cv2.imwrite(output_filename, rgba_image)
    #os.system("cp %s.json sample_%05u.json" % (filename,i))
    tiles, tile_classes = tileImages(rgba_image,"%s.json"%filename,border=border,tile_size=tile_size,step=step,low_value_tile_threshold=low_value_tile_threshold,ignoreBackground=ignoreBackground)

    #Remove RGBA image
    del rgba_image

    return tiles, tile_classes


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
         
  #dump_dataset_to_keras_data_loader(tiles,tile_classes) # moved to dumpeKerasDataset.py
 

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
