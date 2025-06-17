#CUDA_LAUNCH_BLOCKING=1 python3 sam.py dog.png ../s.jpg 

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import json
import sys

newlySelectedAreas = list()

def mouse_callback(event, x, y, flags, param):
    global newlySelectedAreas
    masks = param["masks"]
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at coordinates (x: {x}, y: {y})")
        if (len(masks)>0):
           for i,mask in enumerate(masks):
             segmentationFlag = mask['segmentation'][y][x]
             if (segmentationFlag):
                print("#Area %u X/Y (%u/%u)" % (i,x,y))
                newlySelectedAreas.append(i)



def overlay_mask(image, mask, alpha=0.5, true_color=(0, 255, 0)):
    """
    Overlay a semi-transparent mask on top of an image.

    Parameters:
    - image: Base image
    - mask: 2D boolean mask
    - alpha: Transparency level (0.0 - 1.0)
    - true_color: Color for True values in the mask (default is green)

    Returns:
    - Resulting image with the overlay
    """
    # Convert boolean mask to 3-channel image with specified color for True values
    overlay       = np.zeros_like(image)
    overlay[mask] = true_color

    # Blend the overlay with the original image using addWeighted
    result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    return result



def process_mask(mask, kernel_size=5, iterations=2,min_blob_area=1500):
    """
    Process a binary mask by eliminating noise, filling blanks, and removing small blobs.

    Parameters:
    - mask: Binary mask (2D NumPy array)
    - kernel_size: Size of the structuring element for morphological operations (default is 5)
    - iterations: Number of iterations for morphological operations (default is 2)

    Returns:
    - Processed mask
    """
    mask_uint8 = mask.astype(np.uint8) * 255

    # Define a rectangular kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform morphological closing (dilate followed by erode) to fill gaps and remove small blobs
    processed_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # Find connected components and their statistics
    _, labels, stats, _ = cv2.connectedComponentsWithStats(processed_mask)

    # Filter out small blobs based on the minimum area threshold
    for i in range(1, len(stats)):  # Start from 1 to skip background component
        if stats[i, cv2.CC_STAT_AREA] < min_blob_area:
            processed_mask[labels == i] = 0


    processed_mask_bool = (processed_mask > 0)
    return processed_mask_bool



def union(array1, array2):
    """
    Compute the union of two 2D NumPy arrays of boolean values.

    Parameters:
    - array1: First boolean array
    - array2: Second boolean array

    Returns:
    - Union of the two arrays
    """
    return np.logical_or(array1, array2)



def saveMask(filename,mask):
    vis_image = np.zeros((mask.shape[0], mask.shape[1], 1), dtype=np.uint8)
    vis_image[mask]  = 255
    cv2.imwrite(filename,vis_image)



def show_image_with_callback(path,image,masks):
    global newlySelectedAreas
    selectedAreas = list()

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback, {"masks": masks})

    foregroundMask = np.zeros((image.shape[0], image.shape[1] ), dtype=bool)
    imageToShow = image

    while True:
        if (len(newlySelectedAreas)>0):
          for newArea in newlySelectedAreas:
             selectedAreas.append(newArea)
             foregroundMask = union ( foregroundMask , masks[newArea]['segmentation'])

          foregroundMask = process_mask(foregroundMask, kernel_size=5, iterations=2)
          imageToShow    = overlay_mask(image, foregroundMask)

          saveMask("%s_foreground.png"%path ,foregroundMask) 
          newlySelectedAreas = [] #Flush UI input
           
        cv2.imshow('Image', imageToShow)
        key = cv2.waitKey(15) & 0xFF

        if key == 27:  # Press 'Esc' to exit
            break

    cv2.destroyAllWindows()



def show_anns(anns,path):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

    plt.imsave(path,img) 
    return img


 
def main():
  sys.path.append("..")
  from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

  sam_checkpoint = "sam_vit_h_4b8939.pth"
  model_type = "vit_h"

  #sam_checkpoint = "sam_vit_l_0b3195.pth"
  #model_type = "vit_l"
  device = "cuda" # "cuda or cpu"


  #sam_checkpoint = "sam_vit_b_01ec64.pth"
  #model_type = "vit_b"
  #device = "cpu" # "cuda or cpu"

  sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
  sam.to(device=device)

  mask_generator = SamAutomaticMaskGenerator(sam)


 
  for index, arg in enumerate(sys.argv[1:], start=1):
        print(f"Argument {index}: {arg}")

        image = cv2.imread(arg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Image ",image.shape)# Convert the image to PyTorch Tensor 


        masks = mask_generator.generate(image)
        print("Masks : ",len(masks))
        if (len(masks)>0):
           print(masks[0].keys())

        show_anns(masks,"%s_SAM.png"%arg)

        with open("%s_SAM.txt"%arg, 'w') as f:
         for submask in masks:
            f.write(str(submask))
       
        #show_image_with_callback(arg,image,masks) 



if __name__ == "__main__":
    main()

