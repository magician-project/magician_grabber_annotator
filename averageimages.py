import cv2
import os
import sys
import numpy as np

def average_images(directory,extension):
    # Get all files with the specified pattern
    #file_pattern = 'colorFrame_0_*.pnm'
    file_pattern = '*.%s'%extension
    file_list = [f for f in os.listdir(directory) if f.endswith('.%s'%extension)]
    file_list.sort()

    # Ensure there are images in the directory
    if not file_list:
        print("No matching images found in the directory.")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(directory, file_list[0])
    first_image = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)

    # Initialize sum array
    image_sum = np.zeros_like(first_image, dtype=np.float64)

    # Loop through all images and accumulate pixel values
    for filename in file_list:
        print("Processing ",filename)
        image_path = os.path.join(directory, filename)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image_sum += img.astype(np.float64)

    # Calculate the average by dividing by the number of images
    average_image = (image_sum / len(file_list)).astype(np.uint8)

    # Save the result
    output_path = os.path.join(directory, 'average_image.pnm')
    cv2.imwrite(output_path, average_image)

    print(f"Average image saved to: {output_path}")

if __name__ == "__main__":
    # Provide the directory containing the images

    if len(sys.argv) != 3:
        print("Usage: python script.py <image_directory> <extension>")
        sys.exit(1)

    image_directory = sys.argv[1]
    extension = sys.argv[2]
    average_images(image_directory,extension)
