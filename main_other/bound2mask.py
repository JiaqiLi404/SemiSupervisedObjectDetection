'''
Transform bounding box -> mask in the whole folder
'''

import os
import json
import numpy as np
from PIL import Image
from matplotlib.path import Path
import matplotlib.pyplot as plt

# Define the path to the folder containing the images and JSON files
folder_path = 'D:/image'

# Loop over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):

        # Define the paths to the input image and JSON file
        image_path = os.path.join(folder_path, filename)
        json_path = os.path.join(folder_path, filename.replace('.png', '.json').replace('.jpg', '.json'))
        # Load the input image
        input_image = np.array(Image.open(image_path).convert('L'))

        # Load the JSON data
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        # Extract the shape information from the JSON
        shapes = json_data['shapes']

        # Create a new mask array with the same shape as the input image
        mask = np.zeros_like(input_image, dtype=np.uint8)

        # Iterate over the shapes and add them to the mask as polygons
        for shape in shapes:
            label = shape['label']
            points = shape['points']
            points = np.array(points)
            path = Path(points)
            y, x = np.mgrid[:mask.shape[0], :mask.shape[1]]
            points = np.vstack((x.ravel(), y.ravel())).T
            mask_values = path.contains_points(points)
            mask_values = mask_values.reshape(mask.shape[:2])
            mask[mask_values] = 1

        # Create a PIL image from the mask array and resize it to the same size as the input image
        mask_image = Image.fromarray(mask, mode='L')  # 'L' mode for grayscale images
        mask_image = mask_image.resize(input_image.shape[:2][::-1])  # reverse the shape to (width, height)

        # Save the mask image to a file using plt.imsave()
        mask_save_path = os.path.join(folder_path, filename.replace('.png', '_mask.png').replace('.jpg', '_mask.png'))
        plt.imsave(mask_save_path, mask_image, cmap='gray')