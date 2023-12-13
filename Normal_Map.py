#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:17:48 2023

@author: NullCipherr
"""

import numpy as np
import cv2
import os

# Define the debugging folder
depuration_folder = "Depuration"

##############################################################################
# Function 'load_image': Loads any image.
##############################################################################
# Parameters
#   - input_path: Path where the desired image is located.
##############################################################################
def load_image(input_path):
    try:
        image = cv2.imread(input_path)

        if image is not None:
            return image
        else:
            print("Error loading the image at the specified path...")
            return None
    except Exception as e:
        print(f"Error loading the image: {e}")
        return None


##############################################################################
# Function 'save_image': Saves any image to a specific path.
##############################################################################
# Parameters
#   - image: Image passed as a parameter to be saved at the desired location.
#   - output_name: Output name for the saved image.
#   - output_folder: Folder where the image will be saved.
##############################################################################
def save_image(image, output_name, output_folder):

    print("\nOutput name -> ", output_name, "\nOutput folder -> ", output_folder)

    # Create the complete path to the location
    output_path = os.path.join(output_folder, output_name)

    try:
        cv2.imwrite(output_path, image)
        print(f"Image successfully saved at {output_path}")
    except Exception as e:
        print(f"Error saving the image: {e}")


##############################################################################
# Function 'calculate_normal_map': Calculates the normal map of an image.
##############################################################################
# Parameters
#   - image: Desired image to calculate the normal map.
##############################################################################
def calculate_normal_map(image):
    # Convert the image to grayscale (necessary for gradient calculation)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save the grayscale image for debugging
    save_image(gray_image, "Gray.jpg", depuration_folder)

    # Calculate horizontal and vertical gradients using the Sobel operator
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)

    # Save gradient images for debugging
    save_image(gradient_x, "Gradient_x.jpg", depuration_folder)
    save_image(gradient_y, "Gradient_y.jpg", depuration_folder)

    # Normalize gradients to the range [-1, 1]
    gradient_x = cv2.normalize(gradient_x, None, -1, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    gradient_y = cv2.normalize(gradient_y, None, -1, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    normalized_gradient_x = (gradient_x * 255).astype(np.uint8)
    normalized_gradient_y = (gradient_y * 255).astype(np.uint8)

    # Save normalized gradient images for debugging
    save_image(normalized_gradient_x, "Normalized_Gradient_x.jpg", depuration_folder)
    save_image(normalized_gradient_y, "Normalized_Gradient_y.jpg", depuration_folder)

    # Calculate the Z component of the normal using the normalization formula
    component_z = np.sqrt(np.clip(1.0 - gradient_x**2 - gradient_y**2, 0, 1))

    normal_z = (component_z * 255).astype(np.uint8)

    # Save the normal_z image for debugging
    save_image(normal_z, "Normal_z.jpg", depuration_folder)

    # Stack the normal components into an RGB image
    normal_map = np.dstack((gradient_x, gradient_y, component_z))

    return (normal_map * 128 + 128).astype(np.uint8)


if __name__ == "__main__":

    # Path of the input image.
    input_image = "Texture.jpg"

    # Path to the input folder.
    input_folder = "Input"

    # Load the input image.
    input_image_path = os.path.join(input_folder, input_image)

    # Load the input image.
    input_image = load_image(input_image_path)

    if input_image is not None:
        # Apply the 'calculate_normal_map' function to the input image.
        normalMap = calculate_normal_map(input_image)

        # Save the input image.
        save_image(normalMap, "Normal_Map.jpg", "Output")
