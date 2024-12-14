import os
from PIL import Image
import numpy as np

# Process images and save results
def process_images(image_folder):
    results = []

    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):  # Check for image file extensions
            image_path = os.path.join(image_folder, image_file)
            print(f"Processing: {image_path}")

            try:
                # Open the image using PIL
                raw_image = Image.open(image_path)

                # Ensure the image is a valid PIL image
                if isinstance(raw_image, np.ndarray):
                    print(f"Image {image_file} is already in ndarray format.")
                else:
                    print(f"Image {image_file} is not a valid PIL Image.")
                    continue  # Skip this image if it's not valid

                # If you need to process the image further, you can do it here
                # For example: raw_image = some_processing_function(raw_image)
                
                # Append the processed result
                results.append({"Image": image_file, "Status": "Processed"})

            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # If there's an error loading the image, append error status to results
                results.append({"Image": image_file, "Status": f"Error: {e}"})
                continue

    # Optionally: you can save the results to a CSV or log them
    for result in results:
        print(result)

print('Processing Images')
IMAGE_FOLDER = "images"  # Update this with your folder path
process_images(IMAGE_FOLDER)
print('Processed Images')
