import argparse
import os
import torch
import numpy as np
from skimage.io import imsave
from skimage.transform import resize
from PIL import Image
import tifffile as tiff
import ever as er
from ever.core.builder import make_model
from ever.core.config import import_config
from ever.core.checkpoint import remove_module_prefix

# Disable image size check to prevent decompression bomb error
Image.MAX_IMAGE_PIXELS = None

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Image Segmentation using HRNet model")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image")
    return parser.parse_args()

# Main function to load the model, process the image and run inference
def main():
    # Parse arguments
    args = parse_args()
    image_path = args.image_path

    # Set the paths to your downloaded LoveDA files
    ckpt_path = "/path/to/LoveDA/Semantic_Segmentation/configs/baseline/hrnetw32.pth"  # Adjust path
    config_path = "/path/to/LoveDA/Semantic_Segmentation/configs/baseline/hrnetw32.py"  # Adjust path

    # Load model configuration and model
    er.registry.register_all()
    cfg = import_config(config_path)

    model = make_model(cfg['model'])
    state_dict = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(remove_module_prefix(state_dict))
    model.cuda()
    model.eval()

    # Try loading the image with tifffile
    try:
        image = tiff.imread(image_path)
        print(f"Image shape loaded with tifffile: {image.shape}")
    except Exception as e:
        print(f"Error loading with tifffile: {e}")
        print("Trying with PIL...")

        # Fallback to using PIL to load .tif image
        image = Image.open(image_path)
        image = np.array(image)
        print(f"Image shape loaded with PIL: {image.shape}")

    # If the image has more than 3 channels (RGBA or other formats), convert to 3 channels (RGB)
    if image.ndim == 3 and image.shape[-1] == 4:
        print("Converting RGBA to RGB...")
        from skimage.color import rgba2rgb
        image = rgba2rgb(image)  # Automatically removes the alpha channel

    # Resize the image to 512x512 and normalize
    original_size = image.shape[:2]
    image_resized = resize(image, (512, 512), preserve_range=True, anti_aliasing=True)
    image_resized = np.transpose(image_resized, (2, 0, 1)) / 255.0  # Change (H, W, C) to (C, H, W)

    # Convert to tensor and move to GPU
    image_tensor = torch.tensor(image_resized, dtype=torch.float32).unsqueeze(0).cuda()  # Add batch dimension
    print(f"Final input shape: {image_tensor.shape}")  # Should be [1, 3, 512, 512]

    # Perform inference with the model
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

    # Resize the mask back to original image size
    mask_resized = resize(pred_mask, original_size, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    print("Unique class IDs in the prediction mask:", np.unique(pred_mask))

    # Define color map for segmentation
    color_map = {
        0: (0, 0, 0),      # Background - black
        1: (128, 0, 0),    # Building - dark red
        2: (128, 128, 128),# Road - gray
        3: (0, 0, 255),    # Water - blue
        4: (255, 255, 0),  # Barren - yellow
        5: (0, 128, 0),    # Forest - green
        6: (255, 243, 128) # Agriculture - light yellow
    }

    # Create the color mask based on class IDs
    color_mask = np.zeros((mask_resized.shape[0], mask_resized.shape[1], 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        color_mask[mask_resized == class_id] = color

    # Save the colorized segmentation mask
    output_path = os.path.splitext(image_path)[0] + "_mask.tif"
    imsave(output_path, color_mask)
    print(f"Segmentation mask saved at {output_path}")

if __name__ == "__main__":
    main()
