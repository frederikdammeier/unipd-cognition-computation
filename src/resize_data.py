import os
import argparse
from pathlib import Path
from PIL import Image


def crop_to_square(image):
    """
    Symmetrically crop an image to a square based on the shorter dimension.
    
    Args:
        image: PIL Image object
        
    Returns:
        Cropped PIL Image object
    """
    width, height = image.size
    min_dim = min(width, height)
    
    # Calculate crop box to center the square crop
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    
    return image.crop((left, top, right, bottom))


def resize_images(input_path, output_path, size):
    """
    Resize all .jpg images in input_path to N*N pixels and save to output_path.
    
    Args:
        input_path: Path to input folder structure
        output_path: Path to output folder structure
        size: Target size (N) for N*N pixel output
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Find all .jpg files recursively
    jpg_files = list(input_path.rglob('*.jpg')) + list(input_path.rglob('*.JPG'))
    
    if not jpg_files:
        print(f"No .jpg files found in {input_path}")
        return
    
    print(f"Found {len(jpg_files)} image(s) to process")
    
    for jpg_file in jpg_files:
        try:
            # Get relative path from input_path
            relative_path = jpg_file.relative_to(input_path)
            
            # Create corresponding output directory structure
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Open, crop, and resize image
            image = Image.open(jpg_file)
            
            # Crop to square
            square_image = crop_to_square(image)
            
            # Resize to N*N
            resized_image = square_image.resize((size, size), Image.Resampling.LANCZOS)
            
            # Save to output location
            resized_image.save(output_file, quality=95)
            print(f"✓ Processed: {relative_path}")
            
        except Exception as e:
            print(f"✗ Error processing {jpg_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Resize and crop all .jpg images in a folder structure to N*N pixels"
    )
    parser.add_argument(
        "input_path",
        help="Path to input folder containing .jpg images"
    )
    parser.add_argument(
        "output_path",
        help="Path to output folder for resized images"
    )
    parser.add_argument(
        "-n", "--size",
        type=int,
        default=256,
        help="Target size for square images (N*N pixels). Default: 256"
    )
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.isdir(args.input_path):
        print(f"Error: Input path '{args.input_path}' does not exist")
        return
    
    # Create output path if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Process images
    resize_images(args.input_path, args.output_path, args.size)
    print("Done!")


if __name__ == "__main__":
    main()
