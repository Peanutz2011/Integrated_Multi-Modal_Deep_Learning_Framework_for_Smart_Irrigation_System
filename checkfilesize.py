from pathlib import Path
from PIL import Image

DATA_ROOT = Path("maize_watered_drought_sorted")

# Set your folder path here
folder_path = Path(DATA_ROOT / "train/Watered_Trial001_1")  # Or direct path like Path("path/to/folder")

# Supported image formats
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

widths = []
heights = []

# Loop through all files in the directory (and subdirectories)
for img_path in folder_path.rglob("*"):
  if img_path.suffix.lower() in image_extensions:
    try:
      with Image.open(img_path) as img:
        w, h = img.size
        widths.append(w)
        heights.append(h)
    except Exception as e:
      print(f"Could not read {img_path.name}: {e}")

# Calculate and print averages
if widths and heights:
  avg_width = sum(widths) / len(widths)
  avg_height = sum(heights) / len(heights)
  print(f"Total images analyzed: {len(widths)}")
  print(f"Average Width:  {avg_width:.1f} pixels")
  print(f"Average Height: {avg_height:.1f} pixels")
  print(f"Min dimensions: {min(widths)}x{min(heights)}")
  print(f"Max dimensions: {max(widths)}x{max(heights)}")
else:
  print("No valid images found in the specified folder.")