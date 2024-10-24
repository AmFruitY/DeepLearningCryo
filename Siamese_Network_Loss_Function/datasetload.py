from pathlib import Path
from zipfile import ZipFile
import gdown

# Define the cache directory inside the .keras folder
cache_dir = Path(Path.home()) / ".keras"
anchor_images_path = cache_dir / "left"
positive_images_path = cache_dir / "right"

# Download the left.zip and right.zip from Google Drive
gdown.download(id="1jvkbTr_giSP3Ru8OwGNGc6B4PvVbcO34", output="left.zip", quiet=False)
gdown.download(id="1EzBZUb_mh_Dp_FKD0P4XiYYSd0QBH5zW", output="right.zip", quiet=False)

# Unzip the files to the cache directory
with ZipFile('left.zip', 'r') as zip_ref:
    zip_ref.extractall(cache_dir)

with ZipFile('right.zip', 'r') as zip_ref:
    zip_ref.extractall(cache_dir)
