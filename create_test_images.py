import os
from PIL import Image
import numpy as np

# Create test images directory
os.makedirs("data/raw/image_train", exist_ok=True)
os.makedirs("data/raw/image_test", exist_ok=True)

# Create dummy images (224x224 like VGG16 expects)
# Read the CSV to get image IDs and product IDs
import pandas as pd

X_train = pd.read_csv("data/raw/X_train_update.csv")

# Only create first 100 images for faster development
sample_size = min(100, len(X_train))
X_train_sample = X_train.head(sample_size)

print(f"Creating {len(X_train_sample)} test images...")

for idx, row in X_train_sample.iterrows():
    image_id = row['imageid']
    product_id = row['productid']
    
    # Create dummy image (random noise)
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save
    filename = f"data/raw/image_train/image_{image_id}_product_{product_id}.jpg"
    img.save(filename)
    
    if (idx + 1) % 10 == 0:
        print(f"Created {idx + 1} images...")

print("Test images created!")
