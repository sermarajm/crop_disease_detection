import os
import shutil
import random

# -------------------------------
# CONFIGURATION
# -------------------------------
SOURCE_DIR = "Color Images"   # original dataset folder
DEST_DIR = "dataset"                  # output folder
TRAIN_RATIO = 0.8                     # 80% train, 20% test

TRAIN_DIR = os.path.join(DEST_DIR, "train")
TEST_DIR = os.path.join(DEST_DIR, "test")

# -------------------------------
# CREATE FOLDERS
# -------------------------------
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# -------------------------------
# SPLIT FUNCTION
# -------------------------------
def split_class(class_name):
    src_class_path = os.path.join(SOURCE_DIR, class_name)

    images = os.listdir(src_class_path)
    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_index]
    test_images = images[split_index:]

    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    test_class_dir = os.path.join(TEST_DIR, class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(src_class_path, img),
            os.path.join(train_class_dir, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(src_class_path, img),
            os.path.join(test_class_dir, img)
        )

    print(f"{class_name}: {len(train_images)} train | {len(test_images)} test")

# -------------------------------
# RUN SPLIT
# -------------------------------
for class_folder in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_folder)

    if os.path.isdir(class_path):
        split_class(class_folder)

print("\n✅ Dataset split completed successfully!")
