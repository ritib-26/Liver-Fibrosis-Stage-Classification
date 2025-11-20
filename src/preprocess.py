import os
from PIL import Image
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TARGET_SIZE = (224, 224)
CLASSES = ["F0", "F1", "F2", "F3"]

def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clean_hidden_files(folder):
    """Remove hidden files like .DS_Store"""
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.startswith("."):
                os.remove(os.path.join(root, f))

def load_valid_images(class_dir):
    """Return list of images that can be opened"""
    valid = []
    for img_name in os.listdir(class_dir):
        path = os.path.join(class_dir, img_name)
        try:
            img = Image.open(path)
            img.verify()
            valid.append(img_name)
        except:
            print(f"Corrupt image skipped: {path}")
    return valid

def preprocess_and_save(src_path, dest_path):
    img = Image.open(src_path).convert("L")      # Keep grayscale
    img = img.resize(TARGET_SIZE)
    img.save(dest_path)

def main():
    print("Cleaning hidden files...")
    clean_hidden_files(RAW_DIR)

    # Make processed directories
    for split in ["train", "val", "test"]:
        for c in CLASSES:
            safe_mkdir(os.path.join(PROCESSED_DIR, split, c))

    print("Collecting valid images...")
    data, labels = [], []
    for c in CLASSES:
        class_dir = os.path.join(RAW_DIR, c)
        imgs = load_valid_images(class_dir)
        for img_name in imgs:
            data.append(os.path.join(class_dir, img_name))
            labels.append(c)

    # Stratified split: 70 / 15 / 15
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, test_size=0.3, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    splits = [("train", X_train, y_train),
              ("val",   X_val,   y_val),
              ("test",  X_test,  y_test)]

    print("Processing and saving images...")
    for split_name, X, y in splits:
        for src_path, label in zip(X, y):
            dest_path = os.path.join(PROCESSED_DIR, split_name, label, os.path.basename(src_path))
            preprocess_and_save(src_path, dest_path)

    print("Done! Processed images saved to:", PROCESSED_DIR)

if __name__ == "__main__":
    main()
