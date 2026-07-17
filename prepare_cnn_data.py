import os
import shutil

def prepare_cnn_data(source_dir, dest_dir):
    """
    Reorganizes a YOLO formatted dataset into a standard Image Classification dataset.
    """
    classes = {0: 'bike', 1: 'scooty'}
    splits = ['train', 'val', 'test']
    
    # Create destination directories
    for split in splits:
        for cls_name in classes.values():
            os.makedirs(os.path.join(dest_dir, split, cls_name), exist_ok=True)
            
    # Process each split
    for split in splits:
        images_dir = os.path.join(source_dir, 'images', split)
        labels_dir = os.path.join(source_dir, 'labels', split)
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Skipping {split} - directory not found.")
            continue
            
        print(f"Processing {split} set...")
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
                
            label_path = os.path.join(labels_dir, label_file)
            
            # Read the first line to get the class ID
            try:
                with open(label_path, 'r') as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue
                    class_id = int(first_line.split()[0])
            except Exception as e:
                print(f"Error reading {label_file}: {e}")
                continue
                
            if class_id not in classes:
                print(f"Unknown class ID {class_id} in {label_file}")
                continue
                
            class_name = classes[class_id]
            
            # Find the corresponding image
            image_name_base = os.path.splitext(label_file)[0]
            
            # Look for common image extensions
            image_found = False
            for ext in ['.jpg', '.jpeg', '.png']:
                img_src = os.path.join(images_dir, image_name_base + ext)
                if os.path.exists(img_src):
                    img_dest = os.path.join(dest_dir, split, class_name, image_name_base + ext)
                    
                    # Check if we already have 50 images for this class in this split
                    existing_images = len(os.listdir(os.path.join(dest_dir, split, class_name)))
                    if existing_images < 50:
                        shutil.copy2(img_src, img_dest)
                        image_found = True
                    else:
                        image_found = True # We don't want to print 'not found'
                    break
                    
            if not image_found:
                print(f"Image for label {label_file} not found.")

if __name__ == "__main__":
    SOURCE_DATASET = "dataset/dataset3"
    DEST_DATASET = "dataset_cnn"
    print("Starting data preparation...")
    prepare_cnn_data(SOURCE_DATASET, DEST_DATASET)
    print("Data preparation completed!")
