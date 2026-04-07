import os
import glob

def get_dataset_info(dataset_path="data/asl_digits"):
    """
    Scans the dataset directory and returns a dictionary of label: [image_paths]
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} not found.")
        # Try a different structure if move/mv was different
        dataset_path = "data"
        
    labels = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    dataset_info = {}
    
    for label in labels:
        label_dir = os.path.join(dataset_path, label)
        image_paths = glob.glob(os.path.join(label_dir, "*.jpg")) + \
                      glob.glob(os.path.join(label_dir, "*.png")) + \
                      glob.glob(os.path.join(label_dir, "*.jpeg"))
        if image_paths:
            dataset_info[label] = image_paths
        
    return dataset_info

if __name__ == "__main__":
    info = get_dataset_info()
    print(f"Found {len(info)} classes and {sum(len(v) for v in info.values())} total images.")
