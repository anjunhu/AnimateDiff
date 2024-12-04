from fuzzywuzzy import fuzz
from datasets import Dataset, load_dataset
import json
import os

output_file = "/home/ubuntu/video/Open-Sora/assets/webvid10m/clusters-final.json"

# Load the clusters from your local file
with open(output_file, "r") as f:
    clusters = json.load(f)

# Load the WebVid-10M dataset
dataset = load_dataset("TempoFunk/webvid-10M", split="train")  # Replace "train" with your desired split

# Ensure the output folder exists
os.makedirs("filtered_webvid_datasets", exist_ok=True)

# Process one cluster at a time
for cluster_idx, (cluster_key, cluster_captions) in enumerate(clusters.items()):
    print(f"Processing cluster {cluster_idx + 1}/{len(clusters)}: {cluster_key}")
    
    # Use the first caption of the cluster
    first_caption = cluster_captions[0] if cluster_captions else "no_caption"
    
    # Define a function to filter videos based on similarity for the current cluster
    def filter_videos(example):
        video_name = example["name"]  # Assuming video name is in the "name" field
        if fuzz.ratio(video_name, first_caption) > 85:
            return True
        return False

    # Filter the dataset for the current cluster
    filtered_dataset = dataset.filter(filter_videos)

    # Generate a descriptive folder name
    save_folder_name = f"filtered_webvid_datasets/filtered_webvid_dataset_{cluster_key}_{first_caption[:30]}"
    
    # Save the filtered dataset
    filtered_dataset.save_to_disk(save_folder_name)
    print(f"Filtered dataset for cluster {cluster_key} saved to '{save_folder_name}'")
