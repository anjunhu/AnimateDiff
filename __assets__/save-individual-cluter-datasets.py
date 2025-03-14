import os 
import re
import json
import wandb
import torch
import ffmpeg
import pickle
import pandas as pd
import numpy as np
from pprint import pprint
from datetime import datetime
from fuzzywuzzy import fuzz
from datasets import Dataset, load_dataset
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Housekeeping
OUTPUT_FOLDER = "filtered_webvid_datasets_UN0050_avgCLIPvar"
PARTIAL_SAVE_FILE = os.path.join(OUTPUT_FOLDER, "partial_metadata.pkl")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_FILE = "./clusters-final.json"
with open(OUTPUT_FILE, "r") as f:
    clusters = json.load(f)

# Load the WebVid-10M dataset
dataset = load_dataset("TempoFunk/webvid-10M", split="train")

# Initialize the CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_video_features(video_path, num_frames=16, embedding_dim=512):
    """
    Extracts frame-level features from a video using the CLIP model.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample from the video.
        embedding_dim: Dimensionality of the feature embeddings (set by CLIP model).

    Returns:
        [num_frames, embedding_dim] tensor of features.
    """
    # Load frames from the video
    frames = sample_video_frames(video_path, num_frames)

    # Preprocess frames using CLIP processor
    inputs = clip_processor(images=frames, return_tensors="pt", padding=True)

    # Extract frame features using CLIP
    with torch.no_grad():
        frame_features = clip_model.get_image_features(inputs["pixel_values"])

    # Normalize the features and return them
    frame_features = frame_features / frame_features.norm(dim=-1, keepdim=True)
    return frame_features.cpu().numpy()

def sample_video_frames(video_path, num_frames=16):
    """
    Samples evenly spaced frames from a video.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.

    Returns:
        A list of PIL.Image objects representing sampled frames.
    """
    try:
        # Probe the video to get original width and height
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        original_width = int(video_stream['width'])
        original_height = int(video_stream['height'])

        out, _ = (
            ffmpeg
            .input(video_path, ss=0)  # Start from the beginning
            .filter('fps', fps=25)  # Limit frames per second
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, capture_stderr=True)
        )
        video_frames = np.frombuffer(out, np.uint8)
        video_frames = np.frombuffer(out, np.uint8).reshape([-1, original_height, original_width, 3])
        # print(video_frames.shape)

        # Evenly downsample frames to num_frames
        total_frames = video_frames.shape[0]
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        sampled_frames = [Image.fromarray(video_frames[i]) for i in frame_indices]
        
        return sampled_frames
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return [Image.new("RGB", (224, 224), color=(0, 0, 0))] * num_frames

def calculate_visual_variety(filtered_dataset, num_frames=16):
    all_features = []

    for idx, video in enumerate(filtered_dataset):
        if idx >= 25: break

        video_path = video["contentUrl"] 
        video_features = extract_video_features(video_path, num_frames=num_frames)
        all_features.append(video_features)

    if len(all_features) == 0:
        return None, None

    all_features = np.stack(all_features, axis=0)

    mean_embedding = np.mean(all_features, axis=(0, 1))
    variance_embedding = np.var(all_features, axis=(0, 1))

    return mean_embedding, variance_embedding

def count_clip_tokens(caption):
    """
    Tokenizes a caption using the CLIP tokenizer and returns the number of tokens.
    Args:
        caption: The text caption to tokenize.
    Returns:
        The number of tokens in the caption.
    """
    tokens = clip_processor.tokenizer(caption, truncation=True, return_tensors="pt")
    return tokens["input_ids"].shape[1]

def process_cluster(cluster_idx, cluster_key, cluster_captions, dataset, min_cluster_size=50):
    print("\n\n")
    print("-"*50, f"Processing cluster {cluster_idx + 1}/{len(clusters)}: {cluster_key}")
    
    first_caption = cluster_captions[0] if cluster_captions else "no_caption"
    start_time = datetime.now()

    def filter_videos(example):
        video_name = example["name"]
        return fuzz.ratio(video_name, first_caption) > 85

    filtered_dataset = dataset.filter(filter_videos)
    total_videos = len(filtered_dataset)
    
    # Exclude clusters below the minimum size
    if total_videos < min_cluster_size:
        print(f"Skipping Cluster {cluster_key}: too few videos ({total_videos} < {min_cluster_size}).")
        return None
    print(f"Processing Cluster {cluster_key} of Size {total_videos}")
    
    filtered_dataset = filtered_dataset.select(range(min_cluster_size))
    print(f"We are only using {len(filtered_dataset)} from this cluster.")
    
    avg_caption_length = (
        sum(count_clip_tokens(caption) for caption in cluster_captions) / len(cluster_captions)
        if cluster_captions else 0
    )
    processing_time = (datetime.now() - start_time).total_seconds()
    mean_embedding, variance_embedding = calculate_visual_variety(filtered_dataset)

    if total_videos == 0:
        print(f"Warning: Cluster {cluster_key} resulted in an empty dataset. Skipping...")
        return None

    save_folder_name = f"{OUTPUT_FOLDER}/filtered_webvid_dataset_{(cluster_idx+1):06d}_{re.sub(r'\W+', '_', first_caption.strip()[:30])}"
    filtered_dataset.save_to_disk(save_folder_name)

    row = {
        "cluster_key": cluster_key,
        "first_caption": first_caption,
        "total_videos": total_videos,
        "avg_caption_length": avg_caption_length,
        "mean_embedding": mean_embedding.tolist() if mean_embedding is not None else None,
        "variance_embedding": variance_embedding.tolist() if variance_embedding is not None else None,
        "processing_time_seconds": processing_time,
        "output_folder": save_folder_name,
    }
    pprint({key: value for key, value in row.items() if key not in {"mean_embedding", "variance_embedding"}})
    print("Intra-cluster visual variance", np.mean(row["variance_embedding"]))

    pickle_file_path = os.path.join(save_folder_name, "metadata.pkl")
    with open(pickle_file_path, "wb") as pickle_file:
        pickle.dump(row, pickle_file)
        
    for idx, video in enumerate(filtered_dataset):
        if idx >= 25: break

        gif_path = os.path.join(save_folder_name, f"video_{(idx+1):02d}.gif")
        try:
            video_path = video["contentUrl"]  # Assuming contentUrl points to the video file
            frames = sample_video_frames(video_path, num_frames=16)
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=100,  # Adjust speed of the GIF
                loop=0
            )
        except Exception as e:
            print(f"Failed to save GIF for video {idx + 1}: {e}")


    return row

def main():
    processed_data = []

    try:
        for cluster_idx, (cluster_key, cluster_captions) in enumerate(clusters.items()):
            if cluster_idx < 2552 or any(row["cluster_key"] == cluster_key for row in processed_data):
                continue  # Skip already processed clusters

            result = process_cluster(cluster_idx, cluster_key, cluster_captions, dataset)
            if result:
                processed_data.append(result)

    except KeyboardInterrupt:
        print("Processing interrupted by user (Ctrl+C). Saving current progress...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Save partial metadata to pickle
        with open(PARTIAL_SAVE_FILE, "wb") as f:
            pickle.dump(processed_data, f)
        print(f"Partial metadata saved to {PARTIAL_SAVE_FILE}")

        # Save metadata to CSV
        cluster_df = pd.DataFrame(processed_data)
        csv_file = os.path.join(OUTPUT_FOLDER, "cluster_summary.csv")
        cluster_df.to_csv(csv_file, index=False)
        print(f"Cluster metadata saved to CSV file: {csv_file}")

if __name__ == "__main__":
    main()
