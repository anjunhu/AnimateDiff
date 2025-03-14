import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import wandb
import random
import ffmpeg
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from einops import rearrange
from decord import VideoReader
from datasets import load_dataset, load_from_disk, concatenate_datasets
from animatediff.utils.util import zero_rank_print
import requests
from typing import List


class WebVid10M(Dataset):
    def __init__(
            self,
            video_folders = "/home/ubuntu/video/AnimateDiff/__assets__/filtered_webvid_datasets/",
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False, n_eval_per_cluster=1,
            split="train"
        ):
        self.video_folders = video_folders
        self.eval_data = [] 
        
        if not video_folders:
            print(f"Loading full dataset from Hugging Face ...")
            # Load dataset from Hugging Face
            self.dataset = load_dataset("TempoFunk/webvid-10M", split="train")
            self.length = len(self.dataset)
        else:
            self.list_of_cluster_dataset_dirs = sorted([folder for folder in os.listdir(video_folders)
                                                        if folder.startswith("filtered_webvid_dataset_") and 
                                                        os.path.isdir(os.path.join(video_folders, folder)) ])
            datasets = []
            for folder in self.list_of_cluster_dataset_dirs:
                cluster_path = os.path.join(video_folders, folder)
                print(f"Loading dataset from {folder}...")
                
                # Attempt to load reference visual and metadata
                try:
                    metadata_file = os.path.join(cluster_path, "metadata.pkl")
                    if not os.path.exists(metadata_file):
                        print(f"Metadata file not found in {cluster_path}. Skipping...")
                        continue

                    # Load metadata
                    with open(metadata_file, "rb") as f:
                        metadata = pickle.load(f)

                    # Load reference visual
                    reference_visual = self.download_video(
                        load_from_disk(os.path.join(cluster_path))[0]['contentUrl']
                    )

                    # Add to eval_data
                    first_caption = metadata.get("first_caption", None)
                    if first_caption:
                        for _ in range(n_eval_per_cluster):
                            sample_dict = metadata.copy()
                            sample_dict.update({'reference_visual': reference_visual})
                            self.eval_data.append(sample_dict)
                except Exception as e:
                    print(f"Error processing cluster {folder}: {e}")
                    continue

                # Load dataset from disk and add to combined dataset
                datasets.append(load_from_disk(cluster_path))
            
            self.dataset = concatenate_datasets(datasets)
            self.length = len(self.dataset)
        
        print(f"training data length: {self.length}")
        
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def get_eval_data(self):
        """Retrieve precomputed evaluation data."""
        return self.eval_data
        
    def get_unique_names(self):
        print("Collecting unique training prompts...")
        names = [item['name'].strip() for item in self.dataset if 'name' in item]
        unique_names = list(set(names))
        return unique_names
    
    def is_valid_video_link(self, contentUrl):
        try:
            response = requests.head(contentUrl, timeout=5)
            # print(response)
            # Check if we get a valid response code
            return response.status_code == 200
        except requests.RequestException:
            return False
        
    def download_video(self, url, width=256, height=256, num_frames=16):
        """Fetch video frames using ffmpeg from a URL."""
        try:
            out, _ = (
                ffmpeg
                .input(url, ss=0)  # Start from the beginning
                .filter('fps', fps=25)  # Limit frames per second
                .filter('scale', width, height)  # Resize the video
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=num_frames)  # Limit to num_frames
                .run(capture_stdout=True, capture_stderr=True)
            )
            video_frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            return video_frames
        except Exception as e:
            print(f"Error fetching video: {e}")
            return None
        
    def get_batch(self, idx):
        # Load data for the given index
        video_dict = self.dataset[idx]
        contentUrl = video_dict['contentUrl'] 

        # Check if the video link is valid
        if not self.is_valid_video_link(contentUrl):
            raise ValueError(f"Invalid video link: {contentUrl}")

        # Fetch video data using ffmpeg-python
        video_data = self.download_video(contentUrl, width=256, height=256, num_frames=self.sample_n_frames)
        if video_data is None:
            raise ValueError(f"Failed to download video: {contentUrl}")
        
        # Convert video frames to a PyTorch tensor and normalize
        pixel_values = torch.from_numpy(video_data).permute(0, 3, 1, 2).contiguous()  # (frames, channels, height, width)
        pixel_values = pixel_values / 255.0  # Normalize pixel values

        if self.is_image:
            pixel_values = pixel_values[0]  # In case of image, take a single frame
        
        return pixel_values, video_dict['name']  # Replace 'name' with the appropriate field

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break
            except ValueError as e:
                print(f"Skipping video at index {idx}: {str(e)}")
                idx = random.randint(0, self.length-1)
            except Exception as e:
                print(f"Error processing video at index {idx}: {str(e)}")
                idx = random.randint(0, self.length-1)

        # Apply transformations
        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        return sample


if __name__ == "__main__":
    from animatediff.utils.util import save_videos_grid

    dataset = WebVid10M(
        # video_folders=None, # None = full dataset
        sample_size=256,
        sample_stride=4, sample_n_frames=16,
        is_image=False,
        split="train"  # Specify the split (train, val, etc.)
    )
    
    # unique_train_captions = dataset.get_unique_names()
    # print(len(unique_train_captions), unique_train_captions)
    
    print(dataset.get_n_captions_per_cluster())
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=16)
    # for idx, batch in enumerate(dataloader):
    #     print(batch["pixel_values"].shape, len(batch["text"]))
        # Optionally save the videos using save_videos_grid
