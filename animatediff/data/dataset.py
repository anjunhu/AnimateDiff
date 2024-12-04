import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import ffmpeg
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
            video_folders: List[str]=["/home/ubuntu/video/AnimateDiff/__assets__/filtered_webvid_datasets/filtered_webvid_dataset_3_Fire flames igniting and burni",
                                    #   "/home/ubuntu/video/AnimateDiff/__assets__/filtered_webvid_datasets/filtered_webvid_dataset_1_Abstract background with beaut",
                                      "/home/ubuntu/video/AnimateDiff/__assets__/filtered_webvid_datasets/filtered_webvid_dataset_2_Aluminium can. 3d render of me",
                                      ],
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
            split="train"
        ):
        if not video_folders:
            print(f"Loading full dataset from Hugging Face ...")
            # Load dataset from Hugging Face
            self.dataset = load_dataset("TempoFunk/webvid-10M", split="train")
            self.length = len(self.dataset)
        else:
            datasets = []
            for folder in video_folders:
                print(f"Loading dataset from {folder}...")
                datasets.append(load_from_disk(folder))
            self.dataset = concatenate_datasets(datasets)
            self.length = len(self.dataset)
        print(f"data length: {self.length}")
        
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
    
    unique_train_captions = dataset.get_unique_names()
    print(len(unique_train_captions), unique_train_captions)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=16)
    # for idx, batch in enumerate(dataloader):
    #     print(batch["pixel_values"].shape, len(batch["text"]))
        # Optionally save the videos using save_videos_grid
