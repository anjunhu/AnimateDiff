import argparse
import datetime
import inspect
import os
import re
import json
import glob
import sys
from omegaconf import OmegaConf
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from datasets import load_dataset
from skimage.metrics import structural_similarity as ssim

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights, auto_download
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import imageio
import wandb

@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(project=args.wandb_project)

    # Setup output directories
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(os.path.join(savedir, "sample"), exist_ok=True)
    os.makedirs(os.path.join(savedir, "clusters"), exist_ok=True)

    # Load cluster data
    clusters_file = "./__assets__/clusters-10-500.json"
    with open(clusters_file, 'r') as f:
        clusters = json.load(f)
    
    config = OmegaConf.load(args.config)
    samples = []
    
    # Create validation pipeline
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

    mem_curves = defaultdict(list)  # For storing trajectory data

    for model_idx, model_config in enumerate(config):
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)

        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", 
            unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).cuda()

        # Load controlnet model
        controlnet = controlnet_images = None
        if model_config.get("controlnet_path", "") != "":
            # (Controlnet loading code unchanged)
            assert model_config.get("controlnet_images", "") != ""
            assert model_config.get("controlnet_config", "") != ""
            
            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None

            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))

            auto_download(model_config.controlnet_path, is_dreambooth_lora=False)
            print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
            controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
            controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict = {name: param for name, param in controlnet_state_dict.items() if "pos_encoder.pe" not in name}
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            controlnet.cuda()

            image_paths = model_config.controlnet_images
            if isinstance(image_paths, str): image_paths = [image_paths]

            print(f"controlnet image paths:")
            for path in image_paths: print(path)
            assert len(image_paths) <= model_config.L

            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (model_config.H, model_config.W), (1.0, 1.0), 
                    ratio=(model_config.W/model_config.H, model_config.W/model_config.H)
                ),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else: image_norm = lambda x: x
                
            controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]

            os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
            for i, image in enumerate(controlnet_images):
                Image.fromarray((255. * (image.numpy().transpose(1,2,0))).astype(np.uint8)).save(f"{savedir}/control_images/{i}.png")

            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

            if controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

        # Set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

        pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path=model_config.get("motion_module", ""),
            motion_module_lora_configs=model_config.get("motion_module_lora_configs", []),
            # domain adapter
            adapter_lora_path=model_config.get("adapter_lora_path", ""),
            adapter_lora_scale=model_config.get("adapter_lora_scale", 1.0),
            # image layers
            dreambooth_model_path=model_config.get("dreambooth_path", ""),
            lora_model_path=model_config.get("lora_model_path", ""),
            lora_alpha=model_config.get("lora_alpha", 0.8),
        ).to("cuda")

        # Process each cluster
        for cluster_idx, prompts_list in tqdm(clusters.items(), desc="Processing clusters"):
            cluster_dir = os.path.join(savedir, "clusters", f"{int(cluster_idx):05d}")
            os.makedirs(cluster_dir, exist_ok=True)
            os.makedirs(os.path.join(cluster_dir, "plots"), exist_ok=True)
            
            # Process each prompt in the cluster
            for prompt_idx, prompt in enumerate(prompts_list[:args.n_per_cluster]):
                safe_prompt = re.sub(r'\W+', '_', prompt)
                print(f"Processing cluster {cluster_idx}, prompt: {prompt}")
                
                # Generate the video
                sample = pipeline(
                    prompt,
                    negative_prompt="",
                    num_inference_steps=model_config.steps,
                    guidance_scale=model_config.guidance_scale,
                    width=model_config.W,
                    height=model_config.H,
                    video_length=model_config.L,
                    controlnet_images=controlnet_images,
                    controlnet_image_index=model_config.get("controlnet_image_indexs", [0]),
                )
                samples.append(sample.videos)
                
                # Extract trajectory data
                mse_xt = sample.mse_xt if hasattr(sample, 'mse_xt') else []
                mse_pred_x0 = sample.mse_pred_x0 if hasattr(sample, 'mse_pred_x0') else []
                noise_norms = [diff.norm(p=2).item() for diff in sample.noise_diff] if hasattr(sample, 'noise_diff') else []
                
                # Save trajectory data
                traj_path = os.path.join(cluster_dir, f"{int(cluster_idx):05d}_{prompt_idx:03d}_{safe_prompt}_traj.json")
                with open(traj_path, 'w') as f:
                    json.dump({
                        "prompt": prompt,
                        "cluster_idx": int(cluster_idx),
                        "prompt_idx": prompt_idx,
                        "mse_xt": mse_xt,
                        "mse_pred_x0": mse_pred_x0,
                        "memorized": False,
                        "noise_diff_norms": noise_norms
                    }, f, indent=2)
                
                # Create noise norm plot
                if noise_norms:
                    plt.figure(figsize=(10, 5))
                    plt.plot(noise_norms, label='tc-uc noise norms')
                    plt.xlabel("Denoising Step")
                    plt.ylabel("Noise Norm")
                    plt.title(f"Classifier-Free Guidance Noise Norms - Cluster {int(cluster_idx):05d}")
                    plt.legend()
                    noise_norm_plot_path = os.path.join(cluster_dir, "plots", f"{int(cluster_idx):05d}_{prompt_idx:03d}_{safe_prompt}_noise_norm_plot.png")
                    plt.savefig(noise_norm_plot_path)
                    plt.close()
                
                # Create placeholder reference image with gray color (128, 128, 128)
                video = sample.videos
                placeholder = np.ones((model_config.H, model_config.W, 3), dtype=np.uint8) * 128
                
                # Save video frames with placeholder reference
                vid_np = video[0].cpu().numpy()  # Shape (3, 16, 512, 512)
                vid_np = np.transpose(vid_np, (1, 2, 3, 0))  # Shape (16, 512, 512, 3)
                
                # Create list of frames to concatenate
                frames_list = [placeholder]  # Start with placeholder reference
                for f in range(vid_np.shape[0]):  # Add video frames
                    frame = vid_np[f]  # Shape (512, 512, 3)
                    if frame.max() <= 1.0:  # Check if normalized [0,1] and convert to [0,255]
                        frame = (frame * 255).astype(np.uint8)
                    frames_list.append(frame.astype(np.uint8))
                
                # Concatenate frames horizontally
                final_img = np.concatenate(frames_list, axis=1)
                output_img_path = os.path.join(cluster_dir, f"{int(cluster_idx):05d}_{prompt_idx:03d}_{safe_prompt}.png")
                Image.fromarray(final_img).save(output_img_path)
                
                # Save the video using imageio
                video_output_path = os.path.join(cluster_dir, f"{int(cluster_idx):05d}_{prompt_idx:03d}_{safe_prompt}.mp4")
                imageio.mimwrite(video_output_path, vid_np, fps=8, quality=9)

                # Create noise norm plot
                plt.figure(figsize=(10, 5))
                plt.plot(noise_norms, label='tc-uc noise norms')
                plt.xlabel("Denoising Step")
                plt.ylabel("Noise Norm")
                plt.title("Classifier-Free Guidance Noise Norms")
                plt.legend()
                noise_norm_plot_path = os.path.join(cluster_dir, f"{int(cluster_idx):05d}_{prompt_idx:03d}_{safe_prompt}_noise_norm_plot.png")
                plt.savefig(noise_norm_plot_path)
                plt.close()
                
                # Log to wandb if enabled
                if args.use_wandb:
                    wandb.log({
                        f"cluster_{int(cluster_idx):05d}_prompt_{prompt_idx:03d}": wandb.Image(output_img_path, caption=prompt),
                        f"cluster_{int(cluster_idx):05d}_prompt_{prompt_idx:03d}_noise_norm": wandb.Image(noise_norm_plot_path) if noise_norms else None
                    })
                
                # Standard output for the original script
                short_prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                save_videos_grid(sample.videos, f"{savedir}/sample/{int(cluster_idx):05d}_{prompt_idx:03d}-{short_prompt}.gif")
                print(f"Save to {savedir}/sample/{int(cluster_idx):05d}_{prompt_idx:03d}-{short_prompt}.gif")

    # Save combined video grid (from original script)
    if samples:
        samples = torch.concat(samples)
        save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    # Save config
    OmegaConf.save(config, f"{savedir}/config.yaml")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--inference-config", type=str, default="configs/inference/inference-v1.yaml")    
    parser.add_argument("--config", type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16, help="Video length")
    parser.add_argument("--W", type=int, default=512, help="Width")
    parser.add_argument("--H", type=int, default=512, help="Height")

    parser.add_argument("--without-xformers", action="store_true")
    
    # Arguments for wandb
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="animatediff-clusters")
    parser.add_argument("--n-per-cluster", type=int, default=10)
    
    
    args = parser.parse_args()
    main(args)