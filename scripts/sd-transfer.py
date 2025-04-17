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

# Constants for memorization detection
SSIM_MEM_THRESHOLD = 0.5
MSE_MEM_THRESHOLD = 2800
TARGET_ROOT = "/scratch/local/ssd/anjun/memorization/SDMemTarget/sdv1_bb/original"

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
    os.makedirs(os.path.join(savedir, "mem", "mse"), exist_ok=True)
    os.makedirs(os.path.join(savedir, "baseline", "mse"), exist_ok=True)
    os.makedirs(os.path.join(savedir, "mem", "mp4"), exist_ok=True)
    os.makedirs(os.path.join(savedir, "baseline", "mp4"), exist_ok=True)

    config = OmegaConf.load(args.config)
    samples = []
    
    # Create validation pipeline
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

    sample_idx = 0
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

        # Process memorized prompts
        data = load_dataset(args.mem_data_path, data_files={'train': 'sdv1_bb_edge_groundtruth.parquet'})
        process_memorized_prompts(pipeline, data, savedir, model_config, args, mem_curves, sample_idx)
        sample_idx = len(data['train'])  # Update index for baseline prompts

        # Process baseline prompts
        prompts = model_config.prompt
        n_prompts = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
        
        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
        
        config[model_idx].random_seed = []
        for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
            # Manually set random seed for reproduction
            if random_seed != -1: torch.manual_seed(random_seed)
            else: torch.seed()
            config[model_idx].random_seed.append(torch.initial_seed())
            
            print(f"Current seed: {torch.initial_seed()}")
            print(f"Processing baseline prompt ({prompt})...")
            
            # Generate the video
            sample = pipeline(
                prompt,
                negative_prompt=n_prompt,
                num_inference_steps=model_config.steps,
                guidance_scale=model_config.guidance_scale,
                width=model_config.W,
                height=model_config.H,
                video_length=model_config.L,
                controlnet_images=controlnet_images,
                controlnet_image_index=model_config.get("controlnet_image_indexs", [0]),
            )
            samples.append(sample.videos)
            
            # Calculate metrics
            safe_prompt = re.sub(r'\W+', '_', prompt)
            mse_xt = sample.mse_xt if hasattr(sample, 'mse_xt') else []
            mse_pred_x0 = sample.mse_pred_x0 if hasattr(sample, 'mse_pred_x0') else []
            noise_norms = [diff.norm(p=2).item() for diff in sample.noise_diff] if hasattr(sample, 'noise_diff') else []
            label = 0  # Baseline prompts are not memorized
            
            # Save baseline prompt outputs
            output_dir = os.path.join(savedir, "baseline", "mse")
            
            # Save trajectory data
            traj_path = os.path.join(output_dir, f"{sample_idx}_{label}_{safe_prompt}_traj.json")
            with open(traj_path, 'w') as f:
                json.dump({
                    "prompt": prompt,
                    "index": sample_idx,
                    "label": label,
                    "mse_xt": mse_xt,
                    "mse_pred_x0": mse_pred_x0,
                    "noise_diff_norms": noise_norms
                }, f, indent=2)
                
            # Store curves for plotting
            mem_curves[f"xt_{label}"].append(mse_xt)
            mem_curves[f"x0_{label}"].append(mse_pred_x0)
            mem_curves[f"diff_{label}"].append([a - b for a, b in zip(mse_xt, mse_pred_x0)] if mse_xt and mse_pred_x0 else [])
            mem_curves[f"noise_{label}"].append(noise_norms)
            
            # Save video frames as image grid
            vid_np = sample.videos[0].cpu().numpy()
            final_img = np.concatenate(vid_np.astype(np.uint8), axis=1)
            output_img_path = os.path.join(output_dir, f"{sample_idx}_{label}_{safe_prompt}.png")
            Image.fromarray(final_img).save(output_img_path)
            
            # Save the video using imageio
            video_output_path = os.path.join(savedir, "baseline", "mp4", f'{safe_prompt}.mp4')
            imageio.mimwrite(video_output_path, vid_np, fps=8, quality=9)
            
            # Create noise norm plot
            plt.figure(figsize=(10, 5))
            plt.plot(noise_norms, label='tc-uc noise norms')
            plt.xlabel("Denoising Step")
            plt.ylabel("Noise Norm")
            plt.title("Classifier-Free Guidance Noise Norms")
            plt.legend()
            noise_norm_plot_path = os.path.join(output_dir, f"{sample_idx}_{label}_{safe_prompt}_noise_norm_plot.png")
            plt.savefig(noise_norm_plot_path)
            plt.close()
            
            # Log to wandb if enabled
            if args.use_wandb:
                wandb.log({
                    f"{sample_idx}_{label}_{safe_prompt}_img": wandb.Image(output_img_path, caption=f"Label: {label}"),
                    f"{sample_idx}_{label}_{safe_prompt}_noise_norm_plot": wandb.Image(noise_norm_plot_path)
                })
            
            # Standard output for the original script
            short_prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
            save_videos_grid(sample.videos, f"{savedir}/sample/{sample_idx}-{short_prompt}.gif")
            print(f"Save to {savedir}/sample/{sample_idx}-{short_prompt}.gif")
            
            sample_idx += 1

    # Create aggregate plots if data is available
    if len(mem_curves["xt_0"]) > 0 or len(mem_curves["xt_1"]) > 0:
        create_aggregate_plots(savedir, mem_curves, args)

    # Save combined video grid (from original script)
    if samples:
        samples = torch.concat(samples)
        save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    # Save config
    OmegaConf.save(config, f"{savedir}/config.yaml")


def process_memorized_prompts(pipeline, data, savedir, model_config, args, mem_curves, start_idx):
    """Process memorized prompts from dataset and evaluate memorization."""
    
    for i in range(len(data['train'])):
        if data['train'][i]['overfit_type'] == args.mem_type.upper():
            prompt = data['train'][i]['caption']
            safe_prompt = re.sub(r'\W+', '_', prompt)
            
            # Check if target image exists
            target_glob = os.path.join(TARGET_ROOT, args.mem_type.lower(), f"{i:04d}_*.png")
            target_files = glob.glob(target_glob)
            if not target_files:
                continue
                
            print(f"Processing memorized prompt ({prompt})")
            
            # Generate the video
            sample = pipeline(
                prompt,
                negative_prompt="",
                num_inference_steps=model_config.steps,
                guidance_scale=model_config.guidance_scale,
                width=model_config.W,
                height=model_config.H,
                video_length=model_config.L,
            )
            
            video = sample.videos
            mse_xt = sample.mse_xt if hasattr(sample, 'mse_xt') else []
            mse_pred_x0 = sample.mse_pred_x0 if hasattr(sample, 'mse_pred_x0') else []
            
            # Compare to target image
            label = 0
            target_img = Image.open(target_files[0]).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((video.shape[-1], video.shape[-2])),
                transforms.ToTensor(),
            ])
            target_tensor = transform(target_img) * 255 # (3, 512, 512)
            
            # Check if any frame meets memorization criteria
            # video: (1, 3, 16, 512, 512) target_tensor: (3, 512, 512)
            minmse = float('inf')
            for f in range(video.shape[1]):
                frame_tensor = video[0, :, f]
                mse = F.mse_loss(frame_tensor, target_tensor)
                minmse = min(float(minmse), float(mse))
                if mse.item() < MSE_MEM_THRESHOLD:
                    label = 1
                    break
                    
            max_ssim = 0
            for f in range(video.shape[1]):
                frame = video[0, :, f]  # (3, 512, 512)
                frame_np = frame.permute(1, 2, 0).cpu().numpy() # (512, 512, 3)
                # Make sure it's in the right format for PIL (uint8 for normal images)
                if frame_np.dtype != np.uint8:
                    if frame_np.max() <= 1.0:
                        frame_np = (frame_np * 255).astype(np.uint8)
                    else:
                        frame_np = frame_np.astype(np.uint8)
                frame_pil = Image.fromarray(frame_np)
                frame_pil_resized = frame_pil.resize(target_img.size, Image.BILINEAR)
                
                # Evaluate SSIM in np uint8 space
                frame_np = np.array(frame_pil_resized)
                ssim_val = ssim(np.array(target_img), frame_np, multichannel=True, channel_axis=-1, data_range=255)
                max_ssim = max(max_ssim, ssim_val)
                if ssim_val > SSIM_MEM_THRESHOLD:
                    label = 1
                    break
            
            # Calculate noise norms
            noise_norms = [diff.norm(p=2).item() for diff in sample.noise_diff] if hasattr(sample, 'noise_diff') else []
            
            # Save outputs in mem directory
            output_dir = os.path.join(savedir, "mem", "mse")
            
            # Save trajectory data
            traj_path = os.path.join(output_dir, f"{i:04d}_{label}_mse{minmse:.4f}_{safe_prompt}_traj.json")
            with open(traj_path, 'w') as f:
                json.dump({
                    "prompt": prompt,
                    "index": i,
                    "label": label,
                    "memorized": int(label),
                    "mse_xt": mse_xt,
                    "mse_pred_x0": mse_pred_x0,
                    "noise_diff_norms": noise_norms,
                    "max_ssim": max_ssim,
                    "min_mse": float(minmse)
                }, f, indent=2)
            
            # Store curves for plotting
            mem_curves[f"xt_{label}"].append(mse_xt)
            mem_curves[f"x0_{label}"].append(mse_pred_x0)
            mem_curves[f"diff_{label}"].append([a - b for a, b in zip(mse_xt, mse_pred_x0)] if mse_xt and mse_pred_x0 else [])
            mem_curves[f"noise_{label}"].append(noise_norms)
            
            # Save concatenated image (target + video frames)
            vid_np = video[0].cpu().numpy()  # Shape (3, 16, 512, 512)
            vid_np = np.transpose(vid_np, (1, 2, 3, 0))
            target_np = target_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Shape (512, 512, 3)
            frames_list = [target_np]
            for f in range(vid_np.shape[0]):  # Iterate through 16 frames
                frame = vid_np[f, :, :]  # Shape (512, 512, 3)
                frames_list.append(frame_np)
            
            [print(frame.shape) for frame in frames_list]

            if frames_list[0].shape != frames_list[1].shape:
                print(f"Reshaping target from {frames_list[0].shape} to match {frames_list[1].shape}")
                target_pil = Image.fromarray(frames_list[0].astype(np.uint8))
                target_resized = target_pil.resize((frames_list[1].shape[1], frames_list[1].shape[0]), Image.BILINEAR)
                frames_list[0] = np.array(target_resized)

            frames_list = [frame.astype(np.uint8) for frame in frames_list]
            final_img = np.concatenate(frames_list, axis=1)  # Concatenate along width
            output_img_path = os.path.join(output_dir, f"{i:04d}_{label}_mse{minmse:.4f}_{safe_prompt}.png")
            Image.fromarray(final_img).save(output_img_path)

            # Create noise norm plot
            plt.figure(figsize=(10, 5))
            plt.plot(noise_norms, label='tc-uc noise norms')
            plt.xlabel("Denoising Step")
            plt.ylabel("Noise Norm")
            plt.title("Classifier-Free Guidance Noise Norms")
            plt.legend()
            noise_norm_plot_path = os.path.join(output_dir, f"{i:04d}_{label}_mse{minmse:.4f}_{safe_prompt}_noise_norm_plot.png")
            plt.savefig(noise_norm_plot_path)
            plt.close()
                        
            # Save the video
            video_output_path = os.path.join(savedir, "mem", "mp4", f'{safe_prompt}.mp4')
            imageio.mimwrite(video_output_path, vid_np, fps=8, quality=9)
            
            # Log to wandb if enabled
            if args.use_wandb:
                wandb.log({
                    f"{i:04d}_{label}_mse{minmse:.4f}_{safe_prompt}_img": wandb.Image(output_img_path, caption=f"Label: {label}"),
                    f"{i:04d}_{label}_mse{minmse:.4f}_{safe_prompt}_noise_norm_plot": wandb.Image(noise_norm_plot_path)
                })
            
            # Standard output for original script
            short_prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
            save_videos_grid(video, f"{savedir}/sample/{i}-{short_prompt}.gif")
            print(f"Save to {savedir}/sample/{i}-{short_prompt}.gif")


def create_aggregate_plots(savedir, curves, args):
    """Create aggregate plots for trajectory analysis."""
    
    # Compute global limits
    y_lims = {}
    for group in ["xt", "x0", "diff", "noise"]:
        all_vals = [v for k, lst in curves.items() if k.startswith(group) for traj in lst for v in traj]
        if all_vals:
            y_lims[group] = (min(all_vals), max(all_vals))
        else:
            y_lims[group] = (0, 1)  # default range
            print(f"Warning: No data found for group '{group}', using default y-limits.")

    # Plot 3x2 layout
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    titles = {
        (0, 0): "MSE(x_t, x₀) - Label 0",
        (0, 1): "MSE(x_t, x₀) - Label 1",
        (1, 0): "MSE(predₓ₀, x₀) - Label 0", 
        (1, 1): "MSE(predₓ₀, x₀) - Label 1",
        (2, 0): "ΔMSE = MSE(x_t) - MSE(predₓ₀) - Label 0",
        (2, 1): "ΔMSE = MSE(x_t) - MSE(predₓ₀) - Label 1"
    }

    colors = {
        "xt_0": "skyblue", "xt_1": "blue",
        "x0_0": "lightcoral", "x0_1": "red",
        "diff_0": "gray", "diff_1": "black"
    }
    alphas = {k: 0.3 for k in colors}
    alphas.update({k.replace("_0", "_1"): 0.8 for k in colors if "_0" in k})

    for (i, j), key_prefix in zip(
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
        ["xt_0", "xt_1", "x0_0", "x0_1", "diff_0", "diff_1"]
    ):
        ax = axes[i][j]
        if key_prefix in curves and curves[key_prefix]:
            arr = np.array(curves[key_prefix])
            if len(arr) > 0 and len(arr[0]) > 0:  # Check if there's data
                for traj in arr:
                    if len(traj) > 0:  # Skip empty trajectories
                        ax.plot(range(len(traj)), traj, color=colors[key_prefix], alpha=alphas[key_prefix], linewidth=0.5)
                
                # Only calculate mean and std if there are valid trajectories
                valid_trajs = [traj for traj in arr if len(traj) > 0]
                if valid_trajs:
                    # Ensure all trajectories have same length for mean calculation
                    max_len = max(len(traj) for traj in valid_trajs)
                    padded_trajs = [np.pad(traj, (0, max_len - len(traj)), 'constant', constant_values=np.nan) for traj in valid_trajs]
                    padded_arr = np.array(padded_trajs)
                    
                    # Calculate mean ignoring NaN values
                    mean = np.nanmean(padded_arr, axis=0)
                    std = np.nanstd(padded_arr, axis=0)
                    
                    ax.plot(range(len(mean)), mean, color=colors[key_prefix], linewidth=2.0)
                    ax.fill_between(range(len(mean)), mean - std, mean + std, color=colors[key_prefix], alpha=0.2)

        ax.set_title(titles[(i, j)])
        if len(curves[key_prefix]) > 0 and len(curves[key_prefix][0]) > 0:
            ax.set_xlim(0, len(curves[key_prefix][0])-1)
        else:
            ax.set_xlim(0, 49)  # Default range
            
        if key_prefix.split("_")[0] in y_lims:
            ax.set_ylim(y_lims[key_prefix.split("_")[0]])
            
        if i == 2:
            ax.set_xlabel("Denoising Step")
        if j == 0:
            ax.set_ylabel("MSE")
        ax.grid(True)

    fig.suptitle("Aggregated MSE Trajectories\nWith Mean ± Std", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    agg_plot_path = os.path.join(savedir, "aggregated_trajectory_3x2.png")
    plt.savefig(agg_plot_path)
    if args.use_wandb:
        wandb.log({"aggregated_trajectory_3x2": wandb.Image(agg_plot_path)})
    plt.close()

    # Create a noise norms plot
    plt.figure(figsize=(10, 5))
    
    for key, color in [("noise_0", "gray"), ("noise_1", "black")]:
        if key in curves and curves[key]:
            valid_trajs = [traj for traj in curves[key] if len(traj) > 0]
            if valid_trajs:
                for traj in valid_trajs:
                    plt.plot(range(len(traj)), traj, color=color, alpha=0.3 if key.endswith("_0") else 0.8, linewidth=0.5)
                
                # Ensure all trajectories have same length for mean calculation
                max_len = max(len(traj) for traj in valid_trajs)
                padded_trajs = [np.pad(traj, (0, max_len - len(traj)), 'constant', constant_values=np.nan) for traj in valid_trajs]
                padded_arr = np.array(padded_trajs)
                
                # Calculate mean ignoring NaN values
                mean = np.nanmean(padded_arr, axis=0)
                std = np.nanstd(padded_arr, axis=0)
                
                plt.plot(range(len(mean)), mean, color=color, linewidth=2.0)
                plt.fill_between(range(len(mean)), mean - std, mean + std, color=color, alpha=0.2)
    
    plt.title("Classifier-Free Guidance Noise Norms")
    plt.xlabel("Denoising Step")
    plt.ylabel("Noise Norm")
    plt.legend(["Non-memorized", "Memorized"])
    plt.grid(True)
    
    noise_plot_path = os.path.join(savedir, "noise_norms_plot.png")
    plt.savefig(noise_plot_path)
    if args.use_wandb:
        wandb.log({"noise_norms_plot": wandb.Image(noise_plot_path)})
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--inference-config", type=str, default="configs/inference/inference-v1.yaml")    
    parser.add_argument("--config", type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16, help="Video length")
    parser.add_argument("--W", type=int, default=512, help="Width")
    parser.add_argument("--H", type=int, default=512, help="Height")

    parser.add_argument("--without-xformers", action="store_true")
    
    # New arguments for memorization evaluation
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--evaluate-memorization", action="store_true", help="Enable memorization evaluation")
    parser.add_argument("--mem-type", type=str, default="mv")
    parser.add_argument("--mem-data-path", type=str, help="Path to memorization dataset",
                        default="/scratch/local/ssd/anjun/memorization/MVDream/data/one-step-extraction/")
                        
    
    args = parser.parse_args()
    main(args)