import os
os.environ["DISABLE_PROGRESSBAR"] = "1"
import re
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import subprocess
import wandb

import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from torchvision.models import resnet18
import lpips

from animatediff.data.dataset import WebVid10M
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, zero_rank_print, download_image
from datasets import load_dataset, concatenate_datasets, Dataset



def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)
        
    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)
        print(f"proc_id: {proc_id}; local_rank: {local_rank}; ntasks: {ntasks}; node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}")
        
    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')
    
    return local_rank



def main(
    image_finetune: bool,
    
    name: str,
    use_wandb: bool,
    launcher: str,
    
    output_dir: str,
    pretrained_model_path: str,

    train_data: Dict,
    validation_data: Dict,
    cfg_random_null_text: bool = True,
    cfg_random_null_text_ratio: float = 0.1,
    
    unet_checkpoint_path: str = "",
    unet_additional_kwargs: Dict = {},
    ema_decay: float = 0.9999,
    noise_scheduler_kwargs = None,
    
    max_train_epoch: int = -1,
    max_train_steps: int = 100,
    validation_steps: int = 100,
    validation_steps_tuple: Tuple = (-1,),

    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 0,
    lr_scheduler: str = "constant",

    trainable_modules: Tuple[str] = (None, ),
    num_workers: int = 32,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_epochs: int = 5,
    checkpointing_steps: int = -1,

    mixed_precision_training: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    is_debug: bool = False,
):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    local_rank      = init_dist(launcher=launcher)
    global_rank     = dist.get_rank()
    num_processes   = dist.get_world_size()
    is_main_process = True # global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)
    
    # Logging folder
    folder_name = "debug" if is_debug else name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)
    if is_debug and os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")

    *_, config = inspect.getargvalues(inspect.currentframe())

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if is_main_process and (not is_debug) and use_wandb:
        run = wandb.init(project="animatediff", name=folder_name, config=config)

    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex').eval().to(local_rank)

    # Initialize ResNet
    resnet_model = resnet18(pretrained=True)
    resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:6])
    resnet_model.eval().to(local_rank)
    
    # Define LPIPS-compatible preprocessing
    lpips_preprocess = torchvision.transforms.Compose([
                       torchvision.transforms.Resize((256, 256)),
                       torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                       ])


    def extract_resnet_features(images):
        """
        Extract features using shallow ResNet layers.
        Args:
            images: List of PIL.Image objects.
        Returns:
            Normalized feature tensor.
        """
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = torch.stack([preprocess(img) for img in images]).to(local_rank)
        with torch.no_grad():
            features = resnet_model(image_tensor)
        return features / features.norm(p=2, dim=1, keepdim=True)

    clip_model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer    = transformers.CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = transformers.CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    if not image_finetune:
        unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path, subfolder="unet", 
            unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
        
    # Load pretrained unet weights
    # unet_checkpoint_path = "outputs/training-2024-12-07T20-47-36/checkpoints/checkpoint.ckpt"
    if unet_checkpoint_path != "":
        print(f"from checkpoint: {unet_checkpoint_path}")
        unet_checkpoint_path = torch.load(unet_checkpoint_path, map_location="cpu")
        if "global_step" in unet_checkpoint_path: print(f"global_step: {unet_checkpoint_path['global_step']}")
        state_dict = unet_checkpoint_path["state_dict"] if "state_dict" in unet_checkpoint_path else unet_checkpoint_path
        new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        m, u = unet.load_state_dict(new_state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
        
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    clip_model.requires_grad_(False)
    
    # Set unet trainable parameters
    unet.requires_grad_(False)
    for name, param in unet.named_parameters():
        for trainable_module_name in trainable_modules:
            if trainable_module_name in name:
                param.requires_grad = True
                break
            
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    if is_main_process:
        print(f"trainable params number: {len(trainable_params)}")
        print(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    # Enable xformers
    # if enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         unet.enable_xformers_memory_efficient_attention()
    #     else:
    #         raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)
    clip_model.to(local_rank)
    
    def extract_clip_features(images):
        image_input = clip_processor(images=images, return_tensors="pt").to(clip_model.device)
        with torch.no_grad():
            features = clip_model.get_image_features(**image_input)
        return features / features.norm(p=2, dim=-1, keepdim=True)

    # Get the training dataset
    train_dataset = WebVid10M(**train_data, is_image=image_finetune)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    if is_main_process and (not is_debug) and use_wandb:
        wandb.config.update({
            "video_folders": os.listdir(train_dataset.video_folder),
            "num_samples": len(train_dataset),
        })

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    eval_dataset = train_dataset.eval_data

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)
        
    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    if scale_lr:
        learning_rate = (learning_rate * gradient_accumulation_steps * train_batch_size * num_processes)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    if not image_finetune:
        validation_pipeline = AnimationPipeline(
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler,
        ).to("cuda")
    else:
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=noise_scheduler, safety_checker=None,
        )
    validation_pipeline.enable_vae_slicing()

    # DDP warpper
    unet.to(local_rank)
    unet = DDP(unet, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    logging.info(f"  Num Epochs = {num_train_epochs}")
    logging.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logging.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logging.info(f"  Total optimization steps = {max_train_steps}")
    global_step = unet_checkpoint_path['global_step'] if "global_step" in unet_checkpoint_path else 0
    first_epoch = 0
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=True)
    progress_bar.set_description("Steps")

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        unet.train()
        
        for step, batch in enumerate(train_dataloader):
            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]
                
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        if idx > 10: break
                        pixel_value = pixel_value[None, ...]
                        save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.gif", rescale=True)
                else:
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value / 2. + 0.5
                        torchvision.utils.save_image(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.png")
                    
            ### >>>> Training >>>> ###
            
            # Convert videos to latent space            
            pixel_values = batch["pixel_values"].to(local_rank)
            video_length = pixel_values.shape[1]
            with torch.no_grad():
                if not image_finetune:
                    pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                else:
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()

                latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]
                
            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            # Mixed-precision training
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1
            
            ### <<<< Training <<<< ###
            
            # Wandb logging
            # if is_main_process and (not is_debug) and use_wandb:
            #     wandb.log({"train_loss": loss.item()}, step=global_step)
                
            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0 or step == len(train_dataloader) - 1):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "state_dict": unet.state_dict(),
                }
                if step == len(train_dataloader) - 1:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint-epoch-{epoch+1}.ckpt"))
                else:
                    torch.save(state_dict, os.path.join(save_path, f"checkpoint.ckpt"))
                logging.info(f"Saved state to {save_path} (global_step: {global_step})")
            
            generator = torch.Generator(device=latents.device)
            generator.manual_seed(global_seed)
            
            height = train_data.sample_size[0] if not isinstance(train_data.sample_size, int) else train_data.sample_size
            width  = train_data.sample_size[1] if not isinstance(train_data.sample_size, int) else train_data.sample_size
                
            # Periodically validation
            if is_main_process and (global_step % 1000 == 0 or global_step in [1,]):    
                print("#"*50, f"At Global Step {global_step} - Examine transferred memorisation from SD")
                sd_samples = []    
                
                ############################################################
                # Examine transferred memorisation from SD
                ############################################################
                sdv1_bb_edge = load_dataset("../one-step-extraction/", data_files={'train': 'sdv1_bb_edge_groundtruth.parquet',})
                # sdv1_wb = load_dataset("../one-step-extraction/", data_files={'train': 'sdv1_wb_groundtruth.parquet',})
                seen_captions = set()
                filtered_samples = []

                # Iterate through both datasets
                for dataset in [sdv1_bb_edge['train'], ]:
                    for sample in dataset:
                        caption = sample['caption']
                        if caption not in seen_captions and sample['overfit_type'] == 'MV':
                            seen_captions.add(caption)
                            filtered_samples.append(sample)
                sd_transfer_prompts = Dataset.from_list(filtered_samples)

                for idx, sample in enumerate(sd_transfer_prompts):
                    # Extract features for reference image
                    ref_pil_image = download_image(sample['url'])
                    if ref_pil_image is None: continue
                        
                    prompt = sample['caption']
                    if not image_finetune:
                        sample = validation_pipeline(
                            prompt,
                            generator    = generator,
                            video_length = train_data.sample_n_frames,
                            height       = height,
                            width        = width,
                            **validation_data,
                        ).videos
                        fn = re.sub(r'\W', '_', prompt)
                        save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/sd_{idx}_{fn}.gif")
                        sd_samples.append(sample)
                        
                        # BCTHW -> TCHW for wandb.Video
                        video_frames = rearrange(sample[0], "c t h w -> t c h w").detach().cpu()
                        video_frames_pil = [torchvision.transforms.ToPILImage()(frame) for frame in video_frames]
                        
                        # LPIPS similarity
                        lpips_scores = []
                        for frame in video_frames_pil:
                            ref_tensor = lpips_preprocess(ref_pil_image).unsqueeze(0).to(local_rank)
                            frame_tensor = lpips_preprocess(frame).unsqueeze(0).to(local_rank)
                            lpips_score = lpips_model(ref_tensor, frame_tensor).item()
                            lpips_scores.append(lpips_score)
                        max_lpips = np.max(lpips_scores)

                        # ResNet feature similarity
                        ref_features = extract_resnet_features([ref_pil_image] * len(video_frames_pil))
                        frame_features = extract_resnet_features(video_frames_pil)
                        resnet_similarities = F.cosine_similarity(ref_features, frame_features, dim=-1)
                        max_resnet_similarity = resnet_similarities.max().item()

                        # Extract features for reference images and generated frames
                        video_inputs = clip_processor(images=video_frames_pil, return_tensors="pt").to(local_rank)
                        with torch.no_grad():
                            video_features = clip_model.get_image_features(pixel_values=video_inputs['pixel_values'])
                            video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)                    
                        ref_features = extract_clip_features([ref_pil_image]*len(video_frames_pil))
                        # print(ref_features.shape, video_features.shape)
                        
                        # Compute cosine similarity
                        similarities = F.cosine_similarity(ref_features, video_features, dim=-1)
                        max_clip_similarity = similarities.max().item()

                        # Ensure pixel values are in [0, 255] and of type uint8
                        videos_np = (video_frames.numpy() * 255).clip(0, 255).astype(np.uint8)

                        if use_wandb:
                            wandb.log({
                                f"clip_sim/sd_{idx}_{fn}": max_clip_similarity,
                                f"res_sim/sd_{idx}_{fn}": max_resnet_similarity,
                                f"lpips/sd_{idx}_{fn}": max_lpips,
                                f"generated_videos/sd_{idx}_{fn}": wandb.Video(videos_np, caption=prompt, fps=4, format="gif"),
                                f"reference_images/sd_{idx}_{fn}": wandb.Image(ref_pil_image, caption=prompt)
                            }, step=global_step)
                            
                        print(f"SD Prompt: {prompt}, LPIPS: {max_lpips:.4f}, ResNet: {max_resnet_similarity:.4f}, CLIP: {max_clip_similarity:.4f}")
                    
                    else:
                        # Handle image_finetune case with StableDiffusionPipeline
                        pipeline_output = validation_pipeline(
                            prompt,
                            generator=generator,
                            height=height,
                            width=width,
                            **validation_data,
                        )
                        
                        # Extract images from StableDiffusionPipelineOutput
                        generated_images = pipeline_output.images  # List of PIL.Image objects
                        
                        fn = re.sub(r'\W', '_', prompt)
                        
                        # Convert PIL images to tensor for concatenation and saving
                        # Create a fake time dimension to reuse existing logic
                        image_tensors = []
                        for img in generated_images:
                            img_tensor = torchvision.transforms.ToTensor()(img)
                            image_tensors.append(img_tensor)
                        
                        # Stack images along a new time dimension (treating multiple images as frames)
                        if len(image_tensors) == 1:
                            # If only one image, duplicate it to create a fake time dimension
                            image_tensors = [image_tensors[0]] * train_data.get('sample_n_frames', 8)
                        
                        # Stack to create CTHW format (like video frames)
                        stacked_images = torch.stack(image_tensors, dim=1)  # Shape: [C, T, H, W]
                        stacked_images = stacked_images.unsqueeze(0)  # Add batch dimension: [1, C, T, H, W]
                        
                        # Save using existing video saving logic
                        save_videos_grid(stacked_images, f"{output_dir}/samples/sample-{global_step}/sd_{idx}_{fn}.gif")
                        sd_samples.append(stacked_images)
                        
                        # Convert to PIL images for similarity calculations (use first image)
                        generated_image_pil = generated_images[0]
                        
                        # LPIPS similarity (single image comparison)
                        ref_tensor = lpips_preprocess(ref_pil_image).unsqueeze(0).to(local_rank)
                        gen_tensor = lpips_preprocess(generated_image_pil).unsqueeze(0).to(local_rank)
                        lpips_score = lpips_model(ref_tensor, gen_tensor).item()

                        # ResNet feature similarity (single image comparison)
                        # ref_features = extract_resnet_features([ref_pil_image])
                        # gen_features = extract_resnet_features([generated_image_pil])
                        # resnet_similarity = F.cosine_similarity(ref_features, gen_features, dim=-1).item()

                        # CLIP feature similarity (single image comparison)
                        ref_clip_features = extract_clip_features([ref_pil_image])
                        gen_clip_features = extract_clip_features([generated_image_pil])
                        clip_similarity = F.cosine_similarity(ref_clip_features, gen_clip_features, dim=-1).item()

                        # Convert generated image to numpy for wandb logging
                        gen_image_np = np.array(generated_image_pil)

                        if use_wandb:
                            wandb.log({
                                f"clip_sim/sd_{idx}_{fn}": clip_similarity,
                                # f"res_sim/sd_{idx}_{fn}": resnet_similarity,
                                f"lpips/sd_{idx}_{fn}": lpips_score,
                                f"generated_images/sd_{idx}_{fn}": wandb.Image(generated_image_pil, caption=prompt),
                                f"reference_images/sd_{idx}_{fn}": wandb.Image(ref_pil_image, caption=prompt)
                            }, step=global_step)
                            
                        print(f"SD Prompt: {prompt}, LPIPS: {lpips_score:.4f}, CLIP: {clip_similarity:.4f}")
                
                if not image_finetune:
                    sd_samples = torch.concat(sd_samples)
                    save_path = f"{output_dir}/samples/sd_sample-{global_step}.gif"
                    save_videos_grid(sd_samples, save_path)
                else:
                    sd_samples = torch.concat(sd_samples)
                    save_path = f"{output_dir}/samples/sd_sample-{global_step}.gif"
                    save_videos_grid(sd_samples, save_path)
                    
            if is_main_process and (global_step % (len(train_dataset)//2) == 0 or global_step in [1,]):   
                print("#"*50, f"At Global Step {global_step} - Examine memorisation from WebVid10M")
                wv_samples = []    
                
                ############################################################
                # Examine memorisation from video dataset (WebVid10M)
                ############################################################
                for idx, sample in enumerate(eval_dataset):
                    prompt = sample["first_caption"]
                    reference_visual = sample["reference_visual"]  # Reference visual in numpy format [F, H, W, 3]

                    # Preprocess reference_visual for LPIPS, ResNet, and CLIP evaluation
                    reference_frames = [torchvision.transforms.functional.to_tensor(frame).to(local_rank) for frame in reference_visual]
                    reference_frames = [torchvision.transforms.functional.normalize(frame, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) for frame in reference_frames]

                    # Generate videos or images based on the pipeline
                    if not image_finetune:
                        generated_video = validation_pipeline(
                            prompt,
                            generator=generator,
                            video_length=train_data.sample_n_frames,
                            height=height,
                            width=width,
                            **validation_data,
                        ).videos

                        fn = re.sub(r'\W', '_', prompt)
                        save_videos_grid(generated_video, f"{output_dir}/samples/sample-{global_step}/wv_{idx}_{fn}.gif")
                        wv_samples.append(generated_video)
                        
                        # BCTHW -> TCHW for wandb.Video
                        video_frames = rearrange(generated_video[0], "c t h w -> t c h w").detach().cpu()
                        video_frames_pil = [torchvision.transforms.ToPILImage()(frame) for frame in video_frames]

                        # LPIPS similarity between two VIDEOS
                        ref_frames = [torchvision.transforms.ToPILImage()(torch.from_numpy(frame).permute(2, 0, 1)) for frame in reference_visual]

                        # Ensure same number of frames for comparison
                        min_frames = min(len(video_frames_pil), len(ref_frames))
                        video_frames_pil = video_frames_pil[:min_frames]
                        ref_frames = ref_frames[::2]

                        lpips_scores = []
                        for gen_frame, ref_frame in zip(video_frames_pil, ref_frames):
                            ref_tensor = lpips_preprocess(ref_frame).unsqueeze(0).to(local_rank)
                            frame_tensor = lpips_preprocess(gen_frame).unsqueeze(0).to(local_rank)
                            lpips_score = lpips_model(ref_tensor, frame_tensor).item()
                            lpips_scores.append(lpips_score)
                        max_lpips_wv = np.max(lpips_scores)

                        # ResNet feature similarity between two VIDEOS
                        ref_features = extract_resnet_features(ref_frames)
                        frame_features = extract_resnet_features(video_frames_pil)
                        resnet_similarities = F.cosine_similarity(ref_features, frame_features, dim=-1)
                        max_resnet_similarity_wv = resnet_similarities.max().item()

                        # Extract CLIP features for generated frames and reference video frames
                        video_inputs = clip_processor(images=video_frames_pil, return_tensors="pt").to(local_rank)
                        ref_inputs = clip_processor(images=ref_frames, return_tensors="pt").to(local_rank)

                        with torch.no_grad():
                            video_features = clip_model.get_image_features(pixel_values=video_inputs['pixel_values'])
                            video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
                            
                            ref_features = clip_model.get_image_features(pixel_values=ref_inputs['pixel_values'])
                            ref_features = ref_features / ref_features.norm(p=2, dim=-1, keepdim=True)

                        # Compute cosine similarity between two VIDEOS
                        similarities = F.cosine_similarity(ref_features, video_features, dim=-1)
                        max_clip_similarity_wv = similarities.max().item()

                        # Prepare reference visual for wandb logging
                        # Convert numpy array to tensor and ensure it's in the right format (T, C, H, W)
                        reference_visual_tensor = torch.from_numpy(reference_visual).float()
                        reference_visual_tensor = reference_visual_tensor.permute(0, 3, 1, 2)  # (F, H, W, 3) -> (F, 3, H, W)
                        # Scale values to [0, 255] and convert to uint8
                        reference_visual_wandb = (reference_visual_tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
                        print(reference_visual_wandb.shape)

                        # Ensure pixel values are in [0, 255] and of type uint8
                        videos_np = (video_frames.numpy() * 255).clip(0, 255).astype(np.uint8)

                        # Log results to wandb
                        if use_wandb:
                            wandb.log({
                                f"clip_sim/wv_{idx}_{fn}": max_clip_similarity_wv,
                                f"res_sim/wv_{idx}_{fn}": max_resnet_similarity_wv,
                                f"lpips/wv_{idx}_{fn}": max_lpips_wv,
                                f"generated_videos/wv_{idx}_{fn}": wandb.Video(videos_np, caption=prompt, fps=4, format="gif"),
                                f"reference_videos/wv_{idx}_{fn}": wandb.Video(np.stack(reference_visual_wandb), caption=prompt, fps=4, format="gif"),
                            }, step=global_step)

                        print(f"WV Prompt: {prompt}, LPIPS: {max_lpips_wv:.4f}, ResNet: {max_resnet_similarity_wv:.4f}, CLIP: {max_clip_similarity_wv:.4f}")

                    else:
                        # Handle image_finetune case for WebVid10M evaluation
                        pipeline_output = validation_pipeline(
                            prompt,
                            generator=generator,
                            height=height,
                            width=width,
                            **validation_data,
                        )
                        
                        # Extract images from StableDiffusionPipelineOutput
                        generated_images = pipeline_output.images  # List of PIL.Image objects
                        generated_image_pil = generated_images[0]  # Use first generated image
                        
                        fn = re.sub(r'\W', '_', prompt)
                        
                        # Convert PIL images to tensor for concatenation and saving
                        # Create a fake time dimension to reuse existing logic
                        image_tensors = []
                        for img in generated_images:
                            img_tensor = torchvision.transforms.ToTensor()(img)
                            image_tensors.append(img_tensor)
                        
                        # Stack images along a new time dimension (treating multiple images as frames)
                        if len(image_tensors) == 1:
                            # If only one image, duplicate it to create a fake time dimension
                            image_tensors = [image_tensors[0]] * train_data.get('sample_n_frames', 8)
                        
                        # Stack to create CTHW format (like video frames)
                        stacked_images = torch.stack(image_tensors, dim=1)  # Shape: [C, T, H, W]
                        stacked_images = stacked_images.unsqueeze(0)  # Add batch dimension: [1, C, T, H, W]
                        
                        # Save using existing video saving logic
                        save_videos_grid(stacked_images, f"{output_dir}/samples/sample-{global_step}/wv_{idx}_{fn}.gif")
                        wv_samples.append(stacked_images)
                        
                        # For comparison, use the first reference frame as the reference image
                        ref_image_pil = torchvision.transforms.ToPILImage()(torch.from_numpy(reference_visual[0]).permute(2, 0, 1))
                        
                        # LPIPS similarity (single image comparison)
                        ref_tensor = lpips_preprocess(ref_image_pil).unsqueeze(0).to(local_rank)
                        gen_tensor = lpips_preprocess(generated_image_pil).unsqueeze(0).to(local_rank)
                        lpips_score_wv = lpips_model(ref_tensor, gen_tensor).item()

                        # ResNet feature similarity (single image comparison)
                        ref_features = extract_resnet_features([ref_image_pil])
                        gen_features = extract_resnet_features([generated_image_pil])
                        resnet_similarity_wv = F.cosine_similarity(ref_features, gen_features, dim=-1).item()

                        # CLIP feature similarity (single image comparison)
                        ref_clip_features = extract_clip_features([ref_image_pil])
                        gen_clip_features = extract_clip_features([generated_image_pil])
                        clip_similarity_wv = F.cosine_similarity(ref_clip_features, gen_clip_features, dim=-1).item()

                        # Log results to wandb
                        if use_wandb:
                            wandb.log({
                                f"clip_sim/wv_{idx}_{fn}": clip_similarity_wv,
                                f"res_sim/wv_{idx}_{fn}": resnet_similarity_wv,
                                f"lpips/wv_{idx}_{fn}": lpips_score_wv,
                                f"generated_images/wv_{idx}_{fn}": wandb.Image(generated_image_pil, caption=prompt),
                                f"reference_images/wv_{idx}_{fn}": wandb.Image(ref_image_pil, caption=prompt)
                            }, step=global_step)

                        print(f"WV Prompt: {prompt}, LPIPS: {lpips_score_wv:.4f}, ResNet: {resnet_similarity_wv:.4f}, CLIP: {clip_similarity_wv:.4f}")

                if not image_finetune:
                    wvsamples = torch.cat(wv_samples)
                    save_path = f"{output_dir}/samples/wvsample-{global_step}.gif"
                    save_videos_grid(wvsamples, save_path)
                else:
                    wvsamples = torch.cat(wv_samples)
                    save_path = f"{output_dir}/samples/wvsample-{global_step}.gif"
                    save_videos_grid(wvsamples, save_path)
                logging.info(f"Saved samples to {save_path}")

                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
            
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb",    action="store_true")
    args = parser.parse_args()

    name   = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, use_wandb=args.wandb, **config)
