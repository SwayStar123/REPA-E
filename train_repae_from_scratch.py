import argparse
import copy
import logging
import os
import json
import math
from pathlib import Path
from collections import OrderedDict

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import wandb

from dataset import CustomINH5Dataset, CustomDirDataset
from loss.losses import ReconstructionLoss_Single_Stage, compute_alignment_loss
from models.invae import vae_models
from models.meikai import AE_F32D256
from models.sit import SiT_models
from samplers import euler_sampler
from utils import load_encoders, normalize_latents, denormalize_latents, preprocess_imgs_vae, count_trainable_params

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z - latents_bias) * latents_scale # normalize
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    # Also perform EMA on BN buffers
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())

    for name, buffer in model_buffers.items():
        name = name.replace("module.", "")
        if buffer.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            # Apply EMA only to float buffers
            ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)
        else:
            # Direct copy for non-float buffers
            ema_buffers[name].copy_(buffer)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # set up the logger and checkpoint dirs
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    if args.vae == "f8d4":
        assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        latent_size = args.resolution // 8
        in_channels = 4
    elif args.vae == "f16d32":
        assert args.resolution % 16 == 0, "Image size must be divisible by 16 (for the VAE encoder)."
        latent_size = args.resolution // 16
        in_channels = 32
    elif args.vae == "f32d256":
        assert args.resolution % 32 == 0, "Image size must be divisible by 32 (for the MeiKai encoder)."
        latent_size = args.resolution // 32
        in_channels = 256
    else:
        raise NotImplementedError()

    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
        )
    else:
        raise NotImplementedError()
    z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]

    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        class_dropout_prob=args.cfg_prob,
        z_dims=z_dims,
        encoder_depth=args.encoder_depth,
        bn_momentum=args.bn_momentum,
        **block_kwargs
    )

    # make a copy of the model for EMA
    model = model.to(device)
    ema = copy.deepcopy(model).to(device)  # Create an EMA of the model for use after training

    # Create autoencoder
    if args.vae == "f32d256":
        # Use MeiKai autoencoder with f=32 compression
        ae = AE_F32D256().to(device)
        use_vae = False  # MeiKai is a deterministic autoencoder, not VAE
    else:
        # Use traditional VAE
        ae = vae_models[args.vae]().to(device)
        use_vae = True
    
    requires_grad(ema, False)
    print(f"Total trainable params in {'AE' if not use_vae else 'VAE'}:", count_trainable_params(ae))

    # Initialize latents stats with neutral defaults for BN layer initialization
    # These will be learned during training from scratch
    latents_scale = torch.ones(in_channels, device=device)
    latents_bias = torch.zeros(in_channels, device=device)

    model.init_bn(latents_bias=latents_bias, latents_scale=latents_scale)

    # Apply SyncBN if more than 1 GPU is used
    if accelerator.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    loss_cfg = OmegaConf.load(args.loss_cfg_path)
    ae_loss_fn = ReconstructionLoss_Single_Stage(loss_cfg).to(device)

    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Define the optimizers for SiT, AE/VAE, and AE/VAE loss function separately
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_ae = torch.optim.AdamW(
        ae.parameters(),
        lr=args.vae_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_loss_fn = torch.optim.AdamW(
        ae_loss_fn.parameters(),
        lr=args.disc_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup data
    if args.dataset_type == "h5":
        train_dataset = CustomINH5Dataset(args.data_dir)
    elif args.dataset_type == "dir":
        train_dataset = CustomDirDataset(args.data_dir)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")
    
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    # Start with eval mode for all models
    model.eval()
    ema.eval()
    ae.eval()

    if args.disc_pretrained_ckpt is not None:
        # Load the discriminator from a pretrained checkpoint if provided
        disc_ckpt = torch.load(args.disc_pretrained_ckpt, map_location=device)
        ae_loss_fn.discriminator.load_state_dict(disc_ckpt)
        if accelerator.is_main_process:
            logger.info(f"Loaded discriminator from {args.disc_pretrained_ckpt}")

    # resume
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt_path = f'{args.cont_dir}/checkpoints/{ckpt_name}'

        # If the checkpoint exists, we load the checkpoint and resume the training
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        ae.load_state_dict(ckpt['ae'])
        ae_loss_fn.discriminator.load_state_dict(ckpt['discriminator'])
        optimizer.load_state_dict(ckpt['opt']),
        optimizer_ae.load_state_dict(ckpt['opt_ae'])
        optimizer_loss_fn.load_state_dict(ckpt['opt_disc'])
        global_step = ckpt['steps']

    # Allow larger cache size for DYNAMo compilation
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    # Model compilation for better performance
    if args.compile:
        model = torch.compile(model, backend="inductor", mode="default")
        ae = torch.compile(ae, backend="inductor", mode="default")
        ae_loss_fn = torch.compile(ae_loss_fn, backend="inductor", mode="default")

    model, ae, ae_loss_fn, optimizer, optimizer_ae, optimizer_loss_fn, train_dataloader = accelerator.prepare(
        model, ae, ae_loss_fn, optimizer, optimizer_ae, optimizer_loss_fn, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="gradient-pass-through",
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, in_channels, latent_size, latent_size), device=device)

    # main training loop
    for epoch in range(args.epochs):
        model.train()

        for raw_image, y in train_dataloader:
            raw_image = raw_image.to(device)
            labels = y.to(device)
            z = None

            # extract the dinov2 features
            with torch.no_grad():
                zs = []
                zs_f16 = []  # Keep original 16x16 resolution for encoder/decoder alignment
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        
                        # For f=32 autoencoder, downsample DINO features from 16x16 to 8x8
                        # to match the latent resolution for main REPA loss
                        if args.vae == "f32d256":
                            # z shape: [B, 256, 768] (16*16=256 tokens)
                            # Need to downsample to [B, 64, 768] (8*8=64 tokens)
                            bsz, n_tokens, feat_dim = z.shape
                            h = w = int(n_tokens ** 0.5)  # 16
                            z_reshaped = z.reshape(bsz, h, w, feat_dim).permute(0, 3, 1, 2)  # [B, 768, 16, 16]
                            z_downsampled = torch.nn.functional.avg_pool2d(z_reshaped, kernel_size=2, stride=2)  # [B, 768, 8, 8]
                            z_downsampled = z_downsampled.permute(0, 2, 3, 1).reshape(bsz, -1, feat_dim)  # [B, 64, 768]
                            
                            zs_f16.append(z)  # Keep original 16x16 for encoder/decoder alignment
                            zs.append(z_downsampled)  # Use 8x8 for main REPA loss
                        else:
                            # For other VAEs, use original resolution
                            zs.append(z)
                            zs_f16.append(z)

            ae.train()
            model.train()
            with accelerator.accumulate([model, ae, ae_loss_fn]), accelerator.autocast():
                # 1). Forward pass: Autoencoder
                processed_image = preprocess_imgs_vae(raw_image)
                
                # For VAE: get posterior, sample z, and reconstruct
                # For MeiKai AE: get z and align_proj from encoder, decode with decoder
                if use_vae:
                    posterior, z, recon_image = ae(processed_image)
                    encoder_align_proj = None  # VAE doesn't have encoder alignment
                    decoder_align_proj = None
                else:
                    # MeiKai autoencoder: both encoder and decoder return alignment projections
                    z, encoder_align_proj = ae.encoder(processed_image)
                    recon_image, decoder_align_proj = ae.decoder(z)
                    # Create a dummy posterior for compatibility with loss function
                    from models.invae import DiagonalGaussianDistribution
                    posterior = DiagonalGaussianDistribution(
                        torch.cat([z, torch.zeros_like(z)], dim=1),
                        deterministic=True
                    )

                # 2). Backward pass: AE/VAE, compute the loss, backpropagate, and update the AE/VAE; Then, compute the discriminator loss and update the discriminator
                #    loss_kwargs used for SiT forward function, create here and can be reused for both AE/VAE and SiT
                loss_kwargs = dict(
                    path_type=args.path_type,
                    prediction=args.prediction,
                    weighting=args.weighting,
                )
                # Record the time_input and noises for the alignment, so that we avoid sampling again
                time_input = None
                noises = None

                # Turn off grads for the SiT model (avoid REPA gradient on the SiT model)
                requires_grad(model, False)
                # Avoid BN stats to be updated by the AE/VAE
                model.eval()

                ae_loss, ae_loss_dict = ae_loss_fn(processed_image, recon_image, posterior, global_step, "generator")
                ae_loss = ae_loss.mean()

                # Compute the REPA alignment loss for AE/VAE updates
                loss_kwargs["align_only"] = True
                
                # For MeiKai AE: 
                # 1. Main REPA loss: Align f=32 latents (8x8) through diffusion model
                # 2. Direct encoder alignment: Align encoder's f=16 features (16x16) directly with DINO
                # 3. Direct decoder alignment: Align decoder's f=16 features (16x16) directly with DINO
                #
                # For VAE: Only main REPA loss (align latents through diffusion model)
                
                # Main REPA alignment loss (latents through diffusion model)
                ae_align_outputs = model(
                    x=z,
                    y=labels,
                    zs=zs,
                    loss_kwargs=loss_kwargs,
                    time_input=time_input,
                    noises=noises,
                )
                ae_loss = ae_loss + args.vae_align_proj_coeff * ae_align_outputs["proj_loss"].mean()
                
                # Save the `time_input` and `noises` for reuse in SiT forward pass
                time_input = ae_align_outputs["time_input"]
                noises = ae_align_outputs["noises"]
                
                if not use_vae:
                    # Additional direct alignment losses for encoder and decoder features at f=16
                    # These don't go through the diffusion model, just direct cosine similarity
                    # Use zs_f16 which has the original 16x16 resolution
                    
                    # Encoder alignment: Compare encoder_align_proj (16x16) with DINO features
                    # Wrap in list to match the expected signature
                    encoder_proj_loss = compute_alignment_loss([encoder_align_proj], zs_f16)
                    ae_loss = ae_loss + args.encoder_align_proj_coeff * encoder_proj_loss
                    
                    # Decoder alignment: Compare decoder_align_proj (16x16) with DINO features
                    # Wrap in list to match the expected signature
                    decoder_proj_loss = compute_alignment_loss([decoder_align_proj], zs_f16)
                    ae_loss = ae_loss + args.decoder_align_proj_coeff * decoder_proj_loss

                accelerator.backward(ae_loss)
                if accelerator.sync_gradients:
                    grad_norm_ae = accelerator.clip_grad_norm_(ae.parameters(), args.max_grad_norm)
                optimizer_ae.step()
                optimizer_ae.zero_grad(set_to_none=True)

                # discriminator loss and update
                d_loss, d_loss_dict = ae_loss_fn(processed_image, recon_image, posterior, global_step, "discriminator")
                d_loss = d_loss.mean()
                accelerator.backward(d_loss)
                if accelerator.sync_gradients:
                    grad_norm_disc = accelerator.clip_grad_norm_(ae_loss_fn.parameters(), args.max_grad_norm)
                optimizer_loss_fn.step()
                optimizer_loss_fn.zero_grad(set_to_none=True)

                # Turn the grads back on for the SiT model, and put the model into training mode
                requires_grad(model, True)
                model.train()

                # 3). Forward pass: SiT
                # **Avoid diffusion loss to backpropagate to the VAE, so we detach the `z`**
                loss_kwargs["weighting"] = args.weighting
                loss_kwargs["align_only"] = False
                sit_outputs = model(
                    x=z.detach(),
                    y=labels,
                    zs=zs,
                    loss_kwargs=loss_kwargs,
                    time_input=time_input,
                    noises=noises,
                )

                # 4). Compute diffusion loss and REPA alignment loss, backpropagate the SiT loss, and update the model
                sit_loss = sit_outputs["denoising_loss"].mean() + args.proj_coeff * sit_outputs["proj_loss"].mean()
                accelerator.backward(sit_loss)
                if accelerator.sync_gradients:
                    grad_norm_sit = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # 5). Update SiT EMA
                if accelerator.sync_gradients:
                    unwrapped_model = accelerator.unwrap_model(model)
                    update_ema(ema, unwrapped_model._orig_mod if args.compile else unwrapped_model)

            # enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Prepare the logs based on the current step
                logs = {
                    "sit_loss": accelerator.gather(sit_loss).mean().detach().item(), 
                    "denoising_loss": accelerator.gather(sit_outputs["denoising_loss"]).mean().detach().item(),
                    "proj_loss": accelerator.gather(sit_outputs["proj_loss"]).mean().detach().item(),
                    "grad_norm_sit": accelerator.gather(grad_norm_sit).mean().detach().item(),
                    "epoch": epoch,
                    "ae_loss": accelerator.gather(ae_loss).mean().detach().item(),
                    "reconstruction_loss": accelerator.gather(ae_loss_dict["reconstruction_loss"].mean()).mean().detach().item(),
                    "perceptual_loss": accelerator.gather(ae_loss_dict["perceptual_loss"].mean()).mean().detach().item(),
                    "kl_loss": accelerator.gather(ae_loss_dict["kl_loss"].mean()).mean().detach().item(),
                    "weighted_gan_loss": accelerator.gather(ae_loss_dict["weighted_gan_loss"].mean()).mean().detach().item(),
                    "discriminator_factor": accelerator.gather(ae_loss_dict["discriminator_factor"].mean()).mean().detach().item(),
                    "gan_loss": accelerator.gather(ae_loss_dict["gan_loss"].mean()).mean().detach().item(),
                    "d_weight": accelerator.gather(ae_loss_dict["d_weight"].mean()).mean().detach().item(),
                    "grad_norm_ae": accelerator.gather(grad_norm_ae).mean().detach().item(),
                    "d_loss": accelerator.gather(d_loss).mean().detach().item(),
                    "grad_norm_disc": accelerator.gather(grad_norm_disc).mean().detach().item(),
                    "logits_real": accelerator.gather(d_loss_dict["logits_real"].mean()).mean().detach().item(),
                    "logits_fake": accelerator.gather(d_loss_dict["logits_fake"].mean()).mean().detach().item(),
                    "lecam_loss": accelerator.gather(d_loss_dict["lecam_loss"].mean()).mean().detach().item(),
                }
                
                # Add alignment-specific logs
                if not use_vae:
                    logs["ae_align_loss"] = accelerator.gather(ae_align_outputs["proj_loss"].mean()).mean().detach().item()
                    logs["encoder_align_loss"] = accelerator.gather(encoder_proj_loss).mean().detach().item()
                    logs["decoder_align_loss"] = accelerator.gather(decoder_proj_loss).mean().detach().item()
                else:
                    logs["vae_align_loss"] = accelerator.gather(ae_align_outputs["proj_loss"].mean()).mean().detach().item()
                
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # `model` and `ae` are wrapped by the `accelerator` object, so we need to unwrap them
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_ae = accelerator.unwrap_model(ae)
                    unwrapped_ae_loss_fn = accelerator.unwrap_model(ae_loss_fn)

                    # model might be compiled, we extract the original model
                    original_model = unwrapped_model._orig_mod if args.compile else unwrapped_model
                    original_ae = unwrapped_ae._orig_mod if args.compile else unwrapped_ae
                    original_discriminator = unwrapped_ae_loss_fn._orig_mod.discriminator if args.compile else unwrapped_ae_loss_fn.discriminator

                    checkpoint = {
                        "model": original_model.state_dict(),
                        "ema": ema.state_dict(),
                        "ae": original_ae.state_dict(),
                        "discriminator": original_discriminator.state_dict(),
                        "opt": optimizer.state_dict(),
                        "opt_ae": optimizer_ae.state_dict(),
                        "opt_disc": optimizer_loss_fn.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                # NOTE: Inference should use eval mode
                model.eval()
                ae.eval()
                with torch.no_grad():
                    unwrapped_model = accelerator.unwrap_model(model)
                    samples = euler_sampler(
                        unwrapped_model,
                        xT, 
                        ys,
                        num_steps=50, 
                        cfg_scale=4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    latents_stats = unwrapped_model.extract_latents_stats()
                    # reshape latents_stats to [1, C, 1, 1]
                    latents_scale = latents_stats['latents_scale'].view(1, in_channels, 1, 1)
                    latents_bias = latents_stats['latents_bias'].view(1, in_channels, 1, 1)
                    
                    # Decode using the appropriate method
                    unwrapped_ae = accelerator.unwrap_model(ae)
                    if use_vae:
                        decoded_samples = unwrapped_ae.decode(
                            denormalize_latents(samples, latents_scale, latents_bias)).sample
                    else:
                        # For MeiKai autoencoder, use decoder directly
                        decoded_samples, _ = unwrapped_ae.decoder(
                            denormalize_latents(samples, latents_scale, latents_bias))
                    
                    decoded_samples = (decoded_samples + 1) / 2.
                out_samples = accelerator.gather(decoded_samples.to(torch.float32))
                accelerator.log({"samples": wandb.Image(array2grid(out_samples))})
                logging.info("Generating EMA samples done.")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging params
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--continue-train-exp-dir", type=str, default=None)
    parser.add_argument("--wandb-history-path", type=str, default=None)

    # SiT model params
    parser.add_argument("--model", type=str, default="SiT-XL/2", choices=SiT_models.keys(),
                        help="The model to train.")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bn-momentum", type=float, default=0.1)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to compile the model for faster training")

    # dataset params
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--dataset-type", type=str, default="dir", choices=["h5", "dir"],
                        help="Dataset type: 'h5' for H5 files, 'dir' for directory structure")
    parser.add_argument("--resolution", type=int, choices=[256], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision params
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization params
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed params
    parser.add_argument("--seed", type=int, default=0)

    # cpu params
    parser.add_argument("--num-workers", type=int, default=4)

    # loss params
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"],
                        help="currently we only support v-prediction")
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, choices=["uniform", "lognormal"],
                        help="Loss weihgting, uniform or lognormal")

    # vae params
    parser.add_argument("--vae", type=str, default="f8d4", choices=["f8d4", "f16d32", "f32d256"])
    parser.add_argument("--vae-ckpt", type=str, default="pretrained/sdvae-f8d4/sdvae-f8d4.pt")

    # vae loss params
    parser.add_argument("--disc-pretrained-ckpt", type=str, default=None)
    parser.add_argument("--loss-cfg-path", type=str, default="configs/l1_lpips_kl_gan.yaml")

    # vae training params
    parser.add_argument("--vae-learning-rate", type=float, default=1e-4)
    parser.add_argument("--disc-learning-rate", type=float, default=1e-4)
    parser.add_argument("--vae-align-proj-coeff", type=float, default=1.5, 
                        help="Alignment coefficient for main REPA loss (latents through diffusion model)")
    parser.add_argument("--encoder-align-proj-coeff", type=float, default=0.5,
                        help="Direct alignment coefficient for encoder features at f=16 (MeiKai autoencoder only)")
    parser.add_argument("--decoder-align-proj-coeff", type=float, default=0.5,
                        help="Direct alignment coefficient for decoder features at f=16 (MeiKai autoencoder only)")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
