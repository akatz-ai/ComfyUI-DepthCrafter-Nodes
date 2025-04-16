import os
import torch
import math
import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

from .depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from .depthcrafter.depth_crafter_ppl import DepthCrafterPipeline

class DepthCrafterNode:
    def __init__(self):
        self.progress_bar = None

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)

    def update_progress(self, *args, **kwargs):
        if self.progress_bar:
            self.progress_bar.update(1)

    def end_progress(self):
        self.progress_bar = None
        
    CATEGORY = "DepthCrafter"


class DownloadAndLoadDepthCrafterModel(DepthCrafterNode):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "enable_model_cpu_offload": ("BOOLEAN", {"default": True}),
            "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("DEPTHCRAFTER_MODEL",)
    RETURN_NAMES = ("depthcrafter_model",)
    FUNCTION = "load_model"
    DESCRIPTION = """
    Downloads and loads the DepthCrafter model.
    - enable_model_cpu_offload: If True, the model will be offloaded to the CPU. (Saves VRAM)
    - enable_sequential_cpu_offload: If True, the model will be offloaded to the CPU in a sequential manner. (Saves the most VRAM but runs slowly)
    Only enable one of the two at a time.
    """

    def load_model(self, enable_model_cpu_offload, enable_sequential_cpu_offload):
        device = mm.get_torch_device()

        model_dir = os.path.join(folder_paths.models_dir, "depthcrafter")
        os.makedirs(model_dir, exist_ok=True)

        # Paths to models
        unet_path = os.path.join(model_dir, "tencent_DepthCrafter")
        pretrain_path = os.path.join(model_dir, "stabilityai_stable-video-diffusion-img2vid-xt")

        depthcrafter_files_to_download = [
            "config.json",
            "diffusion_pytorch_model.safetensors",
        ]
        svd_files_to_download = [
            "feature_extractor/preprocessor_config.json",
            "image_encoder/config.json",
            "image_encoder/model.fp16.safetensors",
            "scheduler/scheduler_config.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "model_index.json",
        ]

        self.start_progress(len(svd_files_to_download) + len(depthcrafter_files_to_download))

        # Check if models exist, if not download them
        from huggingface_hub import hf_hub_download

        if not os.path.exists(unet_path):
            print(f"Downloading UNet model to: {unet_path}")
            for path in depthcrafter_files_to_download:
                hf_hub_download(
                    repo_id="tencent/DepthCrafter",
                    filename=path,
                    local_dir=unet_path,
                    local_dir_use_symlinks=False,
                    revision="c1a22b53f8abf80cd0b025adf29e637773229eca",
                )
                self.update_progress()

        if not os.path.exists(pretrain_path):
            print(f"Downloading pre-trained pipeline to: {pretrain_path}")
            for path in svd_files_to_download:
                hf_hub_download(
                    repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
                    filename=path,
                    local_dir=pretrain_path,
                    local_dir_use_symlinks=False,
                    revision="9e43909513c6714f1bc78bcb44d96e733cd242aa",
                )
                self.update_progress()

        # Load the custom UNet model
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        # Load the pipeline
        pipe = DepthCrafterPipeline.from_pretrained(
            pretrain_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_local_files_only=True,
            low_cpu_mem_usage=True,
        )

        # Model setup
        # try:
        #     pipe.enable_xformers_memory_efficient_attention()
        # except Exception as e:
        #     print(e)
        #     print("Xformers is not enabled")
        pipe.enable_attention_slicing()
        
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        elif enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)


        depthcrafter_model = {
            "pipe": pipe,
            "device": device,
        }

        self.end_progress()

        return (depthcrafter_model,)

class DepthCrafter(DepthCrafterNode):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "depthcrafter_model": ("DEPTHCRAFTER_MODEL", ),
            "images": ("IMAGE", ),
            "force_size": ("BOOLEAN", {"default": True}),
            "num_inference_steps": ("INT", {"default": 5, "min": 1, "max": 100}),
            "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            "window_size": ("INT", {"default": 110, "min": 1, "max": 200}),
            "overlap": ("INT", {"default": 25, "min": 0, "max": 100}),
        }}
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_maps",)
    FUNCTION = "process"
    DESCRIPTION = """
    Runs the DepthCrafter model on the input images.
    **WARNING:** The model internally requires image dimensions (width and height)
    to be multiples of 64. Enable 'force_size' to automatically resize the input
    to the nearest valid dimensions, or ensure your input images already meet
    this requirement if 'force_size' is disabled.
    """
    
    def process(self, depthcrafter_model, images, force_size, num_inference_steps, guidance_scale, window_size, overlap):
        device = depthcrafter_model['device']
        pipe = depthcrafter_model['pipe']
        
        B, H, W, C = images.shape
        orig_H, orig_W = H, W

        if force_size:
            # Round to nearest multiple of 64
            width = round(W / 64) * 64
            height = round(H / 64) * 64
            # Ensure minimum size is 64
            width = max(64, width)
            height = max(64, height)

            if width != W or height != H:
                print(f"DepthCrafter: Resizing input from {W}x{H} to {width}x{height} (multiples of 64)")
                # Permute for interpolation: B, H, W, C -> B, C, H, W
                images_for_resize = images.permute(0, 3, 1, 2)
                images_resized = torch.nn.functional.interpolate(
                    images_for_resize,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                )
                # Permute back: B, C, H, W -> B, H, W, C
                images = images_resized.permute(0, 2, 3, 1)
                # Update H, W to the new dimensions
                H, W = height, width
            else:
                # Dimensions are already multiples of 64 or rounding didn't change them
                width = W
                height = H
        else:
            # Check if dimensions are multiples of 64
            if W % 64 != 0 or H % 64 != 0:
                raise ValueError(
                    f"Input image dimensions ({W}x{H}) are not multiples of 64. "
                    f"Please resize your image to a multiple of 64 (e.g., {round(W / 64) * 64}x{round(H / 64) * 64}) "
                    f"or enable the 'force_size' option."
                )
            # Use original dimensions if they are valid
            width = W
            height = H

        # Permute images to [t, c, h, w] for the pipeline
        images = images.permute(0, 3, 1, 2)  # [B, C, H, W]
        images = images.to(device=device, dtype=torch.float16)
        images = torch.clamp(images, 0, 1)
        
        # Calculate total num of steps
        num_windows = math.ceil((B - window_size) / (window_size - overlap)) + 1
        self.start_progress(num_inference_steps * num_windows)
        
        # Run the pipeline
        with torch.inference_mode():
            result = pipe(
                images,
                height=height,
                width=width,
                output_type="pt",
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=False,
                progress_callback=self.update_progress,
            )
            
        res = result.frames[0]  # [B, H, W, C]
        
        # Convert to grayscale depth map
        res = res.sum(dim=1) / res.shape[1]  # [B, H, W]
        
        # Normalize depth maps
        res_min = res.min()
        res_max = res.max()
        res = (res - res_min) / (res_max - res_min + 1e-8)
        
        # Convert back to tensor with 3 channels
        depth_maps = res.unsqueeze(-1).repeat(1, 1, 1, 3)  # [B, H, W, 3]
        depth_maps = depth_maps.float()
        
        self.end_progress()
        
        return (depth_maps,)
