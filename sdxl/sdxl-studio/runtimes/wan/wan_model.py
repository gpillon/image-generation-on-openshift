import gc
import logging
import io
import base64
from typing import Dict
from PIL import Image

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

import sys
import os
# Add parent directory to path to import common modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.classes import GenerationRequest

import random
import numpy as np
import tempfile
import imageio

from diffusers import WanModelPipeline as WanDiffusersPipeline

_log = logging.getLogger(__name__)

class WanModelPipeline:
    def __init__(self, args):
        self.model_id = args.model_id
        self.single_file_model = args.single_file_model
        self.device = args.device
        self.fps = args.fps
        self.num_frames = args.num_frames
        self.model = None
        self.is_loaded = False
    
    def load(self):
        """Load the model if it's not already loaded"""
        if self.is_loaded:
            return
        
        _log.info(f"Loading WAN model {self.model_id}")
        
        try:
            if self.single_file_model:
                _log.info(f"Loading from single file model: {self.single_file_model}")
                self.model = WanDiffusersPipeline.from_single_file(
                    self.single_file_model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
            else:
                _log.info(f"Loading from model ID: {self.model_id}")
                self.model = WanDiffusersPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
            
            self.model.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.model, "enable_xformers_memory_efficient_attention"):
                self.model.enable_xformers_memory_efficient_attention()
            
            self.is_loaded = True
            _log.info("WAN model loaded successfully")
        
        except Exception as e:
            _log.error(f"Error loading WAN model: {e}")
            import traceback
            _log.error(traceback.format_exc())
            raise
    
    def predict(self, request: GenerationRequest, callback_function=None):
        """
        Generate a video using the WAN model based on the request parameters.
        Returns the path to the video file.
        """
        if not self.is_loaded:
            self.load()
        
        # Set seed for reproducibility
        if request.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(request.seed)
        else:
            seed = random.randint(0, 2**32 - 1)
            _log.info(f"Using random seed: {seed}")
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        _log.info(f"Generating video with prompt: '{request.prompt}'")
        
        # Extract parameters from the request
        num_frames = getattr(request, "num_frames", self.num_frames)
        fps = getattr(request, "fps", self.fps)
        
        # Create a callback object if a callback function is provided
        callback = None
        if callback_function:
            from diffusers.utils import is_accelerate_available
            if is_accelerate_available():
                from diffusers.utils import WanModelOutputCallback
                callback = WanModelOutputCallback(callback_function)
        
        # Generate the video frames
        output = self.model(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            output_type="pt",  # Return pytorch tensors
            num_frames=num_frames,
            generator=generator,
            callback=callback,
            callback_steps=1
        )
        
        # Convert the pytorch tensor video frames to a video file
        frames = output.frames
        
        # Create a temporary file to save the video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            video_path = temp_file.name
        
        # Convert frames to images and save as video
        frame_pil_images = []
        for frame in frames:
            # Convert to PIL Image
            frame = frame.cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            frame = frame.transpose(1, 2, 0)  # CHW -> HWC
            pil_image = Image.fromarray(frame)
            frame_pil_images.append(pil_image)
        
        _log.info(f"Saving video with {len(frame_pil_images)} frames to {video_path}")
        
        # Save as mp4 video using imageio
        imageio.mimsave(video_path, frame_pil_images, fps=fps)
        
        _log.info(f"Video saved to {video_path}")
        
        # Return the path to the video file
        return video_path 