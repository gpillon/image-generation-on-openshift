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

# This import is redundant with the above WanPipeline import
# from diffusers import WanModelPipeline as WanDiffusersPipeline

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
            _log.info("WAN model already loaded, skipping initialization")
            return
        
        _log.info(f"Loading WAN model {self.model_id}")
        
        try:
            # Free up memory
            torch.cuda.empty_cache()
            gc.collect()
            
            if self.single_file_model:
                _log.info(f"Loading from single file model: {self.single_file_model}")
                # Use the correct WanPipeline import (imported directly from diffusers)
                self.model = WanPipeline.from_single_file(
                    self.single_file_model,
                    torch_dtype=torch.float16,
                )
            else:
                _log.info(f"Loading from model ID: {self.model_id}")
                self.model = WanPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                )
            
            # Move the model to the device
            _log.info(f"Moving model to device: {self.device}")
            try:
                # Move to the appropriate device
                if self.device == "cpu":
                    _log.info("Moving model to CPU")
                    self.model = self.model.to("cpu")
                elif self.device == "cuda":
                    _log.info("Moving model to CUDA")
                    self.model = self.model.to("cuda")

            except Exception as e:
                _log.error(f"Error moving model to device {self.device}: {e}, falling back to CPU")
                try:
                    self.model.to("cpu")
                except Exception as inner_e:
                    _log.error(f"Error moving model to CPU: {inner_e}")
            
            # Enable memory efficient attention if available
            if hasattr(self.model, "enable_xformers_memory_efficient_attention"):
                try:
                    _log.info("Enabling xformers memory efficient attention")
                    self.model.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    _log.warning(f"Failed to enable xformers: {e}")
            
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
            _log.info("Model not loaded, loading now")
            self.load()
        
        # Set seed for reproducibility
        generator = torch.Generator("cpu")
        seed = getattr(request, 'seed', None)
        if seed is not None:
            generator = generator.manual_seed(seed)
        
        _log.info(f"Generating video with prompt: '{request.prompt}'")
        
        # Extract parameters from the request
        num_frames = getattr(request, "num_frames", self.num_frames)
        fps = getattr(request, "fps", self.fps)
        
        # Create a wrapper for the callback function
        def callback_wrapper(_pipe, step, _timestep, callback_kwargs):
            _log.debug(f"WAN step: {step}")
            if callback_function:
                return callback_function(_pipe, step, _timestep, callback_kwargs)
            return callback_kwargs
        
        try:
            _log.info("Starting WAN model inference")
            # Call the WanPipeline using only parameters that it supports
            output = self.model(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                height=request.height,
                width=request.width,
                num_frames=num_frames,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                generator=generator,
                callback_on_step_end=callback_wrapper  # Use wrapper to handle callback safely
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
            
        except Exception as e:
            _log.error(f"Error generating video: {e}")
            import traceback
            _log.error(traceback.format_exc())
            raise 