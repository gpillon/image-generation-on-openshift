import gc
import logging
import io
import base64
from typing import Dict
from PIL import Image

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

from classes import GenerationRequest

_log = logging.getLogger(__name__)

class WanModelPipeline:
    def __init__(self, args):
        self.model_id: str = args.model_id or "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        self.device = args.device or "cuda"
        self.single_file_model: str = args.single_file_model or None
        
        self.pipeline = None
        self.ready = False
        self.fps = 15  # Default frames per second for video

    def load(self):
        _log.info(f"Loading WAN model with settings: model_id={self.model_id}, device={self.device}")
        try:
            # Free up memory
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

            if self.single_file_model and self.single_file_model != "":
                _log.info(f"Loading from single file: {self.single_file_model}")
                _log.warning("Single file model not yet supported for WAN, using pretrained model instead")
                # Fall back to pretrained model
                
            # Load the VAE and pipeline
            _log.info(f"Loading VAE from: {self.model_id}")
            vae = AutoencoderKLWan.from_pretrained(
                self.model_id, 
                subfolder="vae", 
                torch_dtype=torch.float32
            )
            
            _log.info(f"Loading WAN pipeline from: {self.model_id}")
            pipeline = WanPipeline.from_pretrained(
                self.model_id,
                vae=vae,
                torch_dtype=torch.bfloat16
            )
            
            # Move to the appropriate device
            if self.device == "cpu":
                _log.info("Moving model to CPU")
                pipeline = pipeline.to("cpu")
            else:
                _log.info("Moving model to CUDA")
                pipeline = pipeline.to("cuda")
            
            self.pipeline = pipeline
            self.ready = True
            _log.info("WAN model loaded successfully")
            
        except Exception as e:
            _log.error(f"Error loading WAN model: {e}")
            import traceback
            _log.error(traceback.format_exc())
            raise

    def predict(self, payload: GenerationRequest, callback_func_base: callable, callback_func_refiner: callable = None) -> None:
        # Extract parameters from the request
        prompt = payload.prompt
        negative_prompt = getattr(payload, 'negative_prompt', None)
        height = getattr(payload, 'height', 480)  # Default height for video
        width = getattr(payload, 'width', 832)    # Default width for video
        num_frames = getattr(payload, 'num_frames', 81)  # Default number of frames
        guidance_scale = getattr(payload, 'guidance_scale', 5.0)
        
        # Set up a fixed seed if requested
        seed = getattr(payload, 'seed', None)
        generator = torch.Generator("cpu")
        if seed is not None:
            generator = generator.manual_seed(seed)
        
        # Log the parameters
        _log.info(f"Generating video with WAN: prompt='{prompt}', height={height}, width={width}, frames={num_frames}")
        
        # Create a custom callback to wrap the provided one
        def video_callback_wrapper(_pipe, step, _timestep, callback_kwargs):
            # Extract latents or intermediate results if available
            # Note: This may need adjustment based on WanPipeline's callback structure
            if "latents" in callback_kwargs:
                latents = callback_kwargs["latents"]
                _log.info(f"WAN latents shape at step {step}: {latents.shape}, dtype: {latents.dtype}")
            
            # Call the original callback
            return callback_func_base(_pipe, step, _timestep, callback_kwargs)
        
        # Generate the video
        try:
            _log.info("Starting WAN pipeline inference")
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=video_callback_wrapper if callback_func_base else None
            )
            
            _log.info("WAN pipeline inference completed successfully")
            
            # Export video to bytes for preview
            video_frames = result.frames[0]
            
            # Save video to a temporary file
            temp_video_path = "temp_output.mp4"
            export_to_video(video_frames, temp_video_path, fps=self.fps)
            
            # For the preview, use the first frame as an image
            first_frame = Image.fromarray(video_frames[0])
            
            # Return the first frame for preview
            return first_frame
            
        except Exception as e:
            _log.error(f"Error during WAN inference: {e}")
            import traceback
            _log.error(traceback.format_exc())
            raise 