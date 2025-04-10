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
                
            # Load the pipeline first (without specifying VAE)
            _log.info(f"Loading WAN pipeline from: {self.model_id}")
            pipeline = WanPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16  # Use float16 instead of bfloat16 for better compatibility
            )
            
            # Now load VAE with matching dtype
            _log.info(f"Loading VAE from: {self.model_id}")
            vae = AutoencoderKLWan.from_pretrained(
                self.model_id, 
                subfolder="vae", 
                torch_dtype=torch.float16  # Match the model's dtype
            )
            
            # Assign the VAE to the pipeline
            pipeline.vae = vae
            
            
            # Move to the appropriate device
            if self.device == "cpu":
                _log.info("Moving model to CPU")
                pipeline = pipeline.to("cpu")
            else:
                _log.info("Moving model to CUDA")
                # pipeline = pipeline.to("cuda")

            pipeline.enable_model_cpu_offload()
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
        self.fps = getattr(payload, 'fps', 15)  # Get fps from payload
        num_inference_steps = getattr(payload, 'num_inference_steps', 50)
        # Set up a fixed seed if requested
        seed = getattr(payload, 'seed', None)
        generator = torch.Generator("cpu")
        if seed is not None:
            generator = generator.manual_seed(seed)
        
        # Log the parameters
        _log.info(f"Generating video with WAN: prompt='{prompt}', height={height}, width={width}, frames={num_frames}, fps={self.fps}")
        
        # Create a custom callback to wrap the provided one
        def video_callback_wrapper(_pipe, step, _timestep, callback_kwargs):
            # Extract latents or intermediate results if available
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
                num_inference_steps=num_inference_steps,
                generator=generator,
                callback_on_step_end=video_callback_wrapper if callback_func_base else None
            )
            
            _log.info("WAN pipeline inference completed successfully")
            
            # Export video to bytes for preview - do this in a separate try block
            # to ensure video is saved even if frame processing fails
            video_frames = result.frames[0]
            
            # Save video to a temporary file immediately
            temp_video_path = "/tmp/temp_output.mp4"
            try:
                export_to_video(video_frames, temp_video_path, fps=self.fps)
                _log.info(f"Video saved to {temp_video_path} with {len(video_frames)} frames at {self.fps} fps")
            except Exception as video_save_error:
                _log.error(f"Error saving video: {video_save_error}")
                raise video_save_error
            
            # Now try to create a preview image from the first frame
            try:
                import numpy as np
                
                # Get the first frame
                first_frame = video_frames[0]
                
                # Check and fix the shape if needed
                if first_frame.shape[0] == 1 and first_frame.shape[1] == 1:
                    # Create a placeholder image since the frame is too small
                    placeholder = np.zeros((480, 480, 3), dtype=np.uint8)
                    placeholder[:,:] = (100, 150, 200)  # Blue-ish background
                    # Add text with PIL
                    preview_img = Image.fromarray(placeholder)
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(preview_img)
                    draw.text((20, 20), "Video generation complete!", (255, 255, 255))
                    draw.text((20, 50), f"Video saved to: {temp_video_path}", (255, 255, 255))
                    draw.text((20, 80), f"Frames: {len(video_frames)}, FPS: {self.fps}", (255, 255, 255))
                else:
                    # Make sure the frame is in the right format (0-255 uint8)
                    if first_frame.dtype == np.float32 or first_frame.dtype == np.float64:
                        # Convert float (0-1) to uint8 (0-255)
                        if first_frame.max() <= 1.0:
                            first_frame = (first_frame * 255).astype(np.uint8)
                        else:
                            first_frame = first_frame.astype(np.uint8)
                    
                    preview_img = Image.fromarray(first_frame)
                
                return preview_img
                
            except Exception as e:
                _log.error(f"Error creating preview image: {e}")
                # Create a fallback preview image
                placeholder = Image.new('RGB', (480, 480), color=(100, 150, 200))
                from PIL import ImageDraw
                draw = ImageDraw.Draw(placeholder)
                draw.text((20, 20), "Video generation complete!", fill=(255, 255, 255))
                draw.text((20, 50), f"Video saved, but preview failed: {str(e)[:50]}", fill=(255, 255, 255))
                
                return placeholder
            
        except Exception as e:
            _log.error(f"Error during WAN inference: {e}")
            import traceback
            _log.error(traceback.format_exc())
            raise 