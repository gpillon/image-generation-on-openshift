import logging
from typing import Dict

import torch
from diffusers import (StableDiffusionXLImg2ImgPipeline,
                       StableDiffusionXLPipeline)

import sys
import os
import gc
# Add parent directory to path to import common modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.classes import GenerationRequest

_log = logging.getLogger(__name__)


class DiffusersPipeline:
    def __init__(self, args):
        self.model_id: str = args.model_id or "/mnt/models"
        self.single_file_model: str = args.single_file_model or None
        self.use_refiner: bool = args.use_refiner or False
        self.refiner_id: str = args.refiner_id or None
        self.refiner_single_file_model: str = args.refiner_single_file_model or None
        self.device: str = args.device or "cuda"
        self.pipeline = None
        self.refiner = None
        self.ready = False

    def load(self):
        try:
            _log.info(f"Loading model with settings: model_id={self.model_id}, single_file_model={self.single_file_model}, device={self.device}")
            
            # First clear any existing pipeline and free memory
            if self.pipeline is not None:
                del self.pipeline
                if self.refiner is not None:
                    del self.refiner
                torch.cuda.empty_cache()
                gc.collect()
            
            # Setup a different pipeline object depending on model loading method
            if self.single_file_model and self.single_file_model != "":
                _log.info(f"Loading from single file: {self.single_file_model}")
                model_path = self.model_id
                if self.single_file_model.startswith("/"):
                    model_path = self.single_file_model
                else:
                    model_path = f"{self.model_id}/{self.single_file_model}"
                _log.info(f"Full model path: {model_path}")
                
                pipeline = StableDiffusionXLPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    safety_checker=None,
                    use_safetensors=True,
                )
                _log.info("Pipeline initialized from single file")
                
                # Setup refiner if needed
                if self.use_refiner and self.refiner_single_file_model:
                    refiner_path = self.refiner_id or self.model_id
                    if self.refiner_single_file_model.startswith("/"):
                        refiner_path = self.refiner_single_file_model
                    else:
                        refiner_path = f"{refiner_path}/{self.refiner_single_file_model}"
                    
                    _log.info(f"Loading refiner from single file: {refiner_path}")
                    refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
                        refiner_path,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        safety_checker=None,
                        use_safetensors=True,
                    )
                    _log.info("Refiner initialized from single file")
                    self.refiner = refiner
            else:
                _log.info(f"Loading from pretrained: {self.model_id}")
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    safety_checker=None,
                    use_safetensors=True,
                )
                _log.info("Pipeline initialized from pretrained")
                
                # Setup refiner if needed
                if self.use_refiner and self.refiner_id:
                    _log.info(f"Loading refiner from pretrained: {self.refiner_id}")
                    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        self.refiner_id,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        safety_checker=None,
                        use_safetensors=True,
                    )
                    _log.info("Refiner initialized from pretrained")
                    self.refiner = refiner
            
            # Setup device options
            try:
                if self.device == "cuda":
                    _log.info("Using CUDA: Standard mode")
                    if torch.cuda.is_available():
                        pipeline = pipeline.to("cuda")
                        if self.refiner:
                            self.refiner = self.refiner.to("cuda")
                        # Try to enable optimizations
                        try:
                            pipeline.enable_xformers_memory_efficient_attention()
                            if self.refiner:
                                self.refiner.enable_xformers_memory_efficient_attention()
                            _log.info("Enabled xformers memory efficient attention")
                        except Exception as e:
                            _log.warning(f"Could not enable xformers: {e}")
                    else:
                        _log.warning("CUDA requested but not available, falling back to CPU")
                        pipeline = pipeline.to("cpu")
                        if self.refiner:
                            self.refiner = self.refiner.to("cpu")
                elif self.device == "enable_model_cpu_offload":
                    _log.info("Using CUDA: Model CPU offload mode")
                    pipeline.enable_model_cpu_offload()
                    if self.refiner:
                        self.refiner.enable_model_cpu_offload()
                elif self.device == "enable_sequential_cpu_offload":
                    _log.info("Using CUDA: Sequential CPU offload mode")
                    pipeline.enable_sequential_cpu_offload()
                    if self.refiner:
                        self.refiner.enable_sequential_cpu_offload()
                elif self.device == "cpu":
                    _log.info("Using CPU: Warning - generation will be very slow")
                    pipeline = pipeline.to("cpu")
                    if self.refiner:
                        self.refiner = self.refiner.to("cpu")
            except Exception as e:
                _log.error(f"Error setting up device: {e}")
                _log.info("Falling back to CPU")
                pipeline = pipeline.to("cpu")
                if self.refiner:
                    self.refiner = self.refiner.to("cpu")
            
            # Set the pipeline
            self.pipeline = pipeline
            self.ready = True
            _log.info("Model loading completed successfully")
            
        except Exception as e:
            _log.error(f"Error loading model: {e}")
            import traceback
            _log.error(traceback.format_exc())
            raise
    
    def predict(self, payload: GenerationRequest, callback_func_base: callable, callback_func_refiner: callable = None) -> None:
        if not self.ready:
            self.load()
        
        # Extract parameters from the request
        prompt = payload.prompt
        height = getattr(payload, 'height', None) or 1024
        width = getattr(payload, 'width', None) or 1024
        guidance_scale = getattr(payload, 'guidance_scale', 8.0)
        num_inference_steps = getattr(payload, 'num_inference_steps', 50)
        
        # Set up a fixed seed if requested
        seed = getattr(payload, 'seed', None)
        generator = None
        if seed is not None:
            try:
                generator = torch.Generator(device=self.device)
                generator = generator.manual_seed(seed)
            except Exception as e:
                _log.warning(f"Error setting seed on device {self.device}: {e}, using CPU generator")
                generator = torch.Generator()
                generator = generator.manual_seed(seed)
        
        # Handle additional parameters
        negative_prompt = getattr(payload, 'negative_prompt', None)
        prompt_2 = getattr(payload, 'prompt_2', None)
        negative_prompt_2 = getattr(payload, 'negative_prompt_2', None)
        
        # Refiner settings
        denoising_limit = getattr(payload, 'denoising_limit', 0.8)
        
        # Original and target sizes for high-res fix
        crops_coords_top_left = getattr(payload, 'crops_coords_top_left', (0, 0))
        original_size = getattr(payload, 'original_size', None)
        target_size = getattr(payload, 'target_size', None)
        
        # Log the parameters
        _log.info(f"Generating image with parameters: prompt='{prompt}', height={height}, width={width}")
        if negative_prompt:
            _log.info(f"Using negative prompt: {negative_prompt}")
        
        # Process the image
        try:
            # Base model generation
            _log.info("Starting base model inference")
            image = self.pipeline(
                prompt=prompt,
                prompt_2=prompt_2,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type="latent" if self.use_refiner else "pil",
                original_size=original_size,
                target_size=target_size,
                crops_coords_top_left=crops_coords_top_left,
                generator=generator,
                callback_on_step_end=callback_func_base
            ).images[0]
            
            # Refiner model generation
            if self.use_refiner and self.refiner:
                _log.info("Starting refiner model inference")
                image = self.refiner(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    negative_prompt=negative_prompt,
                    negative_prompt_2=negative_prompt_2,
                    image=image,  # Pass the latent output from base model
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    denoising_start=denoising_limit,  # Start from denoising_limit (default: 0.8)
                    generator=generator,
                    callback_on_step_end=callback_func_refiner
                ).images[0]
            
            _log.info("Image generation completed successfully")
            return image
            
        except Exception as e:
            _log.error(f"Error during inference: {e}")
            import traceback
            _log.error(traceback.format_exc())
            raise 