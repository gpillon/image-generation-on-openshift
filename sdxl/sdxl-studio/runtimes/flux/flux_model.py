import gc
import logging
from typing import Dict

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel, CLIPTextModel
from huggingface_hub import hf_hub_download, login
from safetensors.torch import load_file

import sys
import os
# Add parent directory to path to import common modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.classes import GenerationRequest

_log = logging.getLogger(__name__)

class FluxModelPipeline:
    def __init__(self, args):
        #self.repo_id: str = args.repo_id or "black-forest-labs/FLUX.1-schnell"
        self.repo_id: str = "black-forest-labs/FLUX.1-schnell" #TODO: this is so bad... :( need to solve this...
        self.model_id: str = args.model_id or "/mnt/models"
        self.device = args.device or "cuda"
        # self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.single_file_model: str = args.single_file_model or None
        # # Encoders paths
        # self.clip_l_path = None
        # self.t5_fp8_path = None
        
        self.pipeline = None
        self.ready = False

    def load(self):
        _log.info(f"Loading Flux model with settings: model_id={self.model_id}, device={self.device}")
        try:
            # Free up memory
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

            if self.single_file_model and self.single_file_model != "":
                print ("WARNING: Single file model not yet supported & optimized for Flux, SHOULD NOT BE USED!")

                _log.info(f"Loading from single file: {self.single_file_model}")
                model_path = self.model_id
                if self.single_file_model.startswith("/"):
                    model_path = self.single_file_model
                else:
                    model_path = f"{self.model_id}/{self.single_file_model}"
                _log.info(f"Full model path: {model_path}")

                # pipeline = FluxPipeline.from_single_file NOT SUPPORTED! 
                # https://github.com/huggingface/diffusers/issues/9053

                pipeline = FluxPipeline.from_pretrained(
                    self.repo_id,
                    transformer=None,
                    #text_encoder_2=None,
                    torch_dtype=torch.float16,
                    device_map="balanced"  # Only valid option for Flux in diffusers
                )

                transformer = FluxTransformer2DModel.from_single_file(model_path)
                #text_encoder_2 = T5EncoderModel.from_pretrained(self.repo_id, subfolder="text_encoder_2", torch_dtype=torch.float16)

                pipeline.transformer = transformer
                #pipeline.text_encoder_2 = text_encoder_2
                _log.info("Pipeline initialized from single file")
            else:
                _log.info(f"Loading from pretrained: {self.model_id}")
                pipeline = FluxPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16,
                    device_map="balanced"  # Only valid option for Flux in diffusers
                )
            
                _log.info("Pipeline initialized from pretrained")

            # Setup optimization
            _log.info("Setting up VAE optimizations")
            pipeline.vae.enable_slicing()
            pipeline.vae.enable_tiling()
            
            # Set device if needed (should be handled by device_map)
            if self.device == "cpu":
                _log.info("Moving model to CPU")
                pipeline.to(torch.device("cpu"))
            # elif self.device == "enable_model_cpu_offload":     # Seems not working with Flux with device_map="balanced" .. but if not "balanced crashes on my pc.. :("  
            #     _log.info("Enabling model CPU offload")
            #     pipeline.enable_model_cpu_offload()
            # elif self.device == "enable_sequential_cpu_offload": # Seems not working with Flux  with device_map="balanced"   but if not "balanced crashes on my pc.. :("  
            #     _log.info("Enabling sequential CPU offload")
            #     pipeline.enable_sequential_cpu_offload()
            
            self.pipeline = pipeline
            self.ready = True
            _log.info("Flux model loaded successfully")
            
        except Exception as e:
            _log.error(f"Error loading Flux model: {e}")
            import traceback
            _log.error(traceback.format_exc())
            raise

    def convert_lists_to_tuples(self, data):
        if isinstance(data, dict):
            return {k: self.convert_lists_to_tuples(v) for k, v in data.items()}
        elif isinstance(data, list):
            return tuple(self.convert_lists_to_tuples(item) for item in data)
        else:
            return data
            
    def predict(self, payload: GenerationRequest, callback_func_base: callable = None) -> None:
        if not self.ready:
            self.load()
        
        # Extract parameters from the request
        prompt = payload.prompt
        height = getattr(payload, 'height', None) or 1024
        width = getattr(payload, 'width', None) or 1024
        guidance_scale = getattr(payload, 'guidance_scale', 5.0)
        num_inference_steps = getattr(payload, 'num_inference_steps', 50)
        
        # Set up a fixed seed if requested
        seed = getattr(payload, 'seed', None)
        generator = torch.Generator("cuda")
        if seed is not None:
            generator = generator.manual_seed(seed)
        
        # Define a debug callback wrapper
        def debug_callback_wrapper(_pipe, step, _timestep, callback_kwargs):
            _log.debug(f"Flux step: {step}")
            if callback_func_base:
                return callback_func_base(_pipe, step, _timestep, callback_kwargs)
            return callback_kwargs
        
        # Flux doesn't support negative prompts in the same way as SDXL, log this limitation
        negative_prompt = getattr(payload, 'negative_prompt', None)
        if negative_prompt:
            _log.warning(f"Flux doesn't support negative prompts in the same way as SDXL. Ignoring negative prompt: {negative_prompt}")
        
        # Log the parameters
        _log.info(f"Generating image with Flux: prompt='{prompt}', height={height}, width={width}")
        
        # Create the image
        try:
            _log.info("Starting Flux pipeline inference")
            result = self.pipeline(
                prompt=prompt,
                #negative_prompt=negative_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                # num_inference_steps=4,
                generator=generator,
                callback_on_step_end=debug_callback_wrapper if callback_func_base else None
            )
            _log.info("Flux pipeline inference completed successfully")
            return result.images[0]
        except Exception as e:
            _log.error(f"Error during Flux inference: {e}")
            import traceback
            _log.error(traceback.format_exc())
            raise 