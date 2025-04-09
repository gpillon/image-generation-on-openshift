import logging
from typing import Dict

import torch
from diffusers import (StableDiffusionXLImg2ImgPipeline,
                       StableDiffusionXLPipeline)

from classes import GenerationRequest

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
            
            # Load the model
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
                
            if self.device:
                _log.info(f"Moving model to device: {self.device}")
                if self.device == "cuda":
                    try:
                        _log.info("Checking CUDA availability")
                        if torch.cuda.is_available():
                            _log.info(f"CUDA is available. Device count: {torch.cuda.device_count()}")
                            _log.info(f"Current device: {torch.cuda.current_device()}")
                            _log.info(f"Device name: {torch.cuda.get_device_name(0)}")
                        else:
                            _log.error("CUDA is not available!")
                            raise RuntimeError("CUDA is not available on this system")
                        
                        pipeline.to(torch.device("cuda"))
                        _log.info("Model moved to CUDA")
                        try:
                            pipeline.enable_xformers_memory_efficient_attention()
                            _log.info("xformers memory efficient attention enabled")
                        except Exception as e:
                            _log.warning(f"Could not enable xformers: {e}")
                    except Exception as e:
                        _log.error(f"Error setting up CUDA: {e}")
                        _log.info("Falling back to CPU")
                        pipeline.to(torch.device("cpu"))
                elif self.device == "cpu":
                    pipeline.to(torch.device("cpu"))
                    _log.info("Model moved to CPU")
                elif self.device == "enable_model_cpu_offload":
                    pipeline.enable_model_cpu_offload()
                    _log.info("Model CPU offload enabled")
                elif self.device == "enable_sequential_cpu_offload":
                    pipeline.enable_sequential_cpu_offload()
                    _log.info("Sequential CPU offload enabled")
                else:
                    raise ValueError(f"Invalid device: {self.device}")
            else:
                try:
                    pipeline.to(torch.device("cuda"))
                    pipeline.enable_xformers_memory_efficient_attention()
                    _log.info("Model moved to CUDA (default)")
                except Exception as e:
                    _log.error(f"Failed to move to CUDA: {e}")
                    _log.info("Falling back to CPU")
                    pipeline.to(torch.device("cpu"))
            
            self.pipeline = pipeline
            _log.info("Base model loaded successfully")

            # Load the refiner model
            if self.use_refiner:
                _log.info("Loading refiner model")
                if self.refiner_single_file_model and self.refiner_single_file_model != "":
                    refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(
                        self.refiner_single_file_model,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        safety_checker=None,
                        use_safetensors=True,
                        text_encoder_2=pipeline.text_encoder_2,
                        vae=pipeline.vae,
                    )
                else:
                    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        self.refiner_id,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        safety_checker=None,
                        use_safetensors=True,
                    )
                if self.device:
                    print(f"Loading refiner model on device: {self.device}")
                    if self.device == "cuda":
                        refiner.to(torch.device("cuda"))
                        refiner.enable_xformers_memory_efficient_attention()
                    elif self.device == "cpu":
                        refiner.to(torch.device("cpu"))
                    elif self.device == "enable_model_cpu_offload":
                        refiner.enable_model_cpu_offload()
                    elif self.device == "enable_sequential_cpu_offload":
                        refiner.enable_sequential_cpu_offload()
                    else:
                        raise ValueError(f"Invalid device: {self.device}")
                else:
                    refiner.to(torch.device("cuda"))
                    refiner.enable_xformers_memory_efficient_attention()
                self.refiner = refiner

            # The ready flag is used by model ready endpoint for readiness probes,
            # set to True when model is loaded successfully without exceptions.
            self.ready = True
            _log.info("Model loading complete, ready flag set to True")
            
        except Exception as e:
            _log.error(f"Error loading model: {e}")
            import traceback
            _log.error(traceback.format_exc())
            raise

    def convert_lists_to_tuples(self, data):
        if isinstance(data, dict):
            return {k: self.convert_lists_to_tuples(v) for k, v in data.items()}
        elif isinstance(data, list):
            return tuple(self.convert_lists_to_tuples(v) for v in data)
        else:
            return data
        

    def predict(self, payload: GenerationRequest, callback_func_base: callable, callback_func_refiner: callable) -> None:
        payload_dict = self.convert_lists_to_tuples(payload.__dict__)
        _log.info(f"Received request: {payload_dict}")

        # Create the image, without refiner if not needed
        if not self.use_refiner:
            image = self.pipeline(
                **payload_dict, callback_on_step_end=callback_func_base
            ).images[0]
        else:
            denoising_limit = payload_dict.get("denoising_limit", 0.8)
            image = self.pipeline(
                **payload_dict,
                output_type="latent",
                denoising_end=denoising_limit,
                callback_on_step_end=callback_func_base,
            ).images
            image = self.refiner(
                **payload_dict,
                image=image,
                denoising_start=denoising_limit,
                callback_on_step_end=callback_func_refiner,
            ).images[0]

        return image
