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
        # Load the model
        if self.single_file_model and self.single_file_model != "":
            pipeline = StableDiffusionXLPipeline.from_single_file(
                self.single_file_model,
                torch_dtype=torch.float16,
                variant="fp16",
                safety_checker=None,
                use_safetensors=True,
            )
        else:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                safety_checker=None,
                use_safetensors=True,
            )
        if self.device:
            print(f"Loading model on device: {self.device}")
            if self.device == "cuda":
                pipeline.to(torch.device("cuda"))
                pipeline.enable_xformers_memory_efficient_attention()
            elif self.device == "cpu":
                pipeline.to(torch.device("cpu"))
            elif self.device == "enable_model_cpu_offload":
                pipeline.enable_model_cpu_offload()
            elif self.device == "enable_sequential_cpu_offload":
                pipeline.enable_sequential_cpu_offload()
            else:
                raise ValueError(f"Invalid device: {self.device}")
        else:
            pipeline.to(torch.device("cuda"))
            pipeline.enable_xformers_memory_efficient_attention()
        self.pipeline = pipeline

        # Load the refiner model
        if self.use_refiner:
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
