import gc
import logging
from typing import Dict

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel, CLIPTextModel
from huggingface_hub import hf_hub_download, login
from safetensors.torch import load_file

from classes import GenerationRequest

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

                # pipeline = FluxPipeline.from_pretrained(
                #     self.repo_id,
                #     transformer=transformer,
                #     # config_path=config_path,
                #     torch_dtype=torch.float16,
                #     device_map="balanced"  # Only valid option for Flux in diffusers
                # )

            
                # # Login to HuggingFace if token is available
                # if self.hf_token:
                #     _log.info("Logging in to HuggingFace Hub")
                #     login(token=self.hf_token)
                
                # # Download text encoders if needed
                # _log.info("Downloading text encoders")
                # self.clip_l_path = hf_hub_download(
                #     repo_id="comfyanonymous/flux_text_encoders", 
                #     filename="clip_l.safetensors"
                # )
                # self.t5_fp8_path = hf_hub_download(
                #     repo_id="comfyanonymous/flux_text_encoders", 
                #     filename="t5xxl_fp8_e4m3fn.safetensors"
                # )
                # _log.info(f"Text encoders downloaded: {self.clip_l_path}, {self.t5_fp8_path}")
                

                # checkpoint = load_file(model_path)

                # # Optionally, if the checkpoint combines multiple component weights,
                # # split the weights by module name. Adjust the key names as necessary.
                # unet_state_dict = {
                #     key[len("unet."):]: value
                #     for key, value in checkpoint.items() if key.startswith("unet.")
                # }

                # text_encoder_state_dict = {
                #     key[len("text_encoder."):]: value
                #     for key, value in checkpoint.items() if key.startswith("text_encoder.")
                # }

                # vae_state_dict = {
                #     key[len("vae."):]: value
                #     for key, value in checkpoint.items() if key.startswith("vae.")
                # }

                # # Load the pipeline configuration from a local directory.
                # # The directory should contain the necessary config files.
                # model_config_dir = "./flux_model_config"  # Update this to your actual config folder

                # # Create the pipeline instance from the config directory
                # pipeline = FluxPipeline.from_pretrained(model_config_dir, local_files_only=True)

                # # Now manually load the weights into the respective submodules.
                # # (If your FluxPipeline structure is different, adjust accordingly.)
                # pipeline.unet.load_state_dict(unet_state_dict)
                # pipeline.text_encoder.load_state_dict(text_encoder_state_dict)
                # pipeline.vae.load_state_dict(vae_state_dict)


                # -----------------------------------------------------------------------------------------------------------------------------------------------------

                # transformer = FluxTransformer2DModel.from_single_file(model_path)

                # pipeline = FluxTransformer2DModel.from_single_file(
                #     model_path,
                #     config_path=config_path,
                #     torch_dtype=torch.float16,
                #     device_map="balanced"  # Only valid option for Flux in diffusers
                # )

                # -----------------------------------------------------------------------------------------------------------------------------------------------------

                
                # Load text encoders (optional, may not be necessary)
                # _log.info("Loading text encoders into pipeline")
                # clip_weights = load_file(self.clip_l_path)
                # t5_weights = load_file(self.t5_fp8_path)
                # pipeline.text_encoder.load_state_dict(clip_weights, strict=False)
                # pipeline.text_encoder_2.load_state_dict(t5_weights, strict=False)
                
                # pipeline = StableDiffusionXLPipeline.from_single_file(
                #     model_path,
                #     torch_dtype=torch.float16,
                #     variant="fp16",
                #     safety_checker=None,
                #     use_safetensors=True,
                # )
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
            return tuple(self.convert_lists_to_tuples(v) for v in data)
        else:
            return data

    def predict(self, payload: GenerationRequest, callback_func_base: callable, callback_func_refiner: callable = None) -> None:
        # Extract common parameters from the request
        prompt = payload.prompt
        #negative_prompt = getattr(payload, 'negative_prompt', None)
        height = getattr(payload, 'height', 512)  # Changed from 1024 to 512
        width = getattr(payload, 'width', 512)    # Changed from 1024 to 512
        num_inference_steps = getattr(payload, 'num_inference_steps', 4)  # Flux works well with fewer steps
        guidance_scale = getattr(payload, 'guidance_scale', 3.5)
        
        # Set up a fixed seed if requested
        seed = getattr(payload, 'seed', None)
        generator = torch.Generator("cpu")
        if seed is not None:
            generator = generator.manual_seed(seed)
        
        # Log the parameters
        _log.info(f"Generating image with Flux: prompt='{prompt}', height={height}, width={width}, steps={num_inference_steps}")
        
        # Create a custom callback to wrap the provided one and add debugging
        def debug_callback_wrapper(_pipe, step, _timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
            _log.info(f"Flux latents shape at step {step}: {latents.shape}, dtype: {latents.dtype}")
            return callback_func_base(_pipe, step, _timestep, callback_kwargs)
        
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