import base64
import io
import abc
from typing import Any

import torch
from PIL import Image


class LatentsProcessor(abc.ABC):
    """Base interface for processing latents into preview images."""
    
    @abc.abstractmethod
    def process_latents(self, pipeline: Any, latents: torch.Tensor) -> Image.Image:
        """
        Process latents to generate a PIL Image.
        
        Args:
            pipeline: The model pipeline
            latents: The latent representation
            
        Returns:
            PIL Image generated from the latents
        """
        pass
    
    @staticmethod
    def encode_image(image: Image.Image, resize_factor: float = 0.5) -> str:
        """
        Encode PIL image to base64 string with optional resizing.
        
        Args:
            image: PIL Image
            resize_factor: Factor to resize the image by (default: 0.5)
            
        Returns:
            Base64 encoded string of the image
        """
        # Resize the image to save on bandwidth
        if resize_factor != 1.0:
            width, height = image.size
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            image = image.resize((new_width, new_height))
        
        # Convert to base64
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        image_data = img_bytes.read()
        return base64.b64encode(image_data).decode("utf-8")